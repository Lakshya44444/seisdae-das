"""Read CaltechDATA v1. All row and sample indices are zero based."""

import argparse
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib
import tarfile
import requests
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FS, WIDTH, SEED = 100, 24, 2026
LEVELS = (-5, 0, 5, 10)
DOI = "https://data.caltech.edu/records/n47vy-s0s65/files/DASPphaseData.tar?download=1"
DIGEST = "c50468d83401790c9e78f0f21d0798fe"


def fetch(root):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "DASPphaseData.tar"
    if not archive.exists():
        # A range request works on mirrors that refuse or truncate a full TAR.
        # Verify Content-Range and the published whole-archive checksum.
        url = DOI.split("?")[0]
        step = 4 * 1024 * 1024
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            staging = Path(temporary)
            with requests.get(url, headers={"Range":"bytes=0-0"}, stream=True, timeout=60) as response:
                response.raise_for_status()
                if response.status_code == 200:
                    with (staging/"archive").open("wb") as output:
                        shutil.copyfileobj(response.raw, output)
                    total = 0
                else:
                    match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range",""))
                    if response.status_code != 206 or not match:
                        raise IOError("Cannot establish Caltech archive size")
                    total = int(match[1])
            def part(start):
                stop=min(total,start+step)-1
                for retry in range(4):
                    try:
                        with requests.get(url+f"?download=1&part={start}",headers={"Range":f"bytes={start}-{stop}"},
                                          stream=True,timeout=90) as response:
                            response.raise_for_status()
                            if response.headers.get("Content-Range") != f"bytes {start}-{stop}/{total}":
                                raise IOError("Unexpected Caltech range")
                            data=response.raw.read(stop-start+2)
                        if len(data)!=stop-start+1:
                            raise IOError("Incomplete Caltech range")
                        (staging/str(start)).write_bytes(data)
                        return
                    except OSError:
                        if retry==3:
                            raise
                        time.sleep(2**retry)
            if total:
                starts=range(0,total,step)
                with ThreadPoolExecutor(max_workers=5) as pool:
                    list(pool.map(part,starts))
                with (staging/"archive").open("wb") as output:
                    for start in starts:
                        with (staging/str(start)).open("rb") as chunk:
                            shutil.copyfileobj(chunk,output)
            with (staging/"archive").open("rb") as stream:
                if hashlib.file_digest(stream,"md5").hexdigest()!=DIGEST:
                    raise ValueError("Caltech archive checksum mismatch")
            (staging/"archive").replace(archive)
    with archive.open("rb") as f:
        if hashlib.file_digest(f, "md5").hexdigest() != DIGEST:
            raise ValueError(f"Checksum mismatch. Remove {archive} and retry.")
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    return root / "CaltechDATA"


def select_events(folder):
    """Stratify by array and split earthquakes before examining their channels."""
    rng = np.random.default_rng(SEED)
    rows = []
    for site in ("longvalley", "ridgecrest"):
        catalog = pd.read_csv(folder / f"catalog_{site}.csv")
        chosen = catalog.sort_values(
            ["magnitude", "event_id"], ascending=[False, True]
        ).head(18)
        if len(chosen) != 18:
            raise ValueError(f"Expected at least 18 events for {site}")
        chosen = chosen.iloc[rng.permutation(18)].copy()
        chosen["array"] = site
        chosen["split"] = ["train"] * 12 + ["validation"] * 3 + ["pilot"] * 3
        # The previous study used these 18 events from each array. Freeze the
        # published fit and reserve the entire previously unused catalog for testing.
        holdout = catalog.loc[~catalog.event_id.isin(chosen.event_id)].copy()
        holdout = holdout.sort_values("event_id")
        holdout["array"] = site
        holdout["split"] = "test"
        rows.extend([chosen, holdout])
    events = pd.concat(rows, ignore_index=True)
    if events.event_id.duplicated().any():
        raise ValueError("Repeated event ID across catalogs")
    return events


def prepare(folder, out):
    out.mkdir(parents=True, exist_ok=True)
    events = select_events(folder)
    inventory = []
    for site in ("longvalley", "ridgecrest"):
        catalog = pd.read_csv(folder / f"catalog_{site}.csv")
        for eid in catalog.event_id:
            with h5py.File(folder / "phase_P" / f"{eid}.h5") as f:
                d = f["data/P/data"]
                inventory.append((site, eid, *d.shape, float(d.attrs["dt"])))
    pd.DataFrame(
        inventory, columns=["array", "event_id", "rows", "samples", "dt_s"]
    ).to_csv(out / "inventory.csv", index=False)
    signals, noises, raws, picks = [], [], [], []
    for i, event in events.iterrows():
        ids = pd.read_csv(folder / f"dasinfo_{event['array']}.csv")["index"].to_numpy()
        with h5py.File(folder / "phase_P" / f"{event.event_id}.h5") as f:
            group = f["data/P"]
            d = group["data"]
            if d.shape != (len(ids), 200) or not np.isclose(d.attrs["dt"], 1 / FS):
                raise ValueError(
                    f"Unexpected geometry or timing for event {event.event_id}"
                )
            snr = group["snr"][:]
            shifts = group["shift_index"][:].astype(int)
            # Fixed disjoint blocks; contiguous physical IDs avoid crossing spatial gaps.
            best, start = -np.inf, None
            for j in range(0, len(ids) - WIDTH + 1, WIDTH):
                sl = slice(j, j + WIDTH)
                if np.any(np.diff(ids[sl]) != 1) or np.ptp(shifts[sl]) > 8:
                    continue
                score = np.median(snr[sl])
                if np.isfinite(score) and score > best:
                    best, start = score, j
            if start is None:
                raise ValueError(f"No eligible aperture for event {event.event_id}")
            sl = slice(start, start + WIDTH)
            raw = d[sl].astype(np.float64)
            delay = shifts[sl] - int(np.median(shifts[sl]))
            # Translate per-channel aligned P snippets to a common relative clock.
            offsets = np.arange(70)[None, :] - delay[:, None]
            signal = np.take_along_axis(raw, offsets + 86, axis=1)
            noise = np.take_along_axis(raw, offsets + 9, axis=1)
            for block in (signal, noise):
                block -= block.mean(axis=1, keepdims=True)
                if not np.isfinite(block).all() or np.any(
                    np.sum(block**2, axis=1) == 0
                ):
                    raise ValueError(
                        f"Empty or non-finite trace in event {event.event_id}"
                    )
            signals.append(signal)
            noises.append(noise)
            raws.append(raw)
            picks.append(14 + delay)  # nominal clip centre in the analysis window
            events.loc[i, ["first_h5_row", "first_channel", "median_source_snr"]] = (
                start,
                ids[start],
                best,
            )
    np.savez_compressed(
        out / "pairs.npz", reference=signals, noise=noises, raw=raws, nominal_p=picks
    )
    events.to_csv(out / "events.csv", index=False)


def load(folder):
    events = pd.read_csv(folder / "events.csv")
    with np.load(folder / "pairs.npz") as z:
        return events, {key: z[key] for key in z.files}


def mix(ref, noise, level):
    """One scale per aperture preserves relative channel amplitudes and moveout."""
    ps, pn = np.sum(ref**2), np.sum(noise**2)
    if min(ps, pn) <= 0:
        raise ValueError("Both signal and noise need positive energy")
    return ref + noise * np.sqrt(ps / (pn * 10.0 ** (level / 10)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, help="Extracted CaltechDATA directory. Omit to download."
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/prepared")
    args = parser.parse_args()
    prepare(args.raw or fetch(ROOT / "data/raw"), args.out)
