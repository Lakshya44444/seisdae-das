"""Build small, auditable alternate background blocks for the original validation earthquakes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt
from ..data import ROOT, load, fetch, WIDTH

DEST = ROOT / "data/noise_v2"
URL = "https://scedc-pds.s3.us-west-2.amazonaws.com/Ridgecrest_DAS/SEG-Y/hourly/2020062417.segy"


def other_blocks(raw):
    events, pairs = load(ROOT / "data/prepared")
    data = {}
    rows = []
    for i, e in events[events.split == "validation"].iterrows():
        first = int(e.first_h5_row)
        ids = pd.read_csv(raw / f"dasinfo_{e['array']}.csv")["index"].to_numpy()
        with h5py.File(raw / "phase_P" / f"{e.event_id}.h5") as h:
            group = h["data/P"]
            sets = group["shift_index"][:].astype(int)
            for offset in range(WIDTH, len(ids), WIDTH):
                j = (first + offset) % (len(ids) // WIDTH * WIDTH)
                ix = slice(j, j + WIDTH)
                if j == first or np.any(np.diff(ids[ix]) != 1) or np.ptp(sets[ix]) > 8:
                    continue
                shifts = sets[ix] - int(np.median(sets[ix]))
                noise = np.take_along_axis(
                    group["data"][ix].astype(float),
                    9 + np.arange(70) - shifts[:, None],
                    axis=1,
                )
                noise -= noise.mean(1, keepdims=True)
                if np.isfinite(noise).all() and np.all(noise.std(1) > 0):
                    break
            else:
                raise ValueError(f"No usable disjoint block for {e.event_id}")
        data[str(e.event_id)] = noise
        rows.append(
            dict(
                event_id=e.event_id,
                array=e["array"],
                signal_first_row=first,
                noise_first_row=j,
                noise_first_channel=ids[j],
            )
        )
    DEST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEST / "other_blocks.npz", **data)
    pd.DataFrame(rows).to_csv(DEST / "other_blocks.csv", index=False)


def continuous():
    # Fixed in advance: 24 neighboring trace rows, 30 s beginning at 17:10:00 UTC.
    count = 30 * 250

    def read(ch):
        start = 3600 + ch * (240 + 900000 * 4) + 240 + 600 * 250 * 4
        end = start + count * 4 - 1
        req = urllib.request.Request(
            URL + f"?seisdae_range={start}", headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(req, timeout=45) as response:
            if (
                response.status != 206
                or response.headers["Content-Range"]
                != f"bytes {start}-{end}/4500303600"
            ):
                raise IOError("Unexpected byte range; refusing a full download")
            body = response.read(count * 4 + 1)
        if len(body) != count * 4:
            raise IOError("Incomplete trace")
        return np.frombuffer(body, dtype="<f4"), dict(
            trace_row=ch,
            start_byte=start,
            end_byte=end,
            sha256=hashlib.sha256(body).hexdigest(),
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        parts = list(pool.map(read, range(400, 424)))
    raw = np.stack([x for x, row in parts]).astype(float)
    if not np.isfinite(raw).all() or np.any(raw.std(1) == 0):
        raise ValueError("Invalid raw SEG-Y values")
    # Continuous raw phase is proportional to strain. Differentiate, remove median
    # across these channels, band-limit and anti-alias before 250 -> 100 Hz.
    y = np.gradient(detrend(raw, axis=-1), 1 / 250, axis=-1)
    y -= np.median(y, axis=0, keepdims=True)
    y = sosfiltfilt(
        butter(3, [1, 10], fs=250, btype="bandpass", output="sos"), y, axis=-1
    )
    y = resample_poly(y, 2, 5, axis=-1)
    DEST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEST / "continuous.npz", background=y.astype("float32"))
    (DEST / "continuous.json").write_text(
        json.dumps(
            dict(
                url=URL,
                start_utc="2020-06-24T17:10:00Z",
                duration_s=30,
                source_hz=250,
                output_hz=100,
                trace_rows=[400, 423],
                sample_start=150000,
                bytes_fetched=raw.size * 4,
                segments=[r for _, r in parts],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--continuous", action="store_true")
    args = parser.parse_args()
    if args.continuous:
        continuous()
    else:
        other_blocks(args.raw or fetch(ROOT / "data/raw"))
