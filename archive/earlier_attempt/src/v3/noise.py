"""Score matched Ridgecrest targets with independently recorded continuous backgrounds."""
import argparse
import hashlib
import json
import requests
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt

from ..data import LEVELS, ROOT, load, mix
from ..v2.metrics import all_scores
from ..v2.sensitivity import Models
from .experiments import CFG, OUT


def sources():
    return [ROOT / "data/noise_v2", *sorted((ROOT / "data/noise_v3").glob("20*"))]


def fetch(folder):
    manifest = json.loads((folder / "continuous.json").read_text())
    def read(segment):
        first, last = segment["start_byte"], segment["end_byte"]
        with requests.get(manifest["url"] + f"?seisdae_range={first}", headers={"Range": f"bytes={first}-{last}"},
                          stream=True, timeout=60) as response:
            if (response.status_code != 206 or
                    not response.headers.get("Content-Range", "").startswith(f"bytes {first}-{last}/")):
                raise IOError("Server did not honor byte range")
            raw = response.raw.read(last - first + 2)
        if hashlib.sha256(raw).hexdigest() != segment["sha256"]:
            raise IOError("SCEDC trace checksum mismatch")
        return np.frombuffer(raw, dtype="<f4")
    with ThreadPoolExecutor(max_workers=4) as pool:
        raw = np.stack(list(pool.map(read, manifest["segments"]))).astype(float)
    y = np.gradient(detrend(raw, axis=-1), 1 / 250, axis=-1)
    y -= np.median(y, axis=0, keepdims=True)
    y = sosfiltfilt(butter(3, [1, 10], fs=250, btype="bandpass", output="sos"), y, axis=-1)
    result = resample_poly(y, 2, 5, axis=-1).astype("float32")
    np.savez_compressed(folder / "continuous.npz", background=result)


def run():
    events, pairs = load(ROOT / "data/prepared")
    targets = events[(events.split == "test") & (events.array == "ridgecrest")]
    models, rows = Models(), []
    for folder in sources():
        manifest = json.loads((folder / "continuous.json").read_text())
        hour = manifest["url"].rsplit("/", 1)[1].split(".")[0]
        with np.load(folder / "continuous.npz") as block:
            data = block["background"].astype(float)
        for i, e in targets.iterrows():
            s, pick = pairs["reference"][i], pairs["nominal_p"][i]
            for draw, start in enumerate(CFG["external"]["starts"]):
                n = data[:, start:start+70]
                n = n - n.mean(1, keepdims=True)
                for level in LEVELS:
                    x = mix(s, n, level)
                    for method, y in models.both(x).items():
                        rows.append(dict(event_id=e.event_id, recording=hour, draw=draw, level=level,
                                         method=method, **all_scores(s, x, y, pick)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "external_events.csv", index=False)
    d.groupby(["recording","level","method"], as_index=False)[["gain_db","si_gain_db","cc"]].mean().to_csv(
        OUT / "external_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Refresh all three recorded cuts against their SHA-256 digests")
    if parser.parse_args().fetch:
        for folder in sources():
            fetch(folder)
    else:
        run()
