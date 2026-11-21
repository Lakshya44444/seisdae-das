"""Re-extract the published second apertures and check their geometry."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..data import ROOT, WIDTH, fetch, load

DEST = ROOT / "data/multi_v3"


def build(folder):
    events, _ = load(ROOT / "data/prepared")
    manifest = pd.read_csv(DEST / "manifest.csv").set_index("event_id", verify_integrity=True)
    if set(manifest.index) != set(events.event_id):
        raise ValueError("Second-aperture manifest and catalog differ")
    signals, backgrounds, picks = [], [], []
    for e in events.itertuples():
        row = manifest.loc[e.event_id]
        first, second = int(row.first_row), int(row.second_row)
        if first != int(e.first_h5_row) or abs(second - first) < WIDTH:
            raise ValueError(f"Invalid or overlapping apertures for {e.event_id}")
        block = slice(second, second + WIDTH)
        ids = pd.read_csv(folder / f"dasinfo_{e.array}.csv")["index"].to_numpy()[block]
        with h5py.File(folder / "phase_P" / f"{e.event_id}.h5") as f:
            g = f["data/P"]
            raw = g["data"][block].astype(float)
            shifts = g["shift_index"][block].astype(int)
        if len(ids) != WIDTH or not np.all(np.diff(ids) == 1) or np.ptp(shifts) > 8:
            raise ValueError(f"Invalid DAS geometry for {e.event_id}")
        delay = shifts - int(np.median(shifts))
        offsets = np.arange(70) - delay[:, None]
        for base, output in [(86, signals), (9, backgrounds)]:
            x = np.take_along_axis(raw, offsets + base, axis=1)
            output.append(x - x.mean(1, keepdims=True))
        picks.append(14 + delay)
    np.savez_compressed(DEST / "aperture2.npz", reference=signals, noise=backgrounds, nominal_p=picks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    args = parser.parse_args()
    build(args.raw or fetch(ROOT / "data/raw"))
