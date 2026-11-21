"""Compare a rerun's scientific results with the packaged benchmark."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT

KEYS = {
    "per_event.csv": ["array", "event_id", "level", "method"],
    "summary.csv": ["level", "method"],
    "by_array.csv": ["array", "level", "method"],
    "paired_gain.csv": ["level", "baseline"],
}


def verify(actual, expected):
    for name, keys in KEYS.items():

        def read(path):
            df = (
                pd.read_csv(path / name)
                .set_index(keys, verify_integrity=True)
                .sort_index()
            )
            return df.drop(columns="ms_per_trace", errors="ignore").sort_index(axis=1)

        a, b = read(actual), read(expected)
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-5, rtol=1e-5)
        print(f"{name}: {len(a)} matching rows")
    with np.load(actual / "example.npz") as a, np.load(expected / "example.npz") as b:
        if set(a.files) != set(b.files):
            raise ValueError("The example arrays have different method names.")
        for key in a.files:
            np.testing.assert_allclose(a[key], b[key], atol=1e-5, rtol=1e-5)
    print("Example waveforms match.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--expected", type=Path)
    version = p.add_mutually_exclusive_group()
    version.add_argument("--v2", action="store_true")
    version.add_argument("--v3", action="store_true")
    a = p.parse_args()
    if a.v3:
        from scripts.verify_v3 import compare
        compare(a.run, a.expected or ROOT / "results_v3")
    elif a.v2:
        from .v2.check import verify as verify_v2

        verify_v2(a.run, a.expected or ROOT / "results_v2")
    else:
        verify(a.run, a.expected or ROOT / "results")
