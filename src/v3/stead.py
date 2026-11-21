"""Regenerate the pilot STEAD power spectrum using the pinned HDF5 source and trace manifest."""
import hashlib
import json
import urllib.request

import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch

from ..data import ROOT

OUT = ROOT / "results_v3"


def run():
    item = json.loads((OUT / "stead_source.json").read_text())
    folder = ROOT / "data/conventional"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "100samples.hdf5"
    if not path.exists():
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(item["url"], part)
        if hashlib.sha256(part.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("STEAD download checksum mismatch")
        part.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("STEAD file does not match the pinned revision")
    manifest = pd.read_csv(ROOT / "results/source_training/stead_manifest.csv")
    with h5py.File(path) as h5:
        traces = np.stack([h5["data"][name][:, 2] for name in manifest.trace_name])
    hz, psd = welch(traces, fs=100, nperseg=70, nfft=256, axis=-1)
    pd.DataFrame({"hz": hz, "pilot_stead": (psd / psd.sum(1, keepdims=True)).mean(0)}).to_csv(
        OUT / "stead_spectrum.csv", index=False)


if __name__ == "__main__":
    run()
