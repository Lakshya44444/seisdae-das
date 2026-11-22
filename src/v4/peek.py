"""Save a picture of downloaded gathers to check window placement by eye.

  python -m src.v4.peek eq_38595986 eq_39492720

Writes figures_v4/peek_<id>.png with the predicted P (10 s) and S times.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .build import load_gather, manifest
from .das import ROOT, SUFFIX

FS = 100


def main(ids):
    m = manifest().set_index("id")
    out = ROOT / f"figures_v4{SUFFIX}"
    out.mkdir(exist_ok=True)
    for name in ids:
        x, r = load_gather(name), m.loc[name]
        clip = np.percentile(np.abs(x), 99)
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[2, 1], layout="constrained")
        ax[0].imshow(x, aspect="auto", cmap="seismic", vmin=-clip, vmax=clip,
                     extent=(0, x.shape[1] / FS, x.shape[0], 0), interpolation="none")
        ax[1].plot(np.arange(x.shape[1]) / FS, x[x.shape[0] // 2], lw=0.5, color="#16586a")
        for a in ax:
            if np.isfinite(r.get("itp", np.nan)):
                a.axvline(r.itp / FS, color="#aa521f", ls="--", lw=1, label="predicted P")
                a.axvline(r.its / FS, color="#2a7a2a", ls="--", lw=1, label="predicted S")
        ax[0].set_ylabel("Channel")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Middle channel")
        ax[1].legend(loc="upper right", fontsize=8)
        title = f"{name}  {r.kind}  {r.split}"
        if np.isfinite(r.get("magnitude", np.nan)):
            title += f"  M{r.magnitude:.1f}  {r.dist_km:.0f} km  median SNR {r.median_snr_db:.1f} dB"
        fig.suptitle(title)
        path = out / f"peek_{name}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main(sys.argv[1:])
