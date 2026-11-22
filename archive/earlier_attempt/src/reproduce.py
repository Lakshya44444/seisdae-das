"""Predict on a released RN.BAVN Z trace and optionally compare with upstream."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import FS, ROOT
from .neural import PATH, N, denoise, model


def run(upstream=None):
    with np.load(ROOT / "data/example/bug2019gbbo_Z.npz") as d:
        x, pick = d["data"][:N], float(d["itp"]) / FS
    net = model()
    y = denoise(net, x[None])[0]
    summary = {"example": "bug2019gbbo_Z", "samples": len(x), "catalog_p_s": pick}
    if upstream:
        sys.path.insert(0, str(Path(upstream).resolve()))
        from prediction import predict

        original = predict(
            str(PATH), str(ROOT / "models/gr_mixed_stft.config"), [x], loaded_model=net
        )[0][0, :, 0]
        scale = np.max(np.abs(original))
        error = np.max(np.abs(y - original)) / scale
        summary["upstream_relative_peak_error"] = float(error)
        if error > 1e-4:
            raise ValueError(f"Adapter comparison failed: {error}")
    t = np.arange(N) / FS
    fig, ax = plt.subplots(
        2, 1, figsize=(9, 4), sharex=True, sharey=True, layout="constrained"
    )
    for a, v, title in zip(ax, (x - x.mean(), y), ("Recorded", "Released seisDAE")):
        a.plot(t, v, lw=0.6, color="#16586a")
        a.axvline(pick, color="#aa521f", ls="--", label="P pick")
        a.set_ylabel(title)
    ax[0].legend(loc="upper right")
    ax[-1].set_xlabel("Seconds from start")
    fig.suptitle("RN.BAVN · bug2019gbbo · vertical component")
    fig.savefig(ROOT / "figures/01_reproduction.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--upstream", help="Optional path to the pinned JanisHe/seisDAE checkout"
    )
    args = p.parse_args()
    run(args.upstream)
