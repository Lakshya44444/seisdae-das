"""Before/after pictures of recorded test windows (no added noise).

  python -m src.examples

Shows the three largest test-day earthquakes without an overlapping event and
two noise windows drawn with a fixed seed, as raw data, DAS-trained seisDAE and
seisDAE followed by f-k (frozen settings). Each panel is scaled to its own
99th percentile, so the pictures compare shape, not amplitude.
Writes results/figures/examples.png.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .build import load_gather, manifest
from .das import CFG, geometry
from .evaluate import FIG, OUT, Runner
from .methods import ROOT, fk

FS = 100


def main():
    m = manifest()
    overlap = m["overlap"].fillna(0)
    eq = m[(m.kind != "noise") & (m.split == "test") & (overlap == 0)].nlargest(3, "magnitude")
    noise = m[(m.kind == "noise") & (m.split == "test")].sample(2, random_state=CFG["seed"])
    settings = json.loads((OUT / "run.json").read_text())["settings"]
    runner = Runner(ROOT / "models/das_seisdae.h5", geometry()[2])
    rows = [(r.id, f"M{r.magnitude:.1f}, {r.dist_km:.0f} km") for r in eq.itertuples()] + \
           [(r.id, "noise window") for r in noise.itertuples()]
    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 2.6 * len(rows)), sharex=True, layout="constrained")
    for i, (name, label) in enumerate(rows):
        x = load_gather(name)
        das = runner.das(x)
        panels = (("Raw", x), ("seisDAE (DAS-trained)", das),
                  ("seisDAE (DAS) + f-k", fk(das, runner.dx, settings["seisDAE (DAS) + f-k"])))
        for j, (title, y) in enumerate(panels):
            clip = np.percentile(np.abs(y), 99) or 1.0
            axes[i, j].imshow(y, aspect="auto", cmap="seismic", vmin=-clip, vmax=clip,
                              extent=(0, y.shape[1] / FS, y.shape[0], 0), interpolation="none")
            axes[i, j].set_title(f"{title} | {label}", fontsize=8)
        axes[i, 0].set_ylabel("Channel")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "examples.png", dpi=130)
    print(FIG / "examples.png")


if __name__ == "__main__":
    main()
