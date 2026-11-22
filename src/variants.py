"""Compare seisDAE variants on the detection test: released, from scratch, fine-tuned (several seeds).

  python -m src.variants               score every model, then statistics
  python -m src.variants --stats-only  statistics from results/variants_rows.csv

Models: models/gr_mixed_stft (released), models/das_seisdae (from scratch,
seed 2026) and every models/variants/<name>.h5 (names such as scratch_s2027 or
finetune_s2026). No variant has a setting chosen on test days: training uses the
original trainers with their default stopping rules.

Each model is scored on the same test windows as the detection test (73
earthquake windows, 210 noise windows). The baselines (raw data, bandpass
1-10 Hz, f-k 1000 m/s) are read from results/detection_tuning_rows.csv, so
they are not scored again. Paired window bootstrap, as in detection_stats.
Writes results/variants_rows.csv, results/variants_auc.csv and
results/variants_summary.csv.
"""

import re

import numpy as np
import pandas as pd

from .build import load_gather
from .das import CFG
from .detection_stats import auc
from .evaluate import OUT, detection_stat
from .methods import ROOT, SeisDAE

BASELINES = ("Noisy", "Bandpass 1-10 Hz", "f-k 1000 m/s")


def models():
    found = {"released": ROOT / "models/gr_mixed_stft.h5", "scratch_s2026": ROOT / "models/das_seisdae.h5"}
    for h5 in sorted((ROOT / "models/variants").glob("*.h5")):
        found[h5.stem] = h5
    return found


def family(name):
    return re.sub(r"_s\d+$", "", name)


def score():
    cached = pd.read_csv(OUT / "detection_tuning_rows.csv")
    base = cached[cached.method.isin(BASELINES)]
    windows = base[["window", "label"]].drop_duplicates()
    rows = []
    for name, h5 in models().items():
        net = SeisDAE(h5, h5.with_suffix(".config"))
        for k, r in enumerate(windows.itertuples()):
            rows.append(dict(window=r.window, label=r.label, method=name,
                             stat_db=detection_stat(net(load_gather(r.window)))))
            print(f"{name} {k + 1}/{len(windows)}", end="\r")
        print()
    d = pd.concat([base[["window", "label", "method", "stat_db"]], pd.DataFrame(rows)], ignore_index=True)
    d.to_csv(OUT / "variants_rows.csv", index=False)


def main(rescore=True):
    if rescore:
        score()
    d = pd.read_csv(OUT / "variants_rows.csv")

    eq = d[d.label == "earthquake"].pivot(index="window", columns="method", values="stat_db")
    nz = d[d.label == "noise"].pivot(index="window", columns="method", values="stat_db")
    methods = list(BASELINES) + [n for n in d.method.unique() if n not in BASELINES]
    rng = np.random.default_rng(CFG["seed"])
    draws = CFG["evaluation"]["bootstrap_draws"]
    E, N = eq[methods].to_numpy(), nz[methods].to_numpy()
    boot = np.empty((draws, len(methods)))
    for i in range(draws):
        ie, inn = rng.integers(0, len(E), len(E)), rng.integers(0, len(N), len(N))
        for j in range(len(methods)):
            boot[i, j] = auc(E[ie, j], N[inn, j])
    point = {mth: auc(eq[mth], nz[mth]) for mth in methods}
    table = []
    for j, mth in enumerate(methods):
        r = dict(model=mth, family="baseline" if mth in BASELINES else family(mth), auc=point[mth],
                 lo=np.quantile(boot[:, j], 0.025), hi=np.quantile(boot[:, j], 0.975))
        for b in BASELINES + ("released",):
            if mth not in BASELINES and mth != b:
                diff = boot[:, j] - boot[:, methods.index(b)]
                r[f"minus {b}"] = point[mth] - point[b]
                r[f"minus {b} lo"], r[f"minus {b} hi"] = np.quantile(diff, [0.025, 0.975])
        table.append(r)
    table = pd.DataFrame(table)
    table.to_csv(OUT / "variants_auc.csv", index=False)
    summary = table[table.family != "baseline"].groupby("family").agg(
        runs=("auc", "size"), mean_auc=("auc", "mean"), min_auc=("auc", "min"), max_auc=("auc", "max"),
        mean_minus_bandpass=("minus Bandpass 1-10 Hz", "mean"), mean_minus_fk=("minus f-k 1000 m/s", "mean"),
        mean_minus_released=("minus released", "mean"))
    summary.to_csv(OUT / "variants_summary.csv")
    print(f"{len(eq)} earthquake windows, {len(nz)} noise windows")
    cols = ["model", "auc", "lo", "hi", "minus Noisy", "minus Bandpass 1-10 Hz", "minus Bandpass 1-10 Hz lo",
            "minus Bandpass 1-10 Hz hi", "minus f-k 1000 m/s", "minus released", "minus released lo",
            "minus released hi"]
    print(table[[c for c in cols if c in table]].round(3).to_string(index=False))
    print()
    print(summary.round(3).to_string())


if __name__ == "__main__":
    import sys
    main(rescore="--stats-only" not in sys.argv)
