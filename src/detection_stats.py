"""ROC, AUC and paired bootstrap for the detection test.

  python -m src.detection_stats [path/to/detection_rows.csv]

Reads detection_rows.csv written by src.evaluate. Earthquake windows and
noise windows are both resampled in every bootstrap draw, so the intervals
include the uncertainty of the noise-based thresholds. The same resampled
windows are used for every method, so the differences are paired.
Writes detection_auc.csv, detection_paired.csv, detection_paired_byday.csv\nand roc.png. The day bootstrap resamples whole test days as a robustness check.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .das import CFG

DRAWS = CFG["evaluation"]["bootstrap_draws"]
ORDER = ["Noisy", "Bandpass", "f-k", "seisDAE (released)", "seisDAE (DAS-trained)", "seisDAE (DAS) + f-k"]
PAIRS = [("seisDAE (DAS-trained)", "Noisy"), ("seisDAE (DAS-trained)", "f-k"),
         ("seisDAE (DAS-trained)", "seisDAE (released)"), ("seisDAE (DAS) + f-k", "Noisy"),
         ("seisDAE (DAS) + f-k", "f-k"), ("f-k", "Noisy")]


def auc(e, n):
    """P(earthquake statistic > noise statistic), ties counted as one half."""
    e, n = np.asarray(e)[:, None], np.asarray(n)[None, :]
    return float(np.mean((e > n) + 0.5 * (e == n)))


def detected(e, n, fa):
    thr = np.quantile(n, 1 - fa, method="higher")
    return float(np.mean(e > thr))


def tables(rows):
    eq = rows[rows.label == "earthquake"].pivot(index="window", columns="method", values="stat_db")
    nz = rows[rows.label == "noise"].pivot(index="window", columns="method", values="stat_db")
    methods = [m for m in ORDER if m in eq.columns]
    return eq[methods], nz[methods], methods


def bootstrap(eq, nz, methods, seed):
    rng = np.random.default_rng(seed)
    E, N = eq.to_numpy(), nz.to_numpy()
    out = {k: {m: [] for m in methods} for k in ("auc", "fa5", "fa10")}
    for _ in range(DRAWS):
        ie, inn = rng.integers(0, len(E), len(E)), rng.integers(0, len(N), len(N))
        for j, m in enumerate(methods):
            e, n = E[ie, j], N[inn, j]
            out["auc"][m].append(auc(e, n))
            out["fa5"][m].append(detected(e, n, 0.05))
            out["fa10"][m].append(detected(e, n, 0.10))
    return {k: {m: np.array(v) for m, v in d.items()} for k, d in out.items()}


def day_bootstrap(eq, nz, methods, days, seed):
    """Robustness check: resample whole test days (aftershocks and traffic cluster in time)."""
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(days.values())))
    eday = np.array([days[w] for w in eq.index])
    nday = np.array([days[w] for w in nz.index])
    E, N = eq.to_numpy(), nz.to_numpy()
    out = {m: [] for m in methods}
    for _ in range(DRAWS):
        pick = rng.choice(unique, size=len(unique), replace=True)
        ie = np.concatenate([np.flatnonzero(eday == d) for d in pick])
        inn = np.concatenate([np.flatnonzero(nday == d) for d in pick])
        if len(ie) == 0 or len(inn) == 0:
            continue
        for j, m in enumerate(methods):
            out[m].append(auc(E[ie, j], N[inn, j]))
    return {m: np.array(v) for m, v in out.items()}


def window_days(windows):
    """Test day of every window, from the manifest."""
    from .build import manifest

    m = manifest().set_index("id")
    return {w: m.loc[w, "start"].floor("D") for w in windows}


def main(path):
    path = Path(path)
    rows = pd.read_csv(path)
    eq, nz, methods = tables(rows)
    boot = bootstrap(eq, nz, methods, CFG["seed"])
    point = {m: dict(auc=auc(eq[m], nz[m]), fa5=detected(eq[m], nz[m], 0.05), fa10=detected(eq[m], nz[m], 0.10))
             for m in methods}
    summary = []
    for m in methods:
        r = dict(method=m, earthquakes=len(eq), noise_windows=len(nz))
        for k in ("auc", "fa5", "fa10"):
            r[k], (r[k + "_lo"], r[k + "_hi"]) = point[m][k], np.quantile(boot[k][m], [0.025, 0.975])
        summary.append(r)
    summary = pd.DataFrame(summary)
    paired = []
    for a, b in PAIRS:
        if a in methods and b in methods:
            for k in ("auc", "fa5", "fa10"):
                d = boot[k][a] - boot[k][b]
                lo, hi = np.quantile(d, [0.025, 0.975])
                paired.append(dict(comparison=f"{a} minus {b}", measure=k, difference=point[a][k] - point[b][k],
                                   lo=lo, hi=hi, share_draws_above_zero=float(np.mean(d > 0))))
    paired = pd.DataFrame(paired)
    out = path.parent
    days = window_days(list(eq.index) + list(nz.index))
    dboot = day_bootstrap(eq, nz, methods, days, CFG["seed"] + 1)
    byday = []
    for a, b in PAIRS:
        if a in methods and b in methods:
            d = dboot[a] - dboot[b]
            lo, hi = np.quantile(d, [0.025, 0.975])
            byday.append(dict(comparison=f"{a} minus {b}", measure="auc", difference=point[a]["auc"] - point[b]["auc"],
                              lo=lo, hi=hi, share_draws_above_zero=float(np.mean(d > 0)),
                              test_days=len(set(days.values()))))
    byday = pd.DataFrame(byday)
    byday.to_csv(out / "detection_paired_byday.csv", index=False)
    summary.to_csv(out / "detection_auc.csv", index=False)
    paired.to_csv(out / "detection_paired.csv", index=False)
    roc(eq, nz, methods, out / "roc.png")
    pd.set_option("display.width", 160)
    print(summary.round(3).to_string(index=False))
    print()
    print(paired.round(3).to_string(index=False))
    print()
    print("Robustness: whole test days resampled")
    print(byday.round(3).to_string(index=False))


def roc(eq, nz, methods, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5), layout="constrained")
    for m in methods:
        thresholds = np.r_[np.inf, np.sort(np.r_[eq[m], nz[m]])[::-1]]
        tpr = [np.mean(eq[m] > t) for t in thresholds]
        fpr = [np.mean(nz[m] > t) for t in thresholds]
        ax.plot(fpr, tpr, drawstyle="steps-post", label=f"{m} (AUC {auc(eq[m], nz[m]):.2f})")
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("False-alarm rate (test noise windows)")
    ax.set_ylabel("Detected test earthquakes")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "detection_rows.csv")
