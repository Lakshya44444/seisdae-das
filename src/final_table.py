"""Final comparison of every method, read only from results/*.csv (no data, no TensorFlow).

  python -m src.final_table

Synthetic test (SI-SDR gain, dB, 19 test earthquakes x 2 noise draws) and
detection test (ROC AUC, 73 earthquake / 210 noise windows), test days.
Baselines as tuned before: synthetic test bandpass 0.5-45 Hz and f-k 250 m/s
(chosen by SI-SDR on validation days); detection bandpass 1-10 Hz and f-k
1000 m/s (chosen by AUC on the 29 non-test days).

Multi-seed models are averaged per event over their seeds before the paired
bootstrap, so "U-Net v1 minus seisDAE fine-tuned" compares the two families.

Writes results/final_synthetic.csv, results/final_paired.csv, results/final_detection.csv.
"""

import numpy as np
import pandas as pd

from .das import CFG
from .evaluate import E, OUT, bootstrap

LEVELS = [-10, -5, 0, 5, 10]

FAMILIES = {
    "Bandpass": ["Bandpass"],
    "f-k": ["f-k"],
    "seisDAE released": ["released"],
    "seisDAE from scratch (3 seeds)": ["scratch_s2026", "scratch_s2027", "scratch_s2028"],
    "seisDAE fine-tuned (3 seeds)": ["finetune_s2026", "finetune_s2027", "finetune_s2028"],
    "jDAS released": ["jdas_pretrained"],
    "jDAS fine-tuned": ["jdas_finetune_s2026"],
    "jDAS from scratch": ["jdas_scratch_s2026"],
    "2-D U-Net v1 (3 seeds)": ["unet2d_s2026", "unet2d_s2027", "unet2d_s2028"],
    "2-D U-Net v2": ["unet2d_v2_s2026"],
    "2-D U-Net v3": ["unet2d_v3_s2026"],
    "2-D U-Net v4": ["unet2d_v4_s2026"],
}
DETECTION = {
    "Bandpass": ["Bandpass 1-10 Hz"],
    "f-k": ["f-k 1000 m/s"],
    "seisDAE released": ["released"],
}

PAIRS = [("2-D U-Net v1 (3 seeds)", b) for b in
         ("Bandpass", "f-k", "seisDAE released", "seisDAE from scratch (3 seeds)", "seisDAE fine-tuned (3 seeds)",
          "jDAS fine-tuned")]


def per_event():
    old = pd.read_csv(OUT / "synthetic_events.csv").rename(columns={"method": "model"})
    old = old[old.model.isin(["Bandpass", "f-k"])]
    parts = [old[["event", "level", "model", "si_gain_db"]]]
    for f in ("synthetic_seeds_events.csv", "synthetic_jdas_events.csv", "synthetic_unet2d_events.csv"):
        d = pd.read_csv(OUT / f)
        if "draw" in d:
            d = d.groupby(["event", "level", "model"], as_index=False).si_gain_db.mean()
        parts.append(d[["event", "level", "model", "si_gain_db"]])
    d = pd.concat(parts, ignore_index=True)
    d = d.groupby(["event", "level", "model"], as_index=False).si_gain_db.mean()
    wide = d.pivot_table(index=["event", "level"], columns="model", values="si_gain_db")
    fam = pd.DataFrame({name: wide[cols].mean(axis=1) for name, cols in FAMILIES.items()
                        if all(c in wide for c in cols)})
    return fam


def boot(x):
    return bootstrap(np.asarray(x, dtype=float), E["bootstrap_draws"], CFG["seed"])


def main():
    fam = per_event()
    rows, paired = [], []
    for name in fam:
        for lv in LEVELS:
            mean, lo, hi = boot(fam[name].xs(lv, level="level").dropna())
            rows.append(dict(method=name, level=lv, si_gain_db=mean, lo=lo, hi=hi))
    for a, b in PAIRS:
        for lv in LEVELS:
            diff = (fam[a] - fam[b]).xs(lv, level="level").dropna()
            mean, lo, hi = boot(diff)
            paired.append(dict(method=a, minus=b, level=lv, diff_db=mean, lo=lo, hi=hi,
                               events=len(diff), wins=int((diff > 0).sum())))
    syn, paired = pd.DataFrame(rows), pd.DataFrame(paired)
    syn.to_csv(OUT / "final_synthetic.csv", index=False)
    paired.to_csv(OUT / "final_paired.csv", index=False)

    auc = pd.read_csv(OUT / "variants_auc.csv").set_index("model")
    det = []
    for name, cols in FAMILIES.items():
        cols = DETECTION.get(name, cols)
        have = [c for c in cols if c in auc.index]
        if have:
            v = auc.loc[have, "auc"]
            det.append(dict(method=name, auc_mean=v.mean(), auc_min=v.min(), auc_max=v.max(), runs=len(v)))
    det = pd.DataFrame(det)
    det.to_csv(OUT / "final_detection.csv", index=False)

    print("Test days. SI-SDR gain (dB) per input SNR [95% interval]; detection ROC AUC (range over seeds).\n")
    print(f"{'Method':32}" + "".join(f"{str(lv) + ' dB':>19}" for lv in LEVELS) + f"{'AUC':>16}")
    for name in fam:
        s = syn[syn.method == name]
        cells = [f"{r.si_gain_db:+.2f} [{r.lo:+.1f},{r.hi:+.1f}]" for r in s.itertuples()]
        d = det[det.method == name]
        a = ""
        if len(d):
            r = d.iloc[0]
            a = f"{r.auc_mean:.3f}" + (f" ({r.auc_min:.3f}-{r.auc_max:.3f})" if r.runs > 1 else "")
        print(f"{name:32}" + "".join(f"{c:>19}" for c in cells) + f"{a:>16}")
    print("\nPaired, per test event (dB, 'yes' = 95% interval excludes zero; wins = events of 19 where it is higher):")
    for b in paired.minus.unique():
        p = paired[paired.minus == b]
        cells = [f"{r.diff_db:+.2f} {'yes' if r.lo > 0 or r.hi < 0 else 'no '} {r.wins:2d}/19" for r in p.itertuples()]
        print(f"  U-Net v1 - {b:30}" + "".join(f"{c:>19}" for c in cells))
    print("\nBaselines: synthetic test bandpass 0.5-45 Hz, f-k 250 m/s; detection bandpass 1-10 Hz, f-k 1000 m/s.")
    print("Files: results/final_synthetic.csv, final_paired.csv, final_detection.csv")


if __name__ == "__main__":
    main()
