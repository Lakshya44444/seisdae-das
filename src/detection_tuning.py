"""Re-tune the baselines for detection on validation days, then report every setting on test days.

  python -m src.detection_tuning

The frozen settings in results/run.json were chosen with the synthetic SI-SDR
criterion. Here every bandpass band, f-k velocity and cascade velocity of the
grid is scored by detection AUC on validation days only (validation earthquake
windows without an overlapping event, and validation noise windows). The best
setting of each family is fixed, and then all settings are scored on the test
days (a sensitivity table) together with paired bootstrap comparisons.

Writes results/detection_tuning_validation.csv,
results/detection_sensitivity_test.csv, results/detection_retuned_paired.csv,
results/detection_tuning_rows.csv (every test score) and
results/detection_retuned_by_magnitude.csv (exploratory).
"""

import numpy as np
import pandas as pd

from .build import load_gather, manifest
from .das import CFG, geometry
from .detection_stats import auc, bootstrap
from .evaluate import OUT, Runner, detection_stat
from .methods import ROOT, bandpass, fk

E = CFG["evaluation"]


def candidates(runner, x):
    """Every method and every grid setting for one gather."""
    das = runner.das(x)
    out = {"Noisy": x, "seisDAE (released)": runner.gr(x), "seisDAE (DAS-trained)": das}
    for band in E["bandpass_hz"]:
        out[f"Bandpass {band[0]}-{band[1]} Hz"] = bandpass(x, band)
    for v in E["fk_velocity_m_s"]:
        out[f"f-k {v} m/s"] = fk(x, runner.dx, v)
        out[f"seisDAE (DAS) + f-k {v} m/s"] = fk(das, runner.dx, v)
    return out


def score(runner, m, split):
    overlap = m["overlap"].fillna(0) if "overlap" in m else pd.Series(0, index=m.index)
    groups = (("earthquake", m[(m.kind != "noise") & (m.split == split) & (overlap == 0)]),
              ("noise", m[(m.kind == "noise") & (m.split == split)]))
    rows = []
    for label, group in groups:
        for k, r in enumerate(group.itertuples()):
            for method, y in candidates(runner, load_gather(r.id)).items():
                rows.append(dict(window=r.id, label=label, method=method, stat_db=detection_stat(y),
                                 magnitude=getattr(r, "magnitude", np.nan)))
            print(f"{split} {label} {k + 1}/{len(group)}", end="\r")
    print()
    d = pd.DataFrame(rows)
    eq = d[d.label == "earthquake"].pivot(index="window", columns="method", values="stat_db")
    nz = d[d.label == "noise"].pivot(index="window", columns="method", values="stat_db")
    return eq, nz, d


def family(method):
    for name in ("seisDAE (DAS) + f-k", "Bandpass", "f-k"):
        if method.startswith(name):
            return name
    return method


def main():
    m = manifest()
    runner = Runner(ROOT / "models/das_seisdae.h5", geometry()[2])

    veq, vnz, _ = score(runner, m, "validation")
    val = pd.DataFrame([dict(method=c, family=family(c), validation_auc=auc(veq[c], vnz[c])) for c in veq.columns])
    best = val.loc[val.groupby("family").validation_auc.idxmax()].set_index("family").method.to_dict()
    val["chosen"] = val.method.isin(best.values())
    val.to_csv(OUT / "detection_tuning_validation.csv", index=False)
    print(f"validation: {len(veq)} earthquake windows, {len(vnz)} noise windows")
    print(val.sort_values(["family", "method"]).round(3).to_string(index=False))

    teq, tnz, rows = score(runner, m, "test")
    rows.to_csv(OUT / "detection_tuning_rows.csv", index=False)
    methods = list(teq.columns)
    boot = bootstrap(teq, tnz, methods, CFG["seed"])["auc"]
    sens = []
    for c in methods:
        lo, hi = np.quantile(boot[c], [0.025, 0.975])
        sens.append(dict(method=c, family=family(c), chosen_on_validation=c in best.values(),
                         test_auc=auc(teq[c], tnz[c]), lo=lo, hi=hi))
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "detection_sensitivity_test.csv", index=False)
    print(f"\ntest: {len(teq)} earthquake windows, {len(tnz)} noise windows")
    print(sens.sort_values(["family", "method"]).round(3).to_string(index=False))

    fk_best, cascade_best, bp_best = best["f-k"], best["seisDAE (DAS) + f-k"], best["Bandpass"]
    pairs = [("seisDAE (DAS-trained)", "Noisy"), ("seisDAE (DAS-trained)", fk_best), (cascade_best, "Noisy"),
             (cascade_best, fk_best), (fk_best, "Noisy"), (bp_best, "Noisy"),
             ("seisDAE (DAS-trained)", bp_best), (cascade_best, bp_best)]
    paired = []
    for a, b in pairs:
        d = boot[a] - boot[b]
        lo, hi = np.quantile(d, [0.025, 0.975])
        paired.append(dict(comparison=f"{a} minus {b}", difference=auc(teq[a], tnz[a]) - auc(teq[b], tnz[b]),
                           lo=lo, hi=hi, share_draws_above_zero=float(np.mean(d > 0))))
    paired = pd.DataFrame(paired)
    paired.to_csv(OUT / "detection_retuned_paired.csv", index=False)
    print()
    print(paired.round(3).to_string(index=False))

    mag = rows[rows.label == "earthquake"].drop_duplicates("window").set_index("window").magnitude.reindex(teq.index)
    keep = ["Noisy", bp_best, fk_best, "seisDAE (released)", "seisDAE (DAS-trained)", cascade_best]
    bymag = []
    for lo_m, hi_m, name in ((0.0, 1.5, "M < 1.5"), (1.5, 10.0, "M >= 1.5")):
        sel = ((mag >= lo_m) & (mag < hi_m)).to_numpy()
        b = bootstrap(teq[sel][keep], tnz[keep], keep, CFG["seed"] + 3)["auc"]
        for c in keep:
            lo, hi = np.quantile(b[c], [0.025, 0.975])
            bymag.append(dict(bin=name, earthquakes=int(sel.sum()), item=c, value=auc(teq[c][sel], tnz[c]), lo=lo, hi=hi))
        for a, c in (("seisDAE (DAS-trained)", "Noisy"), ("seisDAE (DAS-trained)", bp_best),
                     (cascade_best, bp_best), (bp_best, "Noisy")):
            dd = b[a] - b[c]
            lo, hi = np.quantile(dd, [0.025, 0.975])
            bymag.append(dict(bin=name, earthquakes=int(sel.sum()), item=f"{a} minus {c}",
                              value=auc(teq[a][sel], tnz[a]) - auc(teq[c][sel], tnz[c]), lo=lo, hi=hi))
    bymag = pd.DataFrame(bymag)
    bymag.to_csv(OUT / "detection_retuned_by_magnitude.csv", index=False)
    print("\nExploratory: AUC by catalogue magnitude, re-tuned settings")
    print(bymag.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
