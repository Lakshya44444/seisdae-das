"""Event-level bootstrap. All ratios of counts are recomputed per resample."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import ROOT, SEED

OUT = Path(os.environ.get("SEISDAE_V2_OUT", ROOT / "results_v2")).resolve()
FIG = ROOT / "figures_v2"
DRAWS = 5000
RATIOS = {
    "pick_recovery": ("matched_picks", "reference_picks"),
    "pick_mae_ms": ("pick_abs_ms_sum", "matched_picks"),
    "polarity_agreement": ("polarity_hits", "polarity_total"),
}


def aggregate(d, keys, measures):
    rows = []
    for labels, g in d.groupby(keys, observed=True, sort=True):
        labels = labels if isinstance(labels, tuple) else (labels,)
        g = g.sort_values(["array", "event_id"]).reset_index(drop=True)
        if g.event_id.duplicated().any():
            raise ValueError("Group must contain one row per earthquake")
        rng = np.random.default_rng(SEED)
        ids = np.concatenate(
            [
                rng.choice(block.index, (DRAWS, len(block)), replace=True)
                for _, block in g.groupby("array")
            ],
            axis=1,
        )
        row = dict(zip(keys, labels))
        row.update(
            events=len(g),
            longvalley=int((g.array == "longvalley").sum()),
            ridgecrest=int((g.array == "ridgecrest").sum()),
        )
        for m in measures:
            if m in RATIOS:
                a, b = (g[q].to_numpy(dtype=float) for q in RATIOS[m])
                num, den = a[ids].sum(1), b[ids].sum(1)
                boot = np.divide(num, den, out=np.full(DRAWS, np.nan), where=den > 0)
                point = a.sum() / b.sum() if b.sum() else np.nan
            else:
                a = g[m].to_numpy(dtype=float)
                boot = np.nanmean(a[ids], axis=1)
                point = np.nanmean(a)
            row[m] = point
            row[m + "_lo"], row[m + "_hi"] = np.nanquantile(boot, [0.025, 0.975])
        rows.append(row)
    return pd.DataFrame(rows)


def strata():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    e = pd.read_csv(ROOT / "data/prepared/events.csv")
    d = pd.read_csv(ROOT / "results/per_event.csv").merge(
        e[["event_id", "magnitude", "median_source_snr"]],
        on="event_id",
        validate="many_to_one",
    )
    groupings = {
        "magnitude": (
            [-np.inf, 1.1, 1.8, np.inf],
            ["M < 1.1", "1.1 ≤ M < 1.8", "M ≥ 1.8"],
        ),
        "median_source_snr": (
            [-np.inf, 22, 30, np.inf],
            ["<22 dB", "22–<30 dB", "≥30 dB"],
        ),
    }
    summaries = []
    effects = []
    for key, (edges, names) in groupings.items():
        d["bin"] = pd.cut(d[key], edges, labels=names, right=False)
        summary = aggregate(
            d, ["bin", "level", "method"], ["gain_db", "cc", "pick_recovery"]
        )
        summaries.append(summary.assign(stratifier=key))
        keys = ["event_id", "array", "level", "bin"]
        w = (
            d.pivot(index=keys, columns="method", values="gain_db")
            .dropna()
            .reset_index()
        )
        w["delta_db"] = w["Fine-tuned"] - w["Zero-shot"]
        effects.append(
            aggregate(w, ["bin", "level"], ["delta_db"]).assign(stratifier=key)
        )
    pd.concat(summaries).to_csv(OUT / "strata.csv", index=False)
    pd.concat(effects).to_csv(OUT / "strata_paired.csv", index=False)
    # Direct small-minus-large comparison, with independent event resampling by array within bin.
    w = (
        d[(d.level == 0)]
        .pivot(
            index=["event_id", "array", "magnitude"], columns="method", values="gain_db"
        )
        .reset_index()
    )
    w["delta"] = w["Fine-tuned"] - w["Zero-shot"]
    rng = np.random.default_rng(SEED)
    values = []
    for group in (w[w.magnitude < 1.1], w[w.magnitude >= 1.8]):
        samples = sum(
            rng.choice(site.delta, (DRAWS, len(site))).sum(1)
            for _, site in group.groupby("array")
        ) / len(group)
        values.append((group.delta.mean(), samples))
    effect = values[0][1] - values[1][1]
    pd.DataFrame(
        [
            dict(
                input_db=0,
                small_minus_large_db=values[0][0] - values[1][0],
                lo=np.quantile(effect, 0.025),
                hi=np.quantile(effect, 0.975),
            )
        ]
    ).to_csv(OUT / "magnitude_contrast.csv", index=False)


if __name__ == "__main__":
    strata()
