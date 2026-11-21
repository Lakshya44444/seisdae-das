"""Aggregate metrics by earthquake and bootstrap paired differences within array."""

import numpy as np
import pandas as pd

from .data import SEED


def summarise(df, keys):
    g = df.groupby(keys, sort=True)
    table = g[["gain_db", "cc", "rmse", "nrmse", "ms_per_trace"]].mean()
    counts = g[["reference_picks", "matched_picks", "pick_abs_ms_sum"]].sum()
    table["events"] = g.event_id.nunique()
    table["onset_mae_ms"] = counts.pick_abs_ms_sum / counts.matched_picks.replace(
        0, np.nan
    )
    table["pick_recovery"] = counts.matched_picks / counts.reference_picks.replace(
        0, np.nan
    )
    return table.join(counts[["reference_picks", "matched_picks"]])


def paired_intervals(df, draws=5000):
    """Keep all channels of each earthquake together; preserve array counts."""
    rng = np.random.default_rng(SEED)
    rows = []
    for level, group in df.groupby("level"):
        scores = group.pivot(
            index=["array", "event_id"], columns="method", values="gain_db"
        )
        for baseline in ("Zero-shot", "f-k", "Scalar"):
            if baseline not in scores or "Fine-tuned" not in scores:
                continue
            delta = scores["Fine-tuned"] - scores[baseline]
            bootstrap = np.zeros(draws)
            for _, values in delta.groupby(level="array"):
                x = values.to_numpy()
                bootstrap += rng.choice(x, (draws, len(x)), replace=True).sum(axis=1)
            low, high = np.quantile(bootstrap / len(delta), [0.025, 0.975])
            rows.append(
                (
                    level,
                    baseline,
                    len(delta),
                    delta.mean(),
                    low,
                    high,
                    (delta > 0).sum(),
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "level",
            "baseline",
            "events",
            "tuned_minus_baseline_db",
            "ci95_low_db",
            "ci95_high_db",
            "events_tuned_wins",
        ],
    )
