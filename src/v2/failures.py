"""Event rankings fixed at 0 dB input, with all ties flagged."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..data import ROOT
from .stats import OUT, FIG, aggregate
from .baseline import MEASURES


def run():
    d = pd.concat(
        [pd.read_csv(OUT / f"{name}_events.csv") for name in ("baseline", "decoder")]
    )
    meta = pd.read_csv(ROOT / "data/prepared/events.csv")
    aggregate(d, ["array", "level", "method"], MEASURES).to_csv(
        OUT / "site_summary.csv", index=False
    )
    dc = d[d.method == "Decoder"].merge(
        meta[["event_id", "magnitude", "median_source_snr"]],
        on="event_id",
        validate="many_to_one",
    )
    strata = []
    for metric, edges, labels in [
        (
            "magnitude",
            [-np.inf, 1.1, 1.8, np.inf],
            ["M < 1.1", "1.1 ≤ M < 1.8", "M ≥ 1.8"],
        ),
        (
            "median_source_snr",
            [-np.inf, 22, 30, np.inf],
            ["<22 dB", "22–<30 dB", "≥30 dB"],
        ),
    ]:
        dc["bin"] = pd.cut(dc[metric], edges, labels=labels, right=False)
        strata.append(
            aggregate(
                dc, ["bin", "level", "method"], ["gain_db", "cc", "pick_recovery"]
            ).assign(stratifier=metric)
        )
    pd.concat(strata).to_csv(OUT / "decoder_strata.csv", index=False)
    d = d[d.level == 0].merge(
        meta[["event_id", "magnitude", "median_source_snr", "first_channel"]],
        on="event_id",
        validate="many_to_one",
    )
    chosen = []
    for method, g in d.groupby("method"):
        g = g.assign(tie_at_display_precision=g.gain_db.round(8).duplicated(keep=False))
        for side, asc in [("worst", True), ("best", False)]:
            block = (
                g.assign(rank_gain=g.gain_db.round(8))
                .sort_values(["rank_gain", "event_id"], ascending=[asc, True])
                .head(10)
                .copy()
            )
            block["side"] = side
            block["rank"] = np.arange(1, len(block) + 1)
            chosen.append(block)
    ranks = pd.concat(chosen)
    cols = [
        "method",
        "side",
        "rank",
        "event_id",
        "array",
        "magnitude",
        "median_source_snr",
        "first_channel",
        "gain_db",
        "si_gain_db",
        "cc",
        "tie_at_display_precision",
    ]
    ranks[cols].to_csv(OUT / "extremes.csv", index=False)
    summary = ranks.groupby(["method", "side"]).agg(
        events=("event_id", "size"),
        ridgecrest=("array", lambda x: sum(x == "ridgecrest")),
        median_magnitude=("magnitude", "median"),
        median_source_snr=("median_source_snr", "median"),
        median_first_channel=("first_channel", "median"),
        mean_gain=("gain_db", "mean"),
    )
    summary.to_csv(OUT / "extreme_profiles.csv")
    fig, axs = plt.subplots(2, 4, figsize=(10, 5.4), layout="constrained")
    for ax, (method, g) in zip(axs.flat, ranks.groupby("method")):
        for side, c in [("best", "#117f79"), ("worst", "#bb5c34")]:
            s = g[g.side == side]
            ax.scatter(s["rank"], s.gain_db, s=14, c=c, label=side)
        ax.set_title(method, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].legend(fontsize=7)
    fig.supxlabel("Within-tail rank at 0 dB input")
    fig.supylabel("Event SNR gain (dB)")
    fig.savefig(FIG / "extremes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
