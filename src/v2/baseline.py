from pathlib import Path
import numpy as np
import pandas as pd
from ..data import ROOT, LEVELS, load, mix
from .metrics import all_scores
from .stats import OUT, aggregate

MEASURES = [
    "gain_db",
    "si_gain_db",
    "cc",
    "peak_ratio",
    "polarity_agreement",
    "pick_recovery",
    "pick_mae_ms",
]


def run():
    events, pairs = load(ROOT / "data/prepared")
    rows = []
    for i, e in events[events.split == "test"].iterrows():
        s, n, p = (pairs[k][i] for k in ("reference", "noise", "nominal_p"))
        for level in LEVELS:
            x = mix(s, n, level)
            with np.load(
                ROOT / "runs/notebook_test/predictions" / f"{e.event_id}_{level}.npz"
            ) as z:
                for method in z.files:
                    rows.append(
                        dict(
                            event_id=e.event_id,
                            array=e["array"],
                            level=level,
                            method=method,
                            **all_scores(s, x, z[method], p),
                        )
                    )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "baseline_events.csv", index=False)
    aggregate(d, ["level", "method"], MEASURES).to_csv(
        OUT / "baseline_summary.csv", index=False
    )


if __name__ == "__main__":
    run()
