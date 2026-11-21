"""Test surrogate targets, source SNR and alternative early windows on frozen weights."""
import numpy as np
import pandas as pd

from ..data import ROOT, LEVELS, load, mix
from ..v2.baseline import MEASURES
from ..v2.metrics import all_scores
from ..v2.sensitivity import Models
from ..v2.stats import aggregate
from .experiments import CFG, OUT


def run():
    events, pairs = load(ROOT / "data/prepared")
    model = Models()
    rows = []
    for i, e in events[events.split == "test"].iterrows():
        s, n, pick = (pairs[k][i] for k in ["reference", "noise", "nominal_p"])
        early = pairs["raw"][i, :, :70]
        cases = [("alternate early", s, early - early.mean(1, keepdims=True)),
                 ("stack target", np.broadcast_to(np.median(s, axis=0), s.shape), n)]
        if e.median_source_snr >= CFG["sensitivity"]["source_snr_min"]:
            cases.append(("source at least 35 dB", s, n))
        for condition, target, background in cases:
            # The broadcast median target has one common nominal onset across the block.
            arrivals = np.full_like(pick, 14) if condition == "stack target" else pick
            for level in LEVELS:
                x = mix(target, background, level)
                for method, y in model.both(x).items():
                    rows.append(dict(event_id=e.event_id, array=e["array"], level=level, condition=condition,
                                     method=method, **all_scores(target, x, y, arrivals)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "sensitivity_events.csv", index=False)
    for name, keys in [("summary", ["condition","method","level"]), ("site", ["array","condition","method","level"])]:
        aggregate(d, keys, MEASURES).to_csv(OUT / f"sensitivity_{name}.csv", index=False)


if __name__ == "__main__":
    run()
