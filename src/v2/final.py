"""One held-out pass using the frozen decoder selected on the original validation split."""

import numpy as np
import pandas as pd
from ..data import ROOT, LEVELS, load, mix
from ..neural import model, denoise
from .training import MODEL, LAYER_NAMES
from .baseline import MEASURES
from .stats import OUT, aggregate
from .metrics import all_scores


def decoder_model():
    net = model()
    with np.load(MODEL) as z:
        for name in LAYER_NAMES:
            layer = net.get_layer(name)
            layer.set_weights(
                [z[f"{name}_{j}"] for j in range(len(layer.get_weights()))]
            )
    return net


def run():
    events, p = load(ROOT / "data/prepared")
    net = decoder_model()
    rows = []
    for i, e in events[events.split == "test"].iterrows():
        ref, n, picks = (p[k][i] for k in ("reference", "noise", "nominal_p"))
        for level in LEVELS:
            x = mix(ref, n, level)
            y = denoise(net, x)
            rows.append(
                dict(
                    event_id=e.event_id,
                    array=e["array"],
                    level=level,
                    method="Decoder",
                    **all_scores(ref, x, y, picks),
                )
            )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "decoder_events.csv", index=False)
    aggregate(d, ["method", "level"], MEASURES).to_csv(
        OUT / "decoder_summary.csv", index=False
    )
    legacy = pd.read_csv(OUT / "baseline_events.csv")
    chunks = []
    for method in ["Fine-tuned", "Zero-shot"]:
        j = d.merge(
            legacy[legacy.method == method],
            on=["level", "array", "event_id"],
            validate="one_to_one",
        )
        for metric in ["gain_db", "si_gain_db", "cc"]:
            j[metric] = j[metric + "_x"] - j[metric + "_y"]
        chunks.append(
            aggregate(j, ["level"], ["gain_db", "si_gain_db", "cc"]).assign(
                comparator=method
            )
        )
    pd.concat(chunks).to_csv(OUT / "decoder_paired.csv", index=False)


if __name__ == "__main__":
    run()
