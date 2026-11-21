"""Select a time stretch on development out-of-fold predictions, then score the fixed test."""
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from scipy.special import expit

from .. import neural
from ..data import LEVELS, ROOT, load, mix
from ..v2.baseline import MEASURES
from ..v2.metrics import all_scores
from ..v2.stats import aggregate
from ..v2.training import CACHE, cache, folds, head_fit
from .experiments import CFG, OUT


def predictions(extractor, x, factor, weights):
    expanded = x if factor == 1 else resample_poly(x, factor, 1, axis=-1)
    recovered = {name: [] for name in weights}
    for start in range(0, len(x), 8):
        features, spec, offset = neural.encode(expanded[start:start + 8])
        h = extractor(features, training=False).numpy()
        for name, w in weights.items():
            mask = expit(h @ (w[:, 0] - w[:, 1]))
            out = neural.decode(spec, mask, offset, expanded.shape[1])
            if factor > 1:
                out = resample_poly(out, 1, factor, axis=-1)
            recovered[name].append(out[:, :x.shape[1]])
    return {name: np.concatenate(values) for name, values in recovered.items()}


def select(grid):
    return grid.loc[grid.groupby("method").si_gain_db.idxmax()].reset_index(drop=True)


def run():
    import tf_keras as keras

    events, pairs = load(ROOT / "data/prepared")
    dev = events[events.split.isin(["train", "validation"])]
    net = cache(dev, pairs)
    extractor = keras.Model(net.input, net.layers[-2].input)
    original = net.layers[-2].get_weights()[0][0, 0]
    bank = {}
    for i, e in dev.iterrows():
        with np.load(CACHE / f"{e.event_id}.npz") as data:
            bank[i] = dict(data)

    def evaluate(ids, fold, settings):
        rows = []
        for i in ids:
            s, n, pick = (pairs[key][i] for key in ["reference", "noise", "nominal_p"])
            for level in LEVELS:
                x = mix(s, n, level)
                for factor, weights in settings.items():
                    for name, y in predictions(extractor, x, factor, weights).items():
                        rows.append(dict(event_id=events.at[i, "event_id"], array=events.at[i, "array"],
                                         fold=fold, level=level, factor=factor, method=name,
                                         **all_scores(s, x, y, pick)))
        return rows

    rows = []
    for fold, (fit, stop, score) in enumerate(folds(dev), 1):
        w, _, _ = head_fit(net, bank, fit, stop)
        settings = {f: {"Zero-shot": original, "Fine-tuned": w} for f in CFG["stretch"]["factors"]}
        rows.extend(evaluate(score, fold, settings))
    cv = pd.DataFrame(rows)
    cv.to_csv(OUT / "stretch_cv_events.csv", index=False)
    grid = cv.groupby(["method", "factor"], as_index=False)[["gain_db", "si_gain_db"]].mean()
    choices = select(grid)
    grid.to_csv(OUT / "stretch_grid.csv", index=False)
    choices.to_csv(OUT / "stretch_selection.csv", index=False)
    w = np.load(ROOT / "models/das_head.npz")["kernel"][0, 0]
    settings = {}
    for row in choices.itertuples():
        settings.setdefault(row.factor, {})[row.method] = w if row.method == "Fine-tuned" else original
    fixed = pd.DataFrame(evaluate(events.index[events.split == "test"], 0, settings))
    fixed.to_csv(OUT / "stretch_events.csv", index=False)
    aggregate(fixed, ["method", "factor", "level"], MEASURES).to_csv(OUT / "stretch_summary.csv", index=False)
    fixed = fixed.merge(events[["event_id","magnitude"]], on="event_id", validate="many_to_one")
    fixed["bin"] = pd.cut(fixed.magnitude, [-np.inf, 1.1, 1.8, np.inf], right=False, labels=["low","middle","high"])
    aggregate(fixed, ["bin","method","level"], ["gain_db","si_gain_db"]).to_csv(OUT / "stretch_strata.csv",index=False)


if __name__ == "__main__":
    run()
