"""Frozen-checkpoint sensitivities on the original validation earthquakes."""

from functools import partial
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.signal import resample_poly
from ..data import ROOT, LEVELS, SEED, load, mix
from ..metrics import fk
from .. import neural
from .metrics import all_scores
from .stats import OUT, aggregate
from .baseline import MEASURES
from .noise_data import DEST


def pad_input(x, mode, seed):
    """Return the model input and inverse mapping to the measured clip."""
    x = x - x.mean(1, keepdims=True)
    n = x.shape[1]
    size = n * 4 if mode == "stretch4" else n
    start = (neural.N - size) // 2
    pads = ((0, 0), (start, neural.N - size - start))
    if mode in ("zero", "stretch4"):
        center = resample_poly(x, 4, 1, axis=1) if mode == "stretch4" else x
        full = np.pad(center, pads)
    elif mode == "reflect":
        full = np.pad(x, pads, mode="reflect")
    elif mode == "tile":
        full = x[:, (np.arange(neural.N) - start) % n]
    elif mode == "noise":
        rng = np.random.default_rng(seed)
        full = rng.normal(size=(len(x), neural.N)) * x.std(1, keepdims=True)
        full[:, start : start + n] = x
    else:
        raise ValueError(mode)

    def crop(out):
        y = out[:, start : start + size]
        return resample_poly(y, 1, 4, axis=-1) if mode == "stretch4" else y

    return full, crop


class Models:
    def __init__(self):
        import tf_keras as k

        net = neural.model()
        self.ex = k.Model(net.input, net.layers[-2].input)
        self.weights = {
            "Zero-shot": net.layers[-2].get_weights()[0][0, 0],
            "Fine-tuned": np.load(ROOT / "models/das_head.npz")["kernel"][0, 0],
        }

    def both(self, x, mode="zero", seed=SEED):
        # Preserve the released zero-padding numerics exactly. Other paddings use
        # the same model normalization on the full artificially extended input.
        full, crop = pad_input(x, mode, seed) if mode != "zero" else (x, lambda y: y)
        ys = {name: [] for name in self.weights}
        for j in range(0, len(x), 8):
            feats, z, start = neural.encode(full[j : j + 8])
            h = self.ex(feats, training=False).numpy()
            for name, w in self.weights.items():
                mask = expit(h @ (w[:, 0] - w[:, 1]))
                ys[name].append(crop(neural.decode(z, mask, start, full.shape[1])))
        return {key: np.concatenate(v) for key, v in ys.items()}


def other_donors(events):
    mapping = {}
    for _, g in events.groupby("array"):
        ids = g.sort_values("event_id").index.to_numpy()
        if len(ids) < 2:
            raise ValueError("Need a distinct donor on the same array")
        mapping.update(zip(ids, np.roll(ids, -1)))
    return mapping


def run():
    ev, p = load(ROOT / "data/prepared")
    val = ev[ev.split == "validation"]
    donors = other_donors(val)
    models = Models()
    alt = dict(np.load(DEST / "other_blocks.npz"))
    cont = (
        np.load(DEST / "continuous.npz")["background"]
        if (DEST / "continuous.npz").exists()
        else None
    )
    cases = []
    rows = []
    cont_index = 0
    for i, e in val.iterrows():
        s, n, pick = (p[k][i] for k in ("reference", "noise", "nominal_p"))
        bg = {"other_event": p["noise"][donors[i]], "other_block": alt[str(e.event_id)]}
        if cont is not None and e["array"] == "ridgecrest":
            start = 500 + 800 * cont_index
            cont_index += 1
            v = cont[:, start : start + 70].astype(float)
            bg["continuous"] = v - v.mean(1, keepdims=True)
        else:
            start = None
        cases.append(
            dict(
                event_id=e.event_id,
                array=e["array"],
                noise_event=ev.at[donors[i], "event_id"],
                continuous_start_index=start,
            )
        )
        for level in LEVELS:
            x = mix(s, n, level)
            seed = SEED + int(e.event_id) + int(level)

            def add(study, setting, method, y, ref=s, mixture=x, arrivals=pick):
                rows.append(
                    dict(
                        event_id=e.event_id,
                        array=e["array"],
                        level=level,
                        study=study,
                        setting=str(setting),
                        method=method,
                        **all_scores(ref, mixture, y, arrivals)
                    )
                )

            base = models.both(x)

            def standard(ref, noise, arrivals, study, setting, prior=None):
                xx = mix(ref, noise, level)
                predictions = {
                    "Noisy": xx,
                    "f-k": fk(xx),
                    **(prior if prior is not None else models.both(xx)),
                }
                for method, y in predictions.items():
                    add(study, setting, method, y, ref, xx, arrivals)

            for mode in ["zero", "reflect", "noise", "tile", "stretch4"]:
                pred = base if mode == "zero" else models.both(x, mode, seed)
                for method, y in pred.items():
                    add("padding", mode, method, y)
            for kind, v in dict(original=n, **bg).items():
                standard(
                    s, v, pick, "noise", kind, base if kind == "original" else None
                )
            for width in [8, 16, 24]:
                a = (24 - width) // 2
                sl = slice(a, a + width)
                standard(
                    s[sl],
                    n[sl],
                    pick[sl],
                    "aperture",
                    width,
                    base if width == 24 else None,
                )
            spatial = fk(x)
            for key, y in {
                "f-k": spatial,
                "Fine-tuned": base["Fine-tuned"],
                "f-k then head": models.both(spatial)["Fine-tuned"],
                "head then f-k": fk(base["Fine-tuned"]),
            }.items():
                add("hybrid", "original", key, y)
        print("finished sensitivity event", e.event_id, flush=True)
    d = pd.DataFrame(rows)
    pd.DataFrame(cases).to_csv(OUT / "noise_donors.csv", index=False)
    d.to_csv(OUT / "sensitivity_events.csv", index=False)
    aggregate(d, ["study", "setting", "method", "level"], MEASURES).to_csv(
        OUT / "sensitivity.csv", index=False
    )
    # Matched three-event comparison with the external continuous segment.
    rc = d[(d.study == "noise") & (d.array == "ridgecrest")]
    aggregate(rc, ["setting", "method", "level"], MEASURES).to_csv(
        OUT / "noise_ridgecrest.csv", index=False
    )


if __name__ == "__main__":
    run()
