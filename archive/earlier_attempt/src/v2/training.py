"""Cache the frozen decoder input and fit on development earthquakes only."""

import os

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from ..data import ROOT, LEVELS, SEED, load, mix
from .. import neural
from ..train import fit_head
from .metrics import all_scores
from .baseline import MEASURES
from .stats import OUT, aggregate

CACHE = ROOT / "runs/v2_features"
LAYER_NAMES = ("conv2d_16", "batch_normalization_21", "conv2d_17")
MODEL = ROOT / "models/decoder_v2.npz"


def folds(events):
    rng = np.random.default_rng(SEED)
    blocks = [
        np.array_split(rng.permutation(g.index), 3) for _, g in events.groupby("array")
    ]
    for f in range(3):
        score = np.concatenate([b[f] for b in blocks])
        pool = [
            rng.permutation(np.concatenate([v for j, v in enumerate(b) if j != f]))
            for b in blocks
        ]
        yield sorted(np.concatenate([b[2:] for b in pool])), sorted(
            np.concatenate([b[:2] for b in pool])
        ), sorted(score)


def cache(events, pairs):
    import tf_keras as k

    CACHE.mkdir(parents=True, exist_ok=True)
    net = neural.model()
    ex = k.Model(
        net.input, [net.get_layer("add_4").output, net.layers[-2].input, net.output]
    )
    for i, e in events.iterrows():
        path = CACHE / f"{e.event_id}.npz"
        if path.exists():
            continue
        s, n = pairs["reference"][i], pairs["noise"][i]
        arrays = {j: [] for j in ("decoder", "head", "target", "z")}
        for level in LEVELS:
            x = mix(s, n, level)
            inp, z, _ = neural.encode(x)
            _, zs, _ = neural.encode(s)
            _, zn, _ = neural.encode(x - s)
            active = np.flatnonzero((abs(zs) + abs(zn)).max((0, 1)) > 0)
            if not np.array_equal(active, [59, 60, 61]):
                raise ValueError("Unexpected STFT support")
            full = [ex(inp[j : j + 8], training=False) for j in range(0, len(x), 8)]
            feat, h, original = [
                np.concatenate([np.asarray(t[j]) for t in full]) for j in range(3)
            ]
            arrays["decoder"].append(feat[:, :, 58:63])
            arrays["head"].append(h[:, :, 59:62])
            mag = np.stack([abs(zs[:, :, 59:62]), abs(zn[:, :, 59:62])], axis=-1)
            arrays["target"].append(
                np.divide(
                    mag,
                    mag.sum(-1, keepdims=True),
                    out=np.zeros_like(mag),
                    where=mag.sum(-1, keepdims=True) > 0,
                )
            )
            arrays["z"].append(z[:, :, 59:62])
            if i == events.index[0]:
                fresh = tail(net)
                np.testing.assert_allclose(
                    fresh(feat[:, :, 58:63]).numpy(), original[:, :, 59:62], atol=3e-6
                )
        np.savez(path, **arrays)
    return net


def tail(net):
    import tf_keras as k

    inp = k.Input((100, 5, 8))
    x = inp
    for name in (
        "conv2d_16",
        "batch_normalization_21",
        "leaky_re_lu_21",
        "conv2d_17",
        "softmax",
    ):
        old = net.get_layer(name)
        config = old.get_config()
        if "kernel_initializer" in config:
            config["kernel_initializer"] = "zeros"
        layer = type(old).from_config(config)
        x = layer(x, training=False) if name.startswith("batch") else layer(x)
        layer.set_weights(old.get_weights())
    return k.Model(inp, x[:, :, 1:-1])


def arrays(bank, which, key):
    return np.concatenate(
        [bank[i][key][:, ::4].reshape(-1, *bank[i][key].shape[2:]) for i in which]
    )


def head_fit(net, bank, tr, va, seed=SEED):
    kernel = net.layers[-2].get_weights()[0]

    def unpack(idx):
        x = arrays(bank, idx, "head").reshape(-1, 8)
        y = arrays(bank, idx, "target").reshape(-1, 2)
        good = y.sum(1) > 0
        return x[good], y[good]

    w, _, epoch, loss = fit_head(*unpack(tr), *unpack(va), kernel, seed=seed)
    return w, epoch, loss


def decoder_fit(net, bank, tr, va):
    import tensorflow as tf
    import tf_keras as k

    k.utils.set_random_seed(SEED)
    model = tail(net)
    opt = k.optimizers.Adam(3e-5)
    xt, yt = arrays(bank, tr, "decoder"), arrays(bank, tr, "target")
    xv, yv = arrays(bank, va, "decoder"), arrays(bank, va, "target")

    def loss(x, y):
        pred = tf.clip_by_value(model(x, training=False), 1e-8, 1.0)
        valid = tf.reduce_sum(y, axis=-1) > 0
        return -tf.reduce_mean(
            tf.boolean_mask(tf.reduce_sum(y * tf.math.log(pred), axis=-1), valid)
        )

    @tf.function(reduce_retracing=True)
    def step(x, y):
        with tf.GradientTape() as tape:
            value = loss(x, y)
        opt.apply_gradients(
            zip(tape.gradient(value, model.trainable_weights), model.trainable_weights)
        )

    rng = np.random.default_rng(SEED)
    best = model.get_weights()
    minimum = float(loss(xv, yv))
    epoch_best = 0
    for epoch in range(1, 17):
        order = rng.permutation(len(xt))
        for j in range(0, len(order), 32):
            ids = order[j : j + 32]
            step(xt[ids], yt[ids])
        current = float(loss(xv, yv))
        if current < minimum:
            minimum = current
            epoch_best = epoch
            best = model.get_weights()
        if epoch - epoch_best >= 4:
            break
    model.set_weights(best)
    return model, epoch_best, minimum


def predict(entry, w, decoder=None):
    if decoder is None:
        logit = entry["head"] @ (w[:, 0] - w[:, 1])
        mask = expit(logit)
    else:
        mask = np.array(
            [decoder(v, training=False).numpy()[..., 0] for v in entry["decoder"]]
        )
    spec = np.zeros((*mask.shape[:-1], 121), dtype=np.complex64)
    spec[:, :, :, 59:62] = mask * entry["z"]
    return np.array([neural.decode(z, 1.0, 2965, 70) for z in spec])


def score(events, pairs, bank, ids, w, name, decoder=None):
    rows = []
    for i in ids:
        e = events.loc[i]
        s, n, p = (pairs[k][i] for k in ("reference", "noise", "nominal_p"))
        for level, y in zip(LEVELS, predict(bank[i], w, decoder)):
            rows.append(
                dict(
                    event_id=e.event_id,
                    array=e["array"],
                    level=level,
                    method=name,
                    **all_scores(s, mix(s, n, level), y, p),
                )
            )
    return rows


def run():
    ev, p = load(ROOT / "data/prepared")
    dev = ev[ev.split.isin(["train", "validation"])]
    net = cache(dev, p)
    bank = {}
    for i, e in dev.iterrows():
        with np.load(CACHE / f"{e.event_id}.npz") as z:
            bank[i] = dict(z)
    w0 = net.layers[-2].get_weights()[0][0, 0]
    tr = list(dev[dev.split == "train"].index)
    va = list(dev[dev.split == "validation"].index)
    roles = []
    records = []
    fits = []
    validation = []
    for fold, (a, b, c) in enumerate([(tr, va, va), *folds(dev)]):
        for role, ids in [("fit", a), ("stop", b), ("score", c if fold else [])]:
            roles.extend(
                dict(
                    fold=fold,
                    role=role,
                    event_id=ev.at[i, "event_id"],
                    array=ev.at[i, "array"],
                )
                for i in ids
            )
        head, epoch, loss = head_fit(net, bank, a, b)
        decoder, de, dl = decoder_fit(net, bank, a, b)
        fits.extend(
            [
                dict(fold=fold, method=name, epoch=e, validation_loss=l)
                for name, e, l in [("Fine-tuned", epoch, loss), ("Decoder", de, dl)]
            ]
        )
        if not fold:
            np.testing.assert_allclose(
                head, np.load(ROOT / "models/das_head.npz")["kernel"][0, 0], atol=1e-5
            )
            np.savez(
                MODEL,
                **{
                    f"{name}_{j}": weight
                    for name in LAYER_NAMES
                    for j, weight in enumerate(decoder.get_layer(name).get_weights())
                },
            )
        block = (
            score(ev, p, bank, c, w0, "Zero-shot")
            + score(ev, p, bank, c, head, "Fine-tuned")
            + score(ev, p, bank, c, None, "Decoder", decoder)
        )
        for r in block:
            r["fold"] = fold
        (records if fold else validation).extend(block)
        print("finished fold", fold, flush=True)
    for name, rows in [("cv", records), ("validation", validation)]:
        d = pd.DataFrame(rows)
        d.to_csv(OUT / f"{name}_events.csv", index=False)
        aggregate(d, ["method", "level"], MEASURES).to_csv(
            OUT / f"{name}_summary.csv", index=False
        )
    pd.DataFrame(roles).to_csv(OUT / "folds.csv", index=False)
    pd.DataFrame(fits).to_csv(OUT / "fits.csv", index=False)
    aggregate(
        pd.DataFrame(records),
        ["fold", "method", "level"],
        ["gain_db", "si_gain_db", "cc", "pick_mae_ms"],
    ).to_csv(OUT / "cv_folds.csv", index=False)
    return


def sizes():
    ev, p = load(ROOT / "data/prepared")
    dev = ev[ev.split.isin(["train", "validation"])]
    net = cache(dev, p)
    bank = {}
    for i, e in dev.iterrows():
        with np.load(CACHE / f"{e.event_id}.npz") as z:
            bank[i] = dict(z)
    val = list(dev[dev.split == "validation"].index)
    rows = []
    draws = []
    for seed in range(SEED, SEED + 5):
        rng = np.random.default_rng(seed)
        groups = [
            rng.permutation(g.index)
            for _, g in dev[dev.split == "train"].groupby("array")
        ]
        for count in [2, 6, 12, 24]:
            ids = sorted(np.concatenate([g[: count // 2] for g in groups]))
            w, _, _ = head_fit(net, bank, ids, val, seed=seed)
            measured = score(ev, p, bank, val, w, "Fine-tuned")
            draws.extend(
                dict(seed=seed, count=count, event_id=ev.at[i, "event_id"]) for i in ids
            )
            rows.extend(dict(r, seed=seed, count=count) for r in measured)
        print("finished size seed", seed, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "size_events.csv", index=False)
    pd.DataFrame(draws).to_csv(OUT / "size_draws.csv", index=False)
    d.groupby(["seed", "count", "level"], as_index=False)[
        ["gain_db", "si_gain_db", "cc"]
    ].mean().to_csv(OUT / "size.csv", index=False)


if __name__ == "__main__":
    import sys

    sizes() if "--sizes" in sys.argv else run()
