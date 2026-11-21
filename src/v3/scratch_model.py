"""Train a compact residual 1-D U-Net independently in each population fold."""
import numpy as np
import pandas as pd

from ..data import ROOT, LEVELS, SEED, load, mix
from ..v2.metrics import all_scores
from ..v2.stats import aggregate
from .experiments import CFG, OUT
from .multi_data import DEST


def build(seed=SEED):
    import tf_keras as k
    init = lambda offset: k.initializers.GlorotUniform(seed=seed + offset)
    source = k.Input((70, 1))
    skip = k.layers.Conv1D(8, 5, padding="same", activation="relu", kernel_initializer=init(0))(source)
    down = k.layers.MaxPooling1D(2)(skip)
    deep = k.layers.Conv1D(16, 5, padding="same", activation="relu", kernel_initializer=init(1))(down)
    up = k.layers.UpSampling1D(2)(deep)
    merged = k.layers.Concatenate()([skip, up])
    decoded = k.layers.Conv1D(8, 5, padding="same", activation="relu", kernel_initializer=init(2))(merged)
    residual = k.layers.Conv1D(1, 1, kernel_initializer=init(3))(decoded)
    return k.Model(source, k.layers.Add()([source, residual]))


def fold_ids(events):
    rng = np.random.default_rng(SEED)
    partitions = [np.array_split(rng.permutation(group.index), 5) for _, group in events.groupby("array")]
    for j in range(5):
        held = np.sort(np.concatenate([blocks[j] for blocks in partitions]))
        yield np.setdiff1d(events.index, held), held


def run():
    import tensorflow as tf
    import tf_keras as k

    e, p = load(ROOT / "data/prepared")
    with np.load(DEST / "aperture2.npz") as z:
        parts = (p, dict(z))
    # Normalization uses the noisy input only. Restore its scale before evaluating.
    mixtures = np.asarray([[[mix(part["reference"][i], part["noise"][i], lv)
                             for lv in LEVELS] for part in parts] for i in e.index], dtype="float32")
    refs = np.asarray([part["reference"] for part in parts], dtype="float32").swapaxes(0, 1)
    scales = np.maximum(np.sqrt(np.mean(mixtures**2, axis=-1, keepdims=True)), 1e-12)
    x = mixtures / scales
    y = refs[:, :, None] / scales
    cfg = CFG["scratch"]
    rows, fits = [], []
    for fold, (train, held) in enumerate(fold_ids(e), 1):
        k.backend.clear_session()
        k.utils.set_random_seed(SEED + fold)
        rng = np.random.default_rng(SEED + fold)
        model = build(SEED + fold)
        opt = k.optimizers.Adam(cfg["learning_rate"])

        @tf.function
        def update(a, b):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(model(a, training=True) - b))
            opt.apply_gradients(zip(tape.gradient(loss, model.trainable_weights), model.trainable_weights))

        for _ in range(cfg["steps"]):
            batch = cfg["batch_size"]
            ids = (rng.choice(train, batch), rng.integers(2, size=batch),
                   rng.integers(len(LEVELS), size=batch), rng.integers(24, size=batch))
            update(x[ids][...,None], y[ids][...,None])
        fits.append(dict(fold=fold, weights=model.count_params(), steps=cfg["steps"], training_events=len(train)))
        for i in held:
            estimate = model(x[i].reshape(-1,70,1), training=False).numpy().reshape(x[i].shape) * scales[i]
            for j, lv in enumerate(LEVELS):
                scores = [all_scores(part["reference"][i], mixtures[i,a,j], estimate[a,j], part["nominal_p"][i])
                          for a, part in enumerate(parts)]
                rows.append(dict(event_id=e.at[i,"event_id"], array=e.at[i,"array"], fold=fold, level=lv,
                                 method="Scratch U-Net", **{m: np.mean([s[m] for s in scores]) for m in ["gain_db","si_gain_db","cc"]}))
        print(f"Scratch fold {fold} complete", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "scratch_events.csv", index=False)
    pd.DataFrame(fits).to_csv(OUT / "scratch_fits.csv", index=False)
    aggregate(d, ["method","level"], ["gain_db","si_gain_db","cc"]).to_csv(OUT / "scratch_summary.csv", index=False)


if __name__ == "__main__":
    run()
