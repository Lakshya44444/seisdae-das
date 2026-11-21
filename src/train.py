"""Fine-tune the two output masks using only train and validation earthquakes."""

import argparse

import numpy as np
import pandas as pd
from scipy.special import softmax

from .data import LEVELS, ROOT, SEED, load, mix
from .neural import encode, model


def feature_bank(extractor, indices, pairs):
    features, targets = [], []
    for i in indices:
        ref, noise = pairs["reference"][i], pairs["noise"][i]
        for level in LEVELS:
            x = mix(ref, noise, level)
            # Six channels per event to limit spatial redundancy.
            subset = slice(0, None, 4)
            inp, _, _ = encode(x[subset])
            _, s, _ = encode(ref[subset])
            _, n, _ = encode((x - ref)[subset])
            active = (np.abs(s) + np.abs(n)).max(axis=(0, 1)) > 0
            h = extractor(inp, training=False).numpy()[:, :, active, :]
            mag = np.stack((np.abs(s), np.abs(n)), axis=-1)[:, :, active]
            valid = mag.sum(axis=-1) > 0
            features.append(h[valid])
            targets.append(mag[valid] / mag.sum(axis=-1)[valid, None])
    return np.concatenate(features), np.concatenate(targets)


def train(folder, out, epochs=40):
    import tf_keras

    events, pairs = load(folder)
    net = model()
    layer = net.layers[-2]
    extractor = tf_keras.Model(net.input, layer.input)
    train_x, train_y = feature_bank(
        extractor, events.index[events.split == "train"], pairs
    )
    val_x, val_y = feature_bank(
        extractor, events.index[events.split == "validation"], pairs
    )
    (kernel,) = layer.get_weights()
    best, history, best_epoch, minimum = fit_head(
        train_x, train_y, val_x, val_y, kernel, epochs
    )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history, columns=["epoch", "training_loss", "validation_loss"]).to_csv(
        out / "loss.csv", index=False
    )
    np.savez(
        out / "das_head.npz",
        kernel=best.reshape(kernel.shape).astype("float32"),
        epoch=best_epoch,
    )
    print(f"Selected epoch {best_epoch}; validation cross-entropy {minimum:.5f}")


def fit_head(train_x, train_y, val_x, val_y, kernel, epochs=40, seed=SEED):
    # A bias-free 1x1 convolution is a linear classifier shared by all bins.
    w = kernel[0, 0].astype("float64")

    def loss(x, y):
        prob = softmax(x @ w, axis=1)
        return -np.mean(np.sum(y * np.log(np.maximum(prob, 1e-15)), axis=1))

    rng = np.random.default_rng(seed)
    best, best_epoch, step = w.copy(), 0, 0
    history = [(0, loss(train_x, train_y), loss(val_x, val_y))]
    minimum = history[0][2]
    m, v = np.zeros_like(w), np.zeros_like(w)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_x))
        for j in range(0, len(order), 2048):
            ids = order[j : j + 2048]
            x, y = train_x[ids], train_y[ids]
            grad = x.T @ (softmax(x @ w, axis=1) - y) / len(ids)
            step += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad**2
            w -= 1e-4 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        history.append((epoch, loss(train_x, train_y), loss(val_x, val_y)))
        if history[-1][2] < minimum:
            best, minimum, best_epoch = w.copy(), history[-1][2], epoch
        if epoch - best_epoch >= 6:
            break
    return best, history, best_epoch, minimum


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    train(args.data, args.out)
