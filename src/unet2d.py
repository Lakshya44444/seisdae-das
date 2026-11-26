"""Our own supervised 2-D U-Net: the whole gather (channels x time) at once.

seisDAE sees one channel at a time; jDAS sees 11 channels and learns without
clean targets. This model is trained directly for the task of the synthetic
test: input = recorded earthquake + recorded noise, target = the recorded
earthquake. It sees many channels together, so it can use moveout across the
fiber, and it learns what our noise (traffic, common-mode residue) looks like.

  python -m src.unet2d train                 train (seed 2026)
  python -m src.unet2d train --seed 2027     another seed
  python -m src.unet2d evaluate              detection + synthetic tests (same as the other models)

Data (no test day is used for training or for any choice)
  signal  clean earthquakes ("strong") of the train split, the same 35 events seisDAE used
  noise   noise windows of the train split (132)
  early stopping on a fixed set of mixtures from the validation split
Each training sample is a random crop of 48 channels x 1024 samples (10.24 s)
that contains the P arrival in 80 % of cases. The noise crop comes from a random
noise window and is scaled to a random SNR between -10 and +10 dB (the test
range). Augmentation: polarity flip, reversed channel order.
Input and target are divided by the input's standard deviation; loss = MSE.

Network: 3 levels, pooling 2 (channels) x 4 (time), 16-32-64-128 filters,
3 x 7 kernels, swish, linear output. Fully convolutional, so a whole
96 x 6001 window is denoised in one pass (time padded to 6016).

Models are saved in models/unet2d/s<seed>.h5 (weights).
"""

import argparse
import time

import numpy as np
import pandas as pd

from .build import load_gather, manifest
from .das import CFG
from .methods import ROOT

RUNS = ROOT / "models/unet2d"
CROP_CH, CROP_T = 48, 1024
POOL = (2, 4)
LEVELS = 3
P_INDEX = CFG["window"]["p_index"]


def build(f0=16, kernel=(3, 7), lr=1e-3, seed=None):
    import tf_keras as keras
    from tf_keras import layers

    if seed is not None:
        keras.utils.set_random_seed(seed)

    def conv(x, f):
        x = layers.Conv2D(f, kernel, padding="same", kernel_initializer="he_normal")(x)
        return layers.Activation("swish")(x)

    inp = keras.Input((None, None, 1))
    x, f, skips = inp, f0, []
    for _ in range(LEVELS):
        x = conv(conv(x, f), f)
        skips.append(x)
        x = layers.MaxPooling2D(POOL)(x)
        f *= 2
    x = conv(conv(x, f), f)
    for skip in reversed(skips):
        f //= 2
        x = layers.UpSampling2D(POOL)(x)
        x = layers.Concatenate()([x, skip])
        x = conv(conv(x, f), f)
    out = layers.Conv2D(1, 1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


class UNet2D:
    """Same interface as the other denoisers: (channels, 6001) in and out."""

    def __init__(self, h5):
        self.model = build()
        self.model.load_weights(str(h5))

    def __call__(self, x):
        x = np.atleast_2d(x).astype(np.float64)
        nc, nt = x.shape
        mc, mt = POOL[0] ** LEVELS, POOL[1] ** LEVELS
        pc, pt = (-nc) % mc, (-nt) % mt
        scale = x.std()
        z = np.pad(x / scale, ((0, pc), (0, pt)), mode="reflect")
        y = self.model.predict(z[None, :, :, None].astype("float32"), verbose=0)[0, :, :, 0]
        return y[:nc, :nt].astype(np.float64) * scale


# ------------------------------------------------------------------ training data
def load_split(split):
    m = manifest()
    ev = m[(m.kind == "strong") & (m.split == split)]
    nz = m[(m.kind == "noise") & (m.split == split)]
    sig = [load_gather(i).astype("float32") for i in ev.id]
    noise = [load_gather(i).astype("float32") for i in nz.id]
    print(f"{split}: {len(sig)} earthquakes, {len(noise)} noise windows")
    return sig, noise


def sample(sig, noise, n, rng):
    """n training pairs (input, target), each (CROP_CH, CROP_T, 1)."""
    x = np.zeros((n, CROP_CH, CROP_T, 1), dtype="float32")
    y = np.zeros_like(x)
    for b in range(n):
        s = sig[rng.integers(len(sig))]
        nz = noise[rng.integers(len(noise))]
        c = rng.integers(0, s.shape[0] - CROP_CH + 1)
        if rng.random() < 0.8:
            t = rng.integers(max(0, P_INDEX - CROP_T + 100), P_INDEX - 100)
        else:
            t = rng.integers(0, s.shape[1] - CROP_T + 1)
        cs = s[c:c + CROP_CH, t:t + CROP_T]
        c2 = rng.integers(0, nz.shape[0] - CROP_CH + 1)
        t2 = rng.integers(0, nz.shape[1] - CROP_T + 1)
        cn = nz[c2:c2 + CROP_CH, t2:t2 + CROP_T]
        level = rng.uniform(-10, 10)
        k = np.sqrt(np.sum(cs ** 2) / (np.sum(cn ** 2) * 10 ** (level / 10) + 1e-30))
        mix = cs + k * cn
        if rng.integers(2):
            mix, cs = mix[::-1], cs[::-1]
        sign = rng.integers(2) * 2 - 1
        scale = mix.std() + 1e-12
        x[b, :, :, 0] = sign * mix / scale
        y[b, :, :, 0] = sign * cs / scale
    return x, y


def train(seed, epochs, steps, batch, patience, lr):
    import tf_keras as keras

    class Batches(keras.utils.Sequence):
        def __init__(self, sig, noise, rng):
            super().__init__()
            self.sig, self.noise, self.rng = sig, noise, rng

        def __len__(self):
            return steps

        def __getitem__(self, i):
            return sample(self.sig, self.noise, batch, self.rng)

    RUNS.mkdir(parents=True, exist_ok=True)
    name = f"s{seed}"
    model = build(lr=lr, seed=seed)
    print(f"U-Net with {model.count_params():,} parameters")
    tr_sig, tr_noise = load_split("train")
    va_sig, va_noise = load_split("validation")
    val_x, val_y = sample(va_sig, va_noise, 256, np.random.default_rng(seed + 1))
    data = Batches(tr_sig, tr_noise, np.random.default_rng(seed))
    best = model.evaluate(val_x, val_y, batch_size=batch, verbose=0)
    print(f"validation loss before training {best:.4f}")
    model.save_weights(str(RUNS / f"{name}.h5"))
    history, wait = [dict(epoch=0, train_loss=np.nan, val_loss=best)], 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss = model.fit(data, epochs=1, shuffle=False, verbose=0).history["loss"][0]
        val = model.evaluate(val_x, val_y, batch_size=batch, verbose=0)
        history.append(dict(epoch=ep, train_loss=loss, val_loss=val))
        mark = ""
        if val < best:
            best, wait, mark = val, 0, "  (best, saved)"
            model.save_weights(str(RUNS / f"{name}.h5"))
        else:
            wait += 1
        print(f"epoch {ep:3d}  train {loss:.4f}  validation {val:.4f}  {time.time() - t0:.0f} s{mark}")
        if wait >= patience:
            print(f"stopped: no validation improvement for {patience} epochs")
            break
    pd.DataFrame(history).to_csv(ROOT / f"results/unet2d_train_{name}.csv", index=False)
    print(f"saved models/unet2d/{name}.h5 (best validation loss {best:.4f})")


# ------------------------------------------------------------------ evaluation
def nets():
    return {f"unet2d_{h5.stem}": UNet2D(h5) for h5 in sorted(RUNS.glob("*.h5"))}


def outputs(models, x):
    return {name: net(x) for name, net in models.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["train", "evaluate"])
    p.add_argument("--seed", type=int, default=CFG["seed"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps", type=int, default=50, help="mini-batches per epoch")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--only", choices=["detection", "synthetic"])
    a = p.parse_args()
    if a.command == "train":
        train(a.seed, a.epochs, a.steps, a.batch, a.patience, a.lr)
        return
    from .jdas import evaluate_detection, evaluate_synthetic

    models = nets()
    print("U-Net models:", ", ".join(models))
    if a.only != "synthetic":
        evaluate_detection(models, out_fn=outputs, tag="unet2d")
    if a.only != "detection":
        evaluate_synthetic(models, out_fn=outputs, names_fn=lambda m: list(m), tag="unet2d")


if __name__ == "__main__":
    main()
