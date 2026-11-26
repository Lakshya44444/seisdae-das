"""jDAS (van den Ende et al., 2023, IEEE TNNLS 34, 3371-3384) on the Ridgecrest DAS windows.

jDAS is a self-supervised, multi-channel DAS denoiser. It hides one channel
of an 11-channel block and predicts it from its neighbours: a signal that is
coherent across channels can be predicted, incoherent noise cannot. No clean
labels are needed, so it can be fine-tuned on our own (noisy) training days.

  python -m src.jdas check                  rebuilt network = released file?
  python -m src.jdas train                  fine-tune the released weights (seed 2026)
  python -m src.jdas train --scratch        same training from random weights
  python -m src.jdas train --seed 2027      another seed
  python -m src.jdas evaluate               detection + synthetic tests for every jDAS model

Models
  models/jdas_pretrained.h5      released jDAS weights (MIT licence, models/jdas_LICENSE.txt)
  models/jdas/<name>.h5          our runs: finetune_s2026, scratch_s2026, ...

Settings (fixed before any test day was looked at)
  * The released model works at 50 Hz on blocks of 11 channels x 2048 samples.
    Our 100 Hz windows are resampled to 50 Hz (6001 -> 3001 samples), denoised,
    and resampled back. Content above 25 Hz is therefore removed.
  * 3001 samples are covered by two blocks, [0:2048] and [953:3001]; the overlap
    is cross-faded linearly. (The original code uses non-overlapping blocks.)
  * Input is divided by its standard deviation and the output multiplied back,
    as in the original code.
  * "+ bandpass" rows apply our bandpass 1-10 Hz after jDAS, as in the jDAS
    example notebook (postfilter=True, 1-10 Hz).
  * Training: train-split windows only (all kinds; no labels are used), early
    stopping on validation-split windows, Adam, MSE on the hidden channel,
    random polarity flip and time reversal as in the original data loader.

Writes
  results/jdas_detection_rows.csv, then the variants table (src/variants.py)
  results/synthetic_jdas_events.csv, synthetic_jdas_summary.csv, synthetic_jdas_paired.csv
  results/jdas_train_<name>.csv (loss per epoch)
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from .build import load_gather, manifest
from .das import CFG
from .methods import ROOT, bandpass

FS_MODEL = 50
NSUB = 11
NT = 2048
PRETRAINED = ROOT / "models/jdas_pretrained.h5"
RUNS = ROOT / "models/jdas"
POSTFILTER = (1, 10)


# ------------------------------------------------------------------ network
def build_unet(f0=4, blocks=4, kernel=(3, 5), lr=1e-4, seed=None):
    """The released jDAS U-Net, layer for layer (read from models/jdas_pretrained.h5).

    Encoder: two 3x5 convolutions + swish per level, then max-pool (1x4, stride 1)
    followed by a fixed [1,3,3,1]/8 blur with stride 4 (anti-aliased pooling).
    Filters 4, 8, 16, 32, 64. Decoder: bilinear upsampling x4 + skip connection.
    Input: the data block and a mask (0 on the hidden channel, 1 elsewhere).
    The output is kept only on the hidden channel.
    """
    import tensorflow as tf
    import tf_keras as keras
    from tf_keras import layers

    if seed is not None:
        keras.utils.set_random_seed(seed)

    class BlurPool(layers.Layer):
        """Fixed filter, no weights. Same quirk as the original: every output
        channel is the sum of the blurred input channels."""

        def call(self, x):
            c = x.shape[-1]
            a = np.array([1.0, 3.0, 3.0, 1.0]) / 8.0
            w = np.repeat(a, c * c).reshape(1, 4, c, c).astype("float32")
            return tf.nn.conv2d(x, w, strides=(1, 1, 4, 1), padding="SAME")

    made = []

    def new_conv(f):
        """Convolutions are named conv_00, conv_01, ... in creation order (used by load_released)."""
        made.append(layers.Conv2D(f, kernel, padding="same", name=f"conv_{len(made):02d}",
                                  kernel_initializer=keras.initializers.Orthogonal()))
        return made[-1]

    def conv(x, f):
        return layers.Activation("swish")(new_conv(f)(x))

    data = keras.Input((NSUB, NT, 1))
    mask = keras.Input((NSUB, NT, 1))
    x = layers.Multiply()([mask, data])
    f = f0
    x = conv(conv(x, f), f)
    skips = [x]
    for _ in range(blocks):
        x = layers.MaxPooling2D((1, 4), strides=(1, 1))(x)
        x = BlurPool()(x)
        f *= 2
        x = conv(conv(x, f), f)
        skips.append(x)
    for i in range(blocks):
        x = layers.UpSampling2D((1, 4), interpolation="bilinear")(x)
        f //= 2
        x = layers.Concatenate()([x, skips[-(i + 2)]])
        x = conv(conv(x, f), f)
    x = new_conv(1)(x)
    hidden = layers.Lambda(lambda m: 1.0 - m)(mask)
    model = keras.Model([data, mask], layers.Multiply()([hidden, x]))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def conv_layers(model):
    import tf_keras as keras

    return sorted((l for l in model.layers if isinstance(l, keras.layers.Conv2D)), key=lambda l: l.name)


def load_released(model, h5=PRETRAINED):
    """Copy the released weights layer by layer, matched by creation order.

    The released file names its convolutions conv2d_19 ... conv2d_37 in the order
    they were created, the same order as build_unet. Matching by that order (and
    checking every shape) avoids relying on Keras' layer sorting, where two
    8 -> 8 convolutions of encoder and decoder have the same shape.
    """
    import re

    import h5py

    with h5py.File(h5, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f
        names = [n.decode() if isinstance(n, bytes) else n for n in g.attrs["layer_names"]]
        convs = sorted((n for n in names if re.fullmatch(r"conv2d(_\d+)?", n)),
                       key=lambda n: int(n.split("_")[1]) if "_" in n else 0)
        saved = []
        for n in convs:
            wn = [w.decode() if isinstance(w, bytes) else w for w in g[n].attrs["weight_names"]]
            saved.append([np.asarray(g[n][w]) for w in wn])
    mine = conv_layers(model)
    if len(mine) != len(saved):
        raise ValueError(f"{len(saved)} convolutions in {h5}, {len(mine)} in build_unet")
    for layer, weights in zip(mine, saved):
        if [w.shape for w in layer.get_weights()] != [w.shape for w in weights]:
            raise ValueError(f"shape mismatch at {layer.name}")
        layer.set_weights(weights)
    return model


def load(h5):
    model = build_unet()
    if h5 == PRETRAINED:
        return load_released(model)
    model.load_weights(str(h5))
    return model


def check():
    """Rebuilt network vs the released file loaded as saved (same random input)."""
    import tf_keras as keras

    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, NSUB, NT, 1)).astype("float32")
    m = np.ones_like(x)
    m[np.arange(4), [0, 3, 5, 10]] = 0
    mine = load(PRETRAINED).predict([x, m], verbose=0)
    print(f"rebuilt model: output std {mine.std():.4f}, nonzero only on hidden channel: "
          f"{bool(np.all(mine[m == 1] == 0))}")
    try:
        ref = keras.models.load_model(str(PRETRAINED), compile=False).predict([x, m], verbose=0)
    except Exception as e:
        print(f"original file could not be loaded directly ({type(e).__name__}); only the check above ran")
        return
    print(f"largest difference to the original: {np.abs(mine - ref).max() / np.abs(ref).max():.2e} (relative)")


# ------------------------------------------------------------------ inference
def masked_blocks(data):
    """One sample per channel: its 11-channel block and the mask hiding it (as jDAS _J_construct)."""
    nch, nt = data.shape
    mid = NSUB // 2
    samples = np.zeros((nch, NSUB, nt, 1), dtype="float32")
    masks = np.ones_like(samples)
    for i in range(nch):
        start = min(max(i - mid, 0), nch - NSUB)
        samples[i, :, :, 0] = data[start:start + NSUB]
        masks[i, i - start] = 0
    return samples, masks


def reconstruct(predict, data):
    """J-invariant reconstruction of one (channels, 2048) block."""
    samples, masks = masked_blocks(data)
    return predict(samples, masks).sum(axis=1)[:, :, 0]


def chunked(predict, x50):
    """Cover any length >= 2048 with blocks; overlaps are cross-faded linearly."""
    nt = x50.shape[1]
    starts = list(range(0, nt - NT + 1, NT))
    if starts[-1] != nt - NT:
        starts.append(nt - NT)
    out, weight = np.zeros_like(x50), np.zeros(nt)
    for k, s in enumerate(starts):
        w = np.ones(NT)
        if k > 0:
            lap = starts[k - 1] + NT - s
            w[:lap] = np.linspace(0, 1, lap + 2)[1:-1]
        if k < len(starts) - 1:
            lap = s + NT - starts[k + 1]
            w[NT - lap:] = np.linspace(1, 0, lap + 2)[1:-1]
        out[:, s:s + NT] += w * reconstruct(predict, x50[:, s:s + NT])
        weight[s:s + NT] += w
    return out / weight


class JDAS:
    """Denoiser with the same interface as methods.SeisDAE: (channels, 6001) at 100 Hz in and out."""

    def __init__(self, h5, postfilter=False, batch=64):
        self.h5, self.postfilter, self.batch = h5, postfilter, batch
        self.model = load(h5)

    def predict(self, samples, masks):
        return self.model.predict([samples, masks], batch_size=self.batch, verbose=0)

    def __call__(self, x):
        x = np.atleast_2d(x).astype(np.float64)
        n = x.shape[1]
        scale = x.std()
        x50 = resample_poly(x / scale, 1, 100 // FS_MODEL, axis=1)
        y = resample_poly(chunked(self.predict, x50), 100 // FS_MODEL, 1, axis=1)[:, :n] * scale
        return bandpass(y, POSTFILTER) if self.postfilter else y


def models():
    """Name -> (weights, postfilter) for every jDAS model that exists."""
    found = {}
    files = {"jdas_pretrained": PRETRAINED}
    files.update({f"jdas_{h5.stem}": h5 for h5 in sorted(RUNS.glob("*.h5"))})
    for name, h5 in files.items():
        if h5.exists():
            found[name] = (h5, False)
            found[name + "_bp"] = (h5, True)
    return found


def denoisers():
    nets = {}
    for name, (h5, post) in models().items():
        nets[name] = JDAS(h5, postfilter=post)
    return nets


# ------------------------------------------------------------------ training
def gathers_50hz(split):
    m = manifest()
    ids = m[m.split == split].id.tolist()
    out = []
    for k, i in enumerate(ids):
        g = load_gather(i)
        out.append((resample_poly(g / g.std(), 1, 100 // FS_MODEL, axis=1)).astype("float32"))
        print(f"loading {split} {k + 1}/{len(ids)}", end="\r")
    print()
    return out


def patches(gathers, n, rng):
    """n random training samples, the same augmentations as the original jDAS DataLoader."""
    samples = np.zeros((n, NSUB, NT, 1), dtype="float32")
    masks = np.ones_like(samples)
    for b in range(n):
        g = gathers[rng.integers(len(gathers))]
        c = rng.integers(0, g.shape[0] - NSUB + 1)
        t = rng.integers(0, g.shape[1] - NT + 1)
        block = g[c:c + NSUB, t:t + NT]
        if rng.integers(2):
            block = block[:, ::-1]
        samples[b, :, :, 0] = block * (rng.integers(2) * 2 - 1)
        masks[b, rng.integers(NSUB)] = 0
    return (samples, masks), samples * (1 - masks)


def train(seed, scratch, epochs, steps, batch, patience, lr):
    import tf_keras as keras

    class Batches(keras.utils.Sequence):
        """New random mini-batches every time, so memory stays small."""

        def __init__(self, gathers, rng):
            super().__init__()
            self.gathers, self.rng = gathers, rng

        def __len__(self):
            return steps

        def __getitem__(self, i):
            return patches(self.gathers, batch, self.rng)

    name = f"{'scratch' if scratch else 'finetune'}_s{seed}"
    RUNS.mkdir(parents=True, exist_ok=True)
    model = build_unet(lr=lr, seed=seed)
    if not scratch:
        load_released(model)
    tr, va = gathers_50hz("train"), gathers_50hz("validation")
    print(f"{len(tr)} train windows, {len(va)} validation windows")
    val_x, val_y = patches(va, 16 * batch, np.random.default_rng(seed + 1))
    data = Batches(tr, np.random.default_rng(seed))
    best = model.evaluate(val_x, val_y, batch_size=batch, verbose=0)
    print(f"{name}: validation loss before training {best:.4f}")
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
    pd.DataFrame(history).to_csv(ROOT / f"results/jdas_train_{name}.csv", index=False)
    print(f"saved models/jdas/{name}.h5 (best validation loss {best:.4f})")
    keras.backend.clear_session()


# ------------------------------------------------------------------ evaluation
def evaluate_detection(nets):
    from . import variants
    from .evaluate import OUT, detection_stat

    windows = pd.read_csv(OUT / "detection_tuning_rows.csv")[["window", "label"]].drop_duplicates()
    rows = []
    for k, r in enumerate(windows.itertuples()):
        g = load_gather(r.window)
        for name, net in nets.items():
            rows.append(dict(window=r.window, label=r.label, method=name, stat_db=detection_stat(net(g))))
        print(f"detection {k + 1}/{len(windows)}", end="\r")
    print()
    new = pd.DataFrame(rows)
    new.to_csv(OUT / "jdas_detection_rows.csv", index=False)
    old = pd.read_csv(OUT / "variants_rows.csv")
    old = old[~old.method.isin(new.method.unique())]
    pd.concat([old, new], ignore_index=True).to_csv(OUT / "variants_rows.csv", index=False)
    variants.main(rescore=False)


def evaluate_synthetic(nets):
    from .evaluate import E, OUT, bootstrap, event_window, good_channels, original, pairs
    from .methods import mix, scores

    m = original(manifest())
    ev = m[(m.kind == "strong") & (m.split == "test")]
    nz = m[(m.kind == "noise") & (m.split == "test")]
    jobs = pairs(ev, nz, E["noise_draws"], CFG["seed"] + 20)
    rows = []
    for k, (r, nid, draw) in enumerate(jobs):
        s, n, win, ok = load_gather(r.id), load_gather(nid), event_window(r), good_channels(r)
        for level in E["levels_db"]:
            x = mix(s, n, level)
            for name, net in nets.items():
                rows.append(dict(event=r.id, draw=draw, level=level, model=name,
                                 si_gain_db=scores(s[ok], x[ok], net(x)[ok], win)["si_gain_db"]))
        print(f"synthetic {k + 1}/{len(jobs)}", end="\r")
    print()
    d = pd.DataFrame(rows).groupby(["event", "level", "model"], as_index=False).si_gain_db.mean()
    d.to_csv(OUT / "synthetic_jdas_events.csv", index=False)

    wide = d.pivot_table(index=["event", "level"], columns="model", values="si_gain_db")
    seeds = pd.read_csv(OUT / "synthetic_seeds_events.csv").pivot_table(
        index=["event", "level"], columns="model", values="si_gain_db")
    old = pd.read_csv(OUT / "synthetic_events.csv").pivot_table(
        index=["event", "level"], columns="method", values="si_gain_db")
    base = {"released": seeds["released"], "Bandpass": old["Bandpass"], "f-k": old["f-k"]}
    if "scratch_s2026" in seeds:
        base["seisDAE DAS-trained"] = seeds["scratch_s2026"]
    for b, col in base.items():
        wide[b] = col.reindex(wide.index)

    summary, paired = [], []
    for name in nets:
        for level, g in wide[name].groupby(level="level"):
            mean, lo, hi = bootstrap(g.to_numpy(), E["bootstrap_draws"], CFG["seed"])
            summary.append(dict(model=name, level=level, si_gain_db=mean, lo=lo, hi=hi))
        for b in base:
            for level, g in (wide[name] - wide[b]).groupby(level="level"):
                g = g.dropna()
                mean, lo, hi = bootstrap(g.to_numpy(), E["bootstrap_draws"], CFG["seed"])
                paired.append(dict(model=name, minus=b, level=level, si_gain_db=mean, lo=lo, hi=hi,
                                   events=len(g), wins=int((g > 0).sum())))
    summary, paired = pd.DataFrame(summary), pd.DataFrame(paired)
    summary.to_csv(OUT / "synthetic_jdas_summary.csv", index=False)
    paired.to_csv(OUT / "synthetic_jdas_paired.csv", index=False)

    print(f"\nSynthetic test: {len(ev)} test earthquakes x {E['noise_draws']} noise draws")
    print("SI-SDR gain (dB) per input SNR. Brackets = 95% interval.\n")
    levels = E["levels_db"]
    print(f"{'Model':26}" + "".join(f"{str(lv) + ' dB':>20}" for lv in levels))
    show = [(b, wide[b]) for b in base] + [(n, wide[n]) for n in nets]
    for name, col in show:
        cells = []
        for lv in levels:
            mean, lo, hi = bootstrap(col.xs(lv, level="level").dropna().to_numpy(), E["bootstrap_draws"],
                                     CFG["seed"])
            cells.append(f"{mean:+.2f} [{lo:+.1f},{hi:+.1f}]")
        print(f"{name:26}" + "".join(f"{c:>20}" for c in cells))
    print("\njDAS minus bandpass / minus f-k (dB, 'yes' = 95% interval excludes zero):")
    for name in nets:
        for b in ("Bandpass", "f-k"):
            p = paired[(paired.model == name) & (paired.minus == b)]
            cells = [f"{r.si_gain_db:+.2f} {'yes' if r.lo > 0 or r.hi < 0 else 'no'}" for r in p.itertuples()]
            print(f"  {name:22} - {b:9}" + "".join(f"{c:>12}" for c in cells))
    print("\nFull numbers: results/synthetic_jdas_summary.csv, results/synthetic_jdas_paired.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["check", "train", "evaluate"])
    p.add_argument("--seed", type=int, default=CFG["seed"])
    p.add_argument("--scratch", action="store_true", help="train from random weights instead of the released ones")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=100, help="mini-batches per epoch")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate (released model's optimizer used 1e-4)")
    p.add_argument("--only", choices=["detection", "synthetic"], help="run one of the two tests")
    a = p.parse_args()
    if a.command == "check":
        check()
        return
    if a.command == "train":
        train(a.seed, a.scratch, a.epochs, a.steps, a.batch, a.patience, a.lr)
        return
    nets = denoisers()
    print("jDAS models:", ", ".join(nets))
    if a.only != "synthetic":
        evaluate_detection(nets)
    if a.only != "detection":
        evaluate_synthetic(nets)


if __name__ == "__main__":
    main()
