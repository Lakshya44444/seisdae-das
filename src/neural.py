"""Adapter for the released gr_mixed_stft.h5 model, pinned in models/."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import numpy as np
from scipy.signal import istft, stft

from .data import FS, ROOT

N = 6001
SPEC = {"fs": FS, "nfft": 198, "nperseg": 99}
PATH = ROOT / "models/gr_mixed_stft.h5"


def encode(x):
    x = np.atleast_2d(x).astype("float32")
    if x.shape[1] > N:
        raise ValueError("Crop to at most 6001 samples before prediction.")
    x = x - x.mean(axis=1, keepdims=True)
    start = (N - x.shape[1]) // 2
    padded = np.pad(x, ((0, 0), (start, N - start - x.shape[1])))
    _, _, z = stft(padded, **SPEC)
    feats = np.stack((z.real, z.imag), axis=-1)
    feats /= np.maximum(np.abs(feats).max(axis=(1, 2), keepdims=True), 1e-30)
    return feats, z, start


def decode(z, mask, start, length):
    _, y = istft(z * mask, **SPEC)
    return y[:, start : start + length]


def model(head=None):
    import tf_keras

    net = tf_keras.models.load_model(PATH, compile=False)
    if head is not None:
        with np.load(head) as h:
            net.layers[-2].set_weights([h["kernel"]])
    return net


def denoise(net, x, batch=8):
    y = []
    for j in range(0, len(x), batch):
        features, z, start = encode(x[j : j + batch])
        masks = net(features, training=False).numpy()[..., 0]
        y.append(decode(z, masks, start, x.shape[1]))
    return np.concatenate(y)
