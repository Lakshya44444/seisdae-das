"""Denoisers compared in V4 and the scores used to compare them.

All gathers are (channels, 6001) arrays at 100 Hz. seisDAE runs trace by trace,
exactly as in the original paper; f-k and the cascade use the whole gather.
"""

import os
import pickle
import sys
import importlib
from pathlib import Path

import numpy as np
from scipy.signal import butter, istft, sosfiltfilt, stft

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
FS = 100


# ---------------------------------------------------------------- classical
def bandpass(x, band):
    sos = butter(3, band, fs=FS, btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def fk(x, dx, vmin):
    """Fan filter: keep apparent velocities above ~vmin (cosine taper 0.8-1.2 vmin)."""
    nc, nt = x.shape
    padded = np.pad(x, ((nc // 2, nc // 2), (nt // 8, nt // 8)), mode="reflect")
    k = np.abs(np.fft.fftfreq(padded.shape[0], dx))[:, None]
    f = np.fft.rfftfreq(padded.shape[1], 1 / FS)[None, :]
    relative = k * vmin / np.maximum(f, 1e-12)
    weight = np.clip((1.2 - relative) / 0.4, 0, 1)
    y = np.fft.irfft2(np.fft.rfft2(padded) * weight, s=padded.shape, axes=(0, 1))
    return y[nc // 2: nc // 2 + nc, nt // 8: nt // 8 + nt]


# ------------------------------------------------------------------ seisDAE
def read_config(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class SeisDAE:
    """Signal-mask inference with the STFT settings stored in the model config.

    Same preprocessing as src/neural.py, which V3 verified against the original
    seisDAE `predict` (relative peak error 3.2e-7). `check_upstream` repeats that
    comparison for any model.
    """

    def __init__(self, h5, config):
        self.h5, self.config = Path(h5), Path(config)
        cfg = read_config(self.config)
        if cfg.get("cwt") or int(cfg["ts_length"]) != 6001 or not np.isclose(cfg["dt"], 1 / FS):
            raise ValueError(f"{config}: expected STFT model, 6001 samples, dt=0.01; got {cfg}")
        self.n = int(cfg["ts_length"])
        self.spec = dict(fs=FS, nfft=int(cfg["kwargs"]["nfft"]), nperseg=int(cfg["kwargs"]["nperseg"]))
        import tf_keras

        self.net = tf_keras.models.load_model(str(self.h5), compile=False)

    def encode(self, x):
        x = np.atleast_2d(x).astype("float32")
        x = x - x.mean(axis=1, keepdims=True)
        _, _, z = stft(x, **self.spec)
        feats = np.stack((z.real, z.imag), axis=-1)
        feats /= np.maximum(np.abs(feats).max(axis=(1, 2), keepdims=True), 1e-30)
        return feats, z

    def __call__(self, x, batch=32):
        x = np.atleast_2d(x)
        if x.shape[1] != self.n:
            raise ValueError(f"seisDAE needs {self.n} samples, got {x.shape[1]}")
        out = []
        for j in range(0, len(x), batch):
            feats, z = self.encode(x[j:j + batch])
            mask = self.net(feats, training=False).numpy()[..., 0]
            _, y = istft(z * mask, **self.spec)
            out.append(y[:, : self.n])
        return np.concatenate(out).astype(np.float64)

    def check_upstream(self, checkout, trace):
        for name in ("", ".layers", ".utils", ".models"):
            sys.modules.setdefault("keras" + name, importlib.import_module("tf_keras" + name))
        sys.path.insert(0, str(Path(checkout).resolve()))
        from prediction import predict

        ref = predict(str(self.h5), str(self.config), [trace], loaded_model=self.net)[0][0, :, 0]
        mine = self(trace[None])[0]
        return float(np.max(np.abs(mine - ref)) / np.max(np.abs(ref)))


# ------------------------------------------------------------------- scores
def si_sdr(reference, estimate):
    r = reference - reference.mean(-1, keepdims=True)
    y = estimate - estimate.mean(-1, keepdims=True)
    den = np.linalg.norm(r, axis=-1) * np.linalg.norm(y, axis=-1)
    cosine = np.divide((r * y).sum(-1), den, out=np.full(len(r), np.nan), where=den > 0)
    aligned = np.clip(cosine ** 2, 1e-12, 1 - 1e-12)
    return 10 * np.log10(aligned / (1 - aligned))


def correlation(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide((a * b).sum(-1), den, out=np.full(len(a), np.nan), where=den > 0)


def scores(s, x, y, window):
    """Gather-level scores of estimate y for reference s and input x.

    gain_db: ordinary error reduction (rewards pure attenuation, see Scalar row).
    si_gain_db: mean per-channel SI-SDR change (unchanged by any per-channel scale).
    *_event: the same inside the earthquake window only.
    """
    out = {}
    for tag, sl in (("", slice(None)), ("_event", window)):
        a, b, c = s[:, sl], x[:, sl], y[:, sl]
        out["gain_db" + tag] = 10 * np.log10(np.sum((b - a) ** 2) / max(np.sum((c - a) ** 2), 1e-30))
        out["si_gain_db" + tag] = float(np.nanmean(si_sdr(a, c) - si_sdr(a, b)))
        out["cc" + tag] = float(np.nanmean(correlation(a, c)))
    return out


def mix(ref, noise, level_db):
    """One scale per gather keeps relative channel amplitudes and moveout."""
    ps, pn = np.sum(ref ** 2), np.sum(noise ** 2)
    return ref + noise * np.sqrt(ps / (pn * 10.0 ** (level_db / 10)))


def coherence(y, window):
    """Mean zero-lag correlation of neighbouring channels inside the window."""
    w = y[:, window]
    return float(np.nanmean(correlation(w[:-1], w[1:])))
