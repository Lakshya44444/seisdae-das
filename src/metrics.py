import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt

from .data import FS


def bandpass(x):
    # Match the acquisition study's published P-wave processing band.
    return sosfiltfilt(butter(3, (1, 10), fs=FS, btype="bandpass", output="sos"), x)


def wavelet(x):
    # For 70 samples db8 permits two useful decomposition levels.
    coeffs = pywt.wavedec(x, "db8", axis=-1)
    high = coeffs[-1]
    mad = np.median(
        np.abs(high - np.median(high, axis=-1, keepdims=True)), axis=-1, keepdims=True
    )
    threshold = mad / 0.67448975 * np.sqrt(2 * np.log(x.shape[1]))
    detail = [np.sign(c) * np.maximum(np.abs(c) - threshold, 0) for c in coeffs[1:]]
    return pywt.waverec([coeffs[0], *detail], "db8", axis=-1)[:, : x.shape[1]]


def fk(x, dx=10.0, vmin=500.0):
    # Symmetric padding reduces wraparound at the ends of a short aperture.
    nc, nt = x.shape
    padded = np.pad(x, ((nc // 2, nc // 2), (nt // 2, nt // 2)), mode="reflect")
    k = np.abs(np.fft.fftfreq(len(padded), dx))[:, None]
    f = np.fft.rfftfreq(padded.shape[1], 1 / FS)[None, :]
    relative = k * vmin / np.maximum(f, 1e-12)
    weight = np.clip((1.2 - relative) / 0.4, 0, 1)
    spec = np.fft.rfft2(padded) * weight
    return np.fft.irfft2(spec, s=padded.shape, axes=(0, 1))[
        nc // 2 : nc // 2 + nc, nt // 2 : nt // 2 + nt
    ]


def pick(x, nominal):
    """First causal 4/12-sample energy ratio >= 1.8 near the nominal P centre."""
    energy = np.r_[0.0, np.cumsum(np.asarray(x, dtype="float64") ** 2)]
    idx = np.arange(max(11, int(nominal) - 8), min(len(x), int(nominal) + 25))
    short = (energy[idx + 1] - energy[idx - 3]) / 4
    long = (energy[idx + 1] - energy[idx - 11]) / 12
    hits = idx[short >= 1.8 * np.maximum(long, 1e-30)]
    return hits[0] if len(hits) else np.nan


def scores(s, x, y, p):
    e0 = np.sum((x - s) ** 2)
    e1 = np.sum((y - s) ** 2)
    r = s - s.mean(axis=1, keepdims=True)
    v = y - y.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(r, axis=1) * np.linalg.norm(v, axis=1)
    cc = np.divide(
        (r * v).sum(axis=1), denom, out=np.full(len(s), np.nan), where=denom > 0
    )
    a = np.array([pick(trace, q) for trace, q in zip(s, p)])
    b = np.array([pick(trace, q) for trace, q in zip(y, p)])
    paired = np.isfinite(a) & np.isfinite(b)
    return {
        "gain_db": 10 * np.log10(e0 / max(e1, 1e-30)),
        "nrmse": np.sqrt(e1 / np.sum(s * s)),
        "cc": np.nanmean(cc),
        "rmse": np.sqrt(np.mean((y - s) ** 2)),
        "reference_picks": np.isfinite(a).sum(),
        "matched_picks": paired.sum(),
        "pick_abs_ms_sum": np.abs(a[paired] - b[paired]).sum() * 1000 / FS,
    }
