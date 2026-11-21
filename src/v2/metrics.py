"""Additional waveform scores; arrival quantities are proxies tied to the clip centre."""

import numpy as np
from ..metrics import scores


def si_sdr(reference, estimate):
    r = np.asarray(reference, dtype=np.float64)
    y = np.asarray(estimate, dtype=np.float64)
    r = r - r.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    # Normalization makes the numerical floor independent of signal scale.
    den = np.linalg.norm(r, axis=-1) * np.linalg.norm(y, axis=-1)
    cosine = np.divide((r * y).sum(-1), den, out=np.full(len(r), np.nan), where=den > 0)
    aligned = np.clip(cosine**2, 1e-12, 1 - 1e-12)
    return 10 * np.log10(aligned / (1 - aligned))


def arrival(reference, estimate, picks):
    peaks = []
    hits = total = 0
    for s, y, p in zip(reference, estimate, picks):
        p = int(p)
        # Subtract each trace's own 5-sample pre-onset baseline.
        r = s[p : p + 15] - s[p - 5 : p].mean()
        v = y[p : p + 15] - y[p - 5 : p].mean()
        a, b = np.max(np.abs(r)), np.max(np.abs(v))
        if a == 0:
            continue
        peaks.append(b / a)
        # Compare the first signed excursion beyond 20% of each local peak.
        # Zero output is a missed polarity and counts as disagreement.
        total += 1
        if b > 0:
            sr = r[np.flatnonzero(abs(r) >= 0.2 * a)[0]]
            sy = v[np.flatnonzero(abs(v) >= 0.2 * b)[0]]
            hits += int(np.sign(sr) == np.sign(sy))
    return dict(
        peak_ratio=np.mean(peaks) if peaks else np.nan,
        polarity_hits=hits,
        polarity_total=total,
    )


def all_scores(s, x, y, p):
    change = si_sdr(s, y) - si_sdr(s, x)
    return {
        **scores(s, x, y, p),
        "si_gain_db": np.nanmean(change),
        "si_valid": int(np.isfinite(change).sum()),
        **arrival(s, y, p),
    }
