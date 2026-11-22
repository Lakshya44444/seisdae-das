"""Reference-free measure: slow (traffic) energy removed versus fast (earthquake) energy kept.

  python -m src.fk_energy

For each test-day earthquake window (no overlapping catalogued event), the
segment from 0.5 s before to 20 s after the predicted P arrival is moved to the
frequency-wavenumber domain (1-45 Hz). Energy with apparent velocity below
250 m/s is "slow" (traffic crosses the block at tens of m/s); energy above
1000 m/s is "fast" (a local earthquake crosses it at km/s). For each method:

  kept_fast   = fast energy after / fast energy before
  kept_slow   = slow energy after / slow energy before
  selectivity = 10 log10(kept_fast / kept_slow)   (dB; independent of output scale)

Methods that contain an f-k filter remove slow energy by construction, so their
selectivity is high by design; the measure is informative mainly for the
single-channel methods (bandpass, seisDAE). The fast region also holds fast
noise. Settings come from the detection tuning (non-test days if available).
Writes results/fk_energy_rows.csv and results/fk_energy_summary.csv.
"""

import re

import numpy as np
import pandas as pd

from .build import load_gather, manifest
from .das import CFG, geometry
from .detection_stats import ORDER
from .evaluate import OUT, Runner
from .methods import FS, ROOT, bandpass, fk

SLOW, FAST, BAND = 250.0, 1000.0, (1.0, 45.0)


def regions(segment, dx):
    """Energy in the slow and fast apparent-velocity regions of one segment."""
    power = np.abs(np.fft.rfft2(segment - segment.mean(axis=1, keepdims=True))) ** 2
    k = np.abs(np.fft.fftfreq(segment.shape[0], dx))[:, None]
    f = np.fft.rfftfreq(segment.shape[1], 1 / FS)[None, :]
    inband = (f >= BAND[0]) & (f <= BAND[1])
    velocity = np.divide(f, k, out=np.full(np.broadcast_shapes(k.shape, f.shape), np.inf), where=k > 0)
    return float(power[inband & (velocity < SLOW)].sum()), float(power[inband & (velocity > FAST)].sum())


def chosen_settings():
    for name in ("detection_tuning_validation_nontest.csv", "detection_tuning_validation.csv"):
        path = OUT / name
        if path.exists():
            t = pd.read_csv(path)
            best = t[t.chosen].set_index("family").method.to_dict()
            band = [float(v) for v in re.findall(r"[\d.]+", best["Bandpass"])[:2]]
            vel = lambda s: float(re.findall(r"[\d.]+", s.split("+")[-1])[0])
            return dict(source=name, band=band, fk=vel(best["f-k"]), cascade=vel(best["seisDAE (DAS) + f-k"]))
    raise SystemExit("Run python -m src.detection_tuning first")


def main():
    m = manifest()
    dx = geometry()[2]
    runner = Runner(ROOT / "models/das_seisdae.h5", dx)
    st = chosen_settings()
    print("settings:", st)
    overlap = m["overlap"].fillna(0)
    ev = m[(m.kind != "noise") & (m.split == "test") & (overlap == 0)]
    itp = CFG["window"]["p_index"]
    seg = slice(itp - FS // 2, itp + 20 * FS)
    rows = []
    for k, r in enumerate(ev.itertuples()):
        x = load_gather(r.id)
        das = runner.das(x)
        outputs = {"Noisy": x, "Bandpass": bandpass(x, st["band"]), "f-k": fk(x, dx, st["fk"]),
                   "seisDAE (released)": runner.gr(x), "seisDAE (DAS-trained)": das,
                   "seisDAE (DAS) + f-k": fk(das, dx, st["cascade"])}
        slow0, fast0 = regions(x[:, seg], dx)
        for method, y in outputs.items():
            slow, fast = regions(y[:, seg], dx)
            rows.append(dict(window=r.id, magnitude=r.magnitude, method=method, kept_fast=fast / fast0,
                             kept_slow=slow / slow0, selectivity_db=10 * np.log10((fast / fast0) / (slow / slow0))))
        print(f"window {k + 1}/{len(ev)}", end="\r")
    print()
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "fk_energy_rows.csv", index=False)
    rng = np.random.default_rng(CFG["seed"])
    wide = d.pivot(index="window", columns="method", values="selectivity_db")
    idx = rng.integers(0, len(wide), size=(CFG["evaluation"]["bootstrap_draws"], len(wide)))
    summary = []
    for method in [o for o in ORDER if o in wide.columns]:
        boot = wide[method].to_numpy()[idx].mean(axis=1)
        g = d[d.method == method]
        summary.append(dict(method=method, windows=len(g), median_kept_fast=g.kept_fast.median(),
                            median_kept_slow=g.kept_slow.median(), mean_selectivity_db=wide[method].mean(),
                            lo=np.quantile(boot, 0.025), hi=np.quantile(boot, 0.975)))
    for a, b in (("seisDAE (DAS-trained)", "Bandpass"), ("seisDAE (DAS-trained)", "Noisy")):
        diff = (wide[a] - wide[b]).to_numpy()
        boot = diff[idx].mean(axis=1)
        summary.append(dict(method=f"{a} minus {b}", windows=len(diff), mean_selectivity_db=diff.mean(),
                            lo=np.quantile(boot, 0.025), hi=np.quantile(boot, 0.975)))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "fk_energy_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
