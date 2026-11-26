"""Onset SNR of recorded test earthquakes, measured as in Steinberg et al. (2025).

  python -m src.onset_snr                 all test windows (73 earthquakes, 210 noise), every method
  python -m src.onset_snr --stats-only    recompute the summary from results/onset_snr_rows.csv

Steinberg et al. (2025, Pure Appl. Geophys., section 4.2) estimate the SNR
continuously: at each time, the maximum amplitude difference (peak-to-peak) in
a 0.5 s window is divided by the RMS of the 2 s before. We use the same
definition on every channel of the recorded (not mixed) test windows:

  onset SNR (dB) = 20 log10( max over t in [P - 0.5 s, P + 3 s] of SNR(t) )

with P the predicted P arrival (sample 1000), then the median over the 96
channels. No reference waveform is needed. The same number on the 210 test
noise windows (same time) shows how large it gets without an earthquake, so the
onset SNR is reported (a) for earthquakes, (b) for noise and (c) as the
detection AUC that separates the two.

This measure was added after the review of 26 November, to match the evaluation
of the seisDAE paper; it was not used to choose any setting or model.

Writes results/onset_snr_rows.csv, onset_snr_summary.csv, onset_snr_paired.csv.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from .build import load_gather, manifest
from .das import CFG
from .evaluate import E, OUT, bootstrap

FS = 100
P = CFG["window"]["p_index"]


def continuous_snr(x, win=0.5, pre=2.0):
    """SNR(t) per channel: peak-to-peak in [t, t+win] over RMS in [t-pre, t] (Steinberg et al., 2025)."""
    w, n = int(win * FS), int(pre * FS)
    nt = x.shape[-1]
    # peak-to-peak in the window starting at t
    hi = maximum_filter1d(x, w, axis=-1, origin=-(w // 2))
    lo = minimum_filter1d(x, w, axis=-1, origin=-(w // 2))
    p2p = hi - lo
    c = np.cumsum(np.pad(x ** 2, ((0, 0), (1, 0))), axis=-1)
    rms = np.full(x.shape, np.nan)
    rms[:, n:] = np.sqrt((c[:, n:nt] - c[:, :nt - n]) / n)
    return p2p / np.maximum(rms, 1e-30)


def onset_snr_db(y):
    s = continuous_snr(y)[:, P - FS // 2:P + 3 * FS]
    return float(np.median(20 * np.log10(np.nanmax(s, axis=1))))


def methods():
    from . import jdas, unet2d
    from .das import geometry
    from .methods import ROOT, SeisDAE, bandpass, fk

    dx = geometry()[2]
    ft = ROOT / "models/variants/finetune_s2026"
    m = {"Raw data": lambda x: x,
         "Bandpass 1-10 Hz": lambda x: bandpass(x, (1, 10)),
         "f-k 1000 m/s": lambda x: fk(x, dx, 1000),
         "seisDAE released": SeisDAE(ROOT / "models/gr_mixed_stft.h5", ROOT / "models/gr_mixed_stft.config"),
         "seisDAE fine-tuned s2026": SeisDAE(ft.with_suffix(".h5"), ft.with_suffix(".config")),
         "jDAS fine-tuned": jdas.JDAS(jdas.RUNS / "finetune_s2026.h5")}
    for h5 in sorted(unet2d.RUNS.glob("*.h5")):
        m[f"U-Net {h5.stem}"] = unet2d.UNet2D(h5)
    return m


def score():
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    m = manifest()
    overlap = m["overlap"].fillna(0) if "overlap" in m else 0
    ev = m[(m.kind != "noise") & (m.split == "test") & (overlap == 0)]
    nz = m[(m.kind == "noise") & (m.split == "test")]
    nets = methods()
    rows = []
    for label, group in (("earthquake", ev), ("noise", nz)):
        for k, r in enumerate(group.itertuples()):
            x = load_gather(r.id)
            for name, f in nets.items():
                rows.append(dict(window=r.id, label=label, method=name, magnitude=getattr(r, "magnitude", np.nan),
                                 onset_snr_db=onset_snr_db(f(x))))
            print(f"onset SNR {label} {k + 1}/{len(group)}", end="\r")
        print()
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "onset_snr_rows.csv", index=False)
    return d


def summarise(d):
    from .detection_stats import auc

    rng = np.random.default_rng(CFG["seed"])
    summary, paired = [], []
    eqw = d[d.label == "earthquake"].pivot_table(index="window", columns="method", values="onset_snr_db")
    nzw = d[d.label == "noise"].pivot_table(index="window", columns="method", values="onset_snr_db")
    for name in eqw.columns:
        e, n = eqw[name].to_numpy(), nzw[name].to_numpy()
        me, lo, hi = bootstrap(e, E["bootstrap_draws"], CFG["seed"])
        mn = float(np.mean(n))
        a = auc(e, n)
        boots = [auc(rng.choice(e, len(e)), rng.choice(n, len(n))) for _ in range(2000)]
        summary.append(dict(method=name, earthquake_db=me, earthquake_lo=lo, earthquake_hi=hi, noise_db=mn,
                            difference_db=me - mn, auc=a, auc_lo=np.percentile(boots, 2.5),
                            auc_hi=np.percentile(boots, 97.5)))
        if name != "Bandpass 1-10 Hz":
            diff = (eqw[name] - eqw["Bandpass 1-10 Hz"]).to_numpy()
            dm, dlo, dhi = bootstrap(diff, E["bootstrap_draws"], CFG["seed"])
            paired.append(dict(method=name, minus="Bandpass 1-10 Hz", diff_db=dm, lo=dlo, hi=dhi,
                               higher=int((diff > 0).sum()), events=len(diff)))
    s, p = pd.DataFrame(summary), pd.DataFrame(paired)
    s.to_csv(OUT / "onset_snr_summary.csv", index=False)
    p.to_csv(OUT / "onset_snr_paired.csv", index=False)
    print(f"\nOnset SNR (Steinberg et al., 2025 definition), test days: {len(eqw)} earthquakes, {len(nzw)} noise windows")
    print("Median over channels; mean over windows. Brackets = 95% interval.\n")
    print(f"{'Method':26}{'earthquakes (dB)':>24}{'noise (dB)':>12}{'difference':>12}{'AUC':>20}")
    for r in s.itertuples():
        print(f"{r.method:26}{r.earthquake_db:8.2f} [{r.earthquake_lo:5.2f},{r.earthquake_hi:5.2f}]"
              f"{r.noise_db:12.2f}{r.difference_db:12.2f}{r.auc:9.3f} [{r.auc_lo:.3f},{r.auc_hi:.3f}]")
    print("\nEarthquake onset SNR minus bandpass, paired per event ('yes' = interval excludes zero):")
    for r in p.itertuples():
        sig = "yes" if r.lo > 0 or r.hi < 0 else "no"
        print(f"  {r.method:26}{r.diff_db:+7.2f} [{r.lo:+.2f},{r.hi:+.2f}] {sig:3}  higher in {r.higher}/{r.events}")
    print("\nFiles: results/onset_snr_rows.csv, onset_snr_summary.csv, onset_snr_paired.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats-only", action="store_true")
    a = p.parse_args()
    d = pd.read_csv(OUT / "onset_snr_rows.csv") if a.stats_only else score()
    summarise(d)


if __name__ == "__main__":
    main()
