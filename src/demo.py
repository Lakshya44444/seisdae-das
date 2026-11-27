"""Demonstration on one real Ridgecrest DAS window: download, denoise, plot.

  python -m src.demo                              test earthquake eq_39290071 (M1.7) and a test noise window
  python -m src.demo --window eq_38595986         any window id from data/ridgecrest/manifest.csv
  python -m src.demo --list                       list the clean test earthquakes to choose from

Needs internet only if the window is not on disk yet: one window (96 channels,
64 s) is read directly from the public SCEDC archive on AWS, about 6 MB, in
well under a minute. No other data, no training.

Methods: bandpass 1-10 Hz, f-k 1000 m/s, seisDAE released, seisDAE DAS-trained
(both in models/), and, if their weights are present, the fine-tuned seisDAE
(models/variants/finetune_s2026.h5) and the 2-D U-Net (models/unet2d/s2026.h5).

Writes to results/demo/<window>/ :
  1_gathers.png          all methods side by side (channel x time), same colour scale
  2_time_frequency.png   waveform and spectrogram on the channel with the highest SNR
  3_fk_spectra.png       frequency-wavenumber spectra (traffic vs earthquake velocities)
  4_continuous_snr.png   continuous SNR around the P arrival (Steinberg et al., 2025)
  numbers.csv            onset SNR and energy removed per method
and prints the numbers. If --noise is given (default: a test noise window), the
same is done for it, so one can see what each method does without an earthquake.
"""

import argparse
import os

import numpy as np
import pandas as pd

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from .build import GATHERS, manifest  # noqa: E402
from .das import block_rows, geometry, read_window  # noqa: E402
from .methods import ROOT, bandpass, fk  # noqa: E402

OUTDIR = ROOT / "results/demo"
DEFAULT_EQ = "eq_39290071"
DEFAULT_NOISE = "noise_20200627172028"


def get_window(name):
    m = manifest().set_index("id")
    if name not in m.index:
        raise SystemExit(f"{name} is not in data/ridgecrest/manifest.csv (try --list)")
    r = m.loc[name]
    path = GATHERS / f"{name}.npz"
    if path.exists():
        with np.load(path) as z:
            return z["data"].astype(np.float64), r
    print(f"downloading {name} ({r.start}) from the SCEDC archive on AWS ...")
    x = read_window(r.start, block_rows()).astype(np.float64)
    GATHERS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=x.astype(np.float32))
    return x, r


def methods():
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    from .methods import SeisDAE

    dx = geometry()[2]
    m = {"Recorded": lambda x: x,
         "Bandpass 1-10 Hz": lambda x: bandpass(x, (1, 10)),
         "f-k 1000 m/s": lambda x: fk(x, dx, 1000),
         "seisDAE released": SeisDAE(ROOT / "models/gr_mixed_stft.h5", ROOT / "models/gr_mixed_stft.config"),
         "seisDAE DAS-trained": SeisDAE(ROOT / "models/das_seisdae.h5", ROOT / "models/das_seisdae.config")}
    ft = ROOT / "models/variants/finetune_s2026"
    if ft.with_suffix(".h5").exists():
        m["seisDAE fine-tuned"] = SeisDAE(ft.with_suffix(".h5"), ft.with_suffix(".config"))
    unet = ROOT / "models/unet2d/s2026.h5"
    if unet.exists():
        from .unet2d import UNet2D

        m["2-D U-Net (ours)"] = UNet2D(unet)
    return m, dx


def plots(name, r, outs, dx, folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    from .build import fast_snr
    from .onset_snr import continuous_snr

    x = outs["Recorded"]
    t = np.arange(x.shape[1]) / 100
    eq = name.startswith("eq_")
    itp = int(r.itp) if eq else 1000
    its = int(r.its) if eq else 1200
    title = (f"{name}: M{r.magnitude:.1f}, {r.dist_km:.0f} km, {str(r.start)[:19]} UTC" if eq
             else f"{name}: noise window, {str(r.start)[:19]} UTC")

    # 1 gathers
    n = len(outs)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.0 * rows), sharex=True, sharey=True, squeeze=False)
    v = np.percentile(np.abs(x), 99)
    for ax, (k, g) in zip(axes.flat, outs.items()):
        ax.imshow(g, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, extent=[0, 60, g.shape[0], 0],
                  interpolation="none")
        if eq:
            ax.axvline(itp / 100, color="#D55E00", lw=0.8, ls="--")
            ax.axvline(its / 100, color="#0072B2", lw=0.8, ls="--")
        ax.set_title(k, fontsize=9)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Channel (7.95 m apart)")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(title + ("; dashed: predicted P (orange) and S (blue)" if eq else "") + "; same colour scale",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / "1_gathers.png", dpi=150)
    plt.close(fig)

    # 2 time-frequency on the channel with the highest SNR (traffic removed for the choice only)
    ch = int(np.argmax(fast_snr(x, itp, its, dx))) if eq else x.shape[0] // 2
    fig, axes = plt.subplots(n, 2, figsize=(10, 1.5 * n + 0.6), sharex=True,
                             gridspec_kw=dict(width_ratios=[1, 1.2]))
    _, _, ref = spectrogram(x[ch], fs=100, nperseg=128, noverlap=120)
    top = 10 * np.log10(ref.max())
    for i, (k, g) in enumerate(outs.items()):
        y = g[ch]
        axes[i, 0].plot(t, y / (np.abs(y).max() + 1e-30), color="black", lw=0.5)
        axes[i, 0].set_ylabel(k, fontsize=7)
        axes[i, 0].set_yticks([])
        if eq:
            for a in axes[i]:
                a.axvline(itp / 100, color="#D55E00", lw=0.8, ls="--")
                a.axvline(its / 100, color="#0072B2", lw=0.8, ls="--")
        ff, tt, S = spectrogram(y, fs=100, nperseg=128, noverlap=120)
        im = axes[i, 1].pcolormesh(tt, ff, 10 * np.log10(S + 1e-30), vmin=top - 60, vmax=top, cmap="magma",
                                   shading="auto")
        axes[i, 1].set_ylim(0, 45)
        axes[i, 1].set_ylabel("Hz", fontsize=7)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.colorbar(im, ax=axes[:, 1], shrink=0.6, label="Power (dB)")
    fig.suptitle(f"{title}; channel {ch}. Left: waveform (each normalised). Right: spectrogram (same dB scale)",
                 fontsize=9)
    fig.savefig(folder / "2_time_frequency.png", dpi=150)
    plt.close(fig)

    # 3 f-k spectra
    show = [k for k in ("Recorded", "Bandpass 1-10 Hz", "seisDAE DAS-trained", "2-D U-Net (ours)") if k in outs]
    fig, axes = plt.subplots(1, len(show), figsize=(3.2 * len(show), 3.2), sharey=True, squeeze=False)
    kax = np.fft.fftshift(np.fft.fftfreq(x.shape[0], dx))
    nt = 2100
    fax = np.fft.fftfreq(nt, 0.01)[:nt // 2]
    spec = lambda g: 20 * np.log10(np.fft.fftshift(np.abs(np.fft.fft2(g[:, 900:900 + nt])), axes=0)[:, :nt // 2] + 1e-12)
    peak = spec(x).max()
    for ax, k in zip(axes[0], show):
        ax.imshow(spec(outs[k]).T, origin="lower", aspect="auto", cmap="magma", vmin=peak - 60, vmax=peak,
                  extent=[kax[0], kax[-1], fax[0], fax[-1]])
        for vel, lab in ((30, "30 m/s"), (1000, "1000 m/s")):
            kk = np.linspace(kax[0], kax[-1], 50)
            ax.plot(kk, np.abs(kk) * vel, color="white", lw=0.7, ls="--")
        ax.set_ylim(0, 30)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("Wavenumber (1/m)")
    axes[0, 0].set_ylabel("Frequency (Hz)")
    fig.suptitle(f"{title}: f-k spectra, 9-30 s. Dashed lines: 30 m/s (vehicles) and 1000 m/s", fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "3_fk_spectra.png", dpi=150)
    plt.close(fig)

    # 4 continuous SNR around P
    sl = slice(itp - 300, itp + 600)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    colors = ["#999999", "#555555", "#009E73", "#B07A00", "#0072B2", "#56B4E9", "#D55E00"]
    for (k, g), c in zip(outs.items(), colors):
        s = continuous_snr(g[ch][None])[0, sl]
        ax.plot(t[sl] - itp / 100, 20 * np.log10(s), color=c, lw=1.1, label=k)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Time after predicted P (s)")
    ax.set_ylabel("Continuous SNR (dB)")
    ax.set_title(f"{title}; channel {ch}. SNR(t) = peak-to-peak in 0.5 s / RMS of the 2 s before", fontsize=8.5)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(folder / "4_continuous_snr.png", dpi=150)
    plt.close(fig)


def run(name, nets, dx):
    from .onset_snr import onset_snr_db

    x, r = get_window(name)
    outs = {k: f(x) for k, f in nets.items()}
    folder = OUTDIR / name
    folder.mkdir(parents=True, exist_ok=True)
    plots(name, r, outs, dx, folder)
    rms = np.sqrt(np.mean(x ** 2))
    rows = [dict(method=k, onset_snr_db=onset_snr_db(y), energy_kept=float(np.sqrt(np.mean(y ** 2)) / rms))
            for k, y in outs.items()]
    d = pd.DataFrame(rows)
    d.to_csv(folder / "numbers.csv", index=False)
    print(f"\n{name}" + (f"  M{r.magnitude:.1f}  {r.dist_km:.0f} km" if name.startswith("eq_") else "  (noise window)"))
    print(f"  {'method':24}{'onset SNR (dB)':>16}{'RMS kept':>10}")
    for row in d.itertuples():
        print(f"  {row.method:24}{row.onset_snr_db:16.1f}{row.energy_kept:10.2f}")
    print(f"  figures: results/demo/{name}/1_gathers.png, 2_time_frequency.png, 3_fk_spectra.png, "
          f"4_continuous_snr.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", default=DEFAULT_EQ)
    p.add_argument("--noise", default=DEFAULT_NOISE, help="also run a noise window ('none' to skip)")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        m = manifest()
        t = m[(m.kind == "strong") & (m.split == "test")].sort_values("magnitude", ascending=False)
        print(t[["id", "magnitude", "dist_km", "start"]].to_string(index=False))
        return
    nets, dx = methods()
    print("methods:", ", ".join(nets))
    run(a.window, nets, dx)
    if a.noise and a.noise != "none":
        run(a.noise, nets, dx)


if __name__ == "__main__":
    main()
