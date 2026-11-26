"""Denoise your own file with every method of this study.

  python -m src.denoise_file FILE [options]

  python -m src.denoise_file record.mseed                     seismometer data (any ObsPy format)
  python -m src.denoise_file gather.npy --fs 250              DAS gather, array (channels, time)
  python -m src.denoise_file gather.h5 --dataset data --fs 250 --dx 8
  python -m src.denoise_file record.mseed --start 120 --seconds 60

Formats
  .npy / .npz       2-D array (channels, time) or 1-D trace; give --fs (npz: first array, or --dataset)
  .h5 / .hdf5       give --dataset (and --transpose if it is stored as (time, channels)) and --fs
  anything else     read with ObsPy (miniSEED, SAC, SEG-Y, ...); sampling rate from the headers;
                    several traces become channels (cut to their common time span)

What it does
  1. detrend, 0.5 Hz high-pass, resample to 100 Hz (the rate of every model here)
  2. runs, on 60 s windows with cross-faded overlaps:
       bandpass 1-10 Hz                       any data
       seisDAE released                       any data (made for seismometers)
       seisDAE DAS-trained                    any data (trained on Ridgecrest DAS)
       2-D U-Net (ours)                       only with >= 8 channels (trained on Ridgecrest DAS)
       f-k 1000 m/s                           only with >= 8 channels and --dx
  3. saves <name>_denoised.npz (all outputs, 100 Hz) and <name>_denoised.png (before / after),
     and prints the RMS of each output relative to the input.

Read the result with care: the DAS models were trained on one 760 m urban fiber
section (Ridgecrest, 7.95 m channel spacing). On another array they may not work
without retraining (see the README, Limitations). There is no clean reference
here, so no quality score is printed - only what each method removed.
"""

import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt

from .methods import FS, ROOT, bandpass, fk

WIN = 6001


def read(path, fs, dataset, transpose, start, seconds):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".npy", ".npz"):
        if suffix == ".npy":
            x = np.load(path)
        else:
            z = np.load(path)
            x = z[dataset] if dataset else z[z.files[0]]
        if fs is None:
            raise SystemExit("Give the sampling rate: --fs")
    elif suffix in (".h5", ".hdf5"):
        import h5py

        if not dataset:
            with h5py.File(path, "r") as f:
                names = []
                f.visititems(lambda n, o: names.append(f"{n} {o.shape}") if hasattr(o, "shape") else None)
            raise SystemExit("Give --dataset, one of:\n  " + "\n  ".join(names))
        with h5py.File(path, "r") as f:
            x = np.asarray(f[dataset])
        if fs is None:
            raise SystemExit("Give the sampling rate: --fs")
    else:
        from obspy import read as obspy_read

        st = obspy_read(str(path))
        st.merge(fill_value=0)
        t0 = max(tr.stats.starttime for tr in st)
        t1 = min(tr.stats.endtime for tr in st)
        st.trim(t0, t1)
        rates = {tr.stats.sampling_rate for tr in st}
        if len(rates) > 1:
            raise SystemExit(f"Traces have different sampling rates {rates}; select one channel type first")
        fs = rates.pop()
        n = min(len(tr.data) for tr in st)
        x = np.stack([tr.data[:n] for tr in st]).astype(np.float64)
        print(f"read {len(st)} trace(s): {', '.join(tr.id for tr in st[:6])}{' ...' if len(st) > 6 else ''}")
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if transpose:
        x = x.T
    a = int(round((start or 0) * fs))
    b = x.shape[1] if seconds is None else a + int(round(seconds * fs))
    x = x[:, a:b]
    if not np.isfinite(x).all():
        x = np.nan_to_num(x)
        print("warning: non-finite samples set to 0")
    return x, float(fs)


def prepare(x, fs):
    x = detrend(x, axis=-1)
    sos = butter(4, 0.5, fs=fs, btype="highpass", output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    if fs != FS:
        r = Fraction(FS / fs).limit_denominator(1000)
        x = resample_poly(x, r.numerator, r.denominator, axis=-1)
    return x


def windowed(f, x, n=WIN):
    """Apply f (which needs exactly n samples) along time, with cross-faded overlaps."""
    nt = x.shape[1]
    if nt < n:
        pad = n - nt
        return windowed(f, np.pad(x, ((0, 0), (0, pad)), mode="reflect" if pad < nt else "constant"), n)[:, :nt]
    step = n // 2
    starts = list(range(0, nt - n + 1, step))
    if starts[-1] != nt - n:
        starts.append(nt - n)
    out, weight = np.zeros_like(x), np.zeros(nt)
    for k, s in enumerate(starts):
        w = np.ones(n)
        if k > 0:
            lap = starts[k - 1] + n - s
            w[:lap] = np.linspace(0, 1, lap + 2)[1:-1]
        if k < len(starts) - 1:
            lap = s + n - starts[k + 1]
            w[n - lap:] = np.linspace(1, 0, lap + 2)[1:-1]
        out[:, s:s + n] += w * f(x[:, s:s + n])
        weight[s:s + n] += w
    return out / weight


def methods(nch, dx):
    from .methods import SeisDAE
    from . import unet2d

    m = {"Bandpass 1-10 Hz": lambda x: bandpass(x, (1, 10))}
    m["seisDAE released"] = SeisDAE(ROOT / "models/gr_mixed_stft.h5", ROOT / "models/gr_mixed_stft.config")
    m["seisDAE DAS-trained"] = SeisDAE(ROOT / "models/das_seisdae.h5", ROOT / "models/das_seisdae.config")
    if nch >= 8:
        h5 = unet2d.RUNS / "s2026.h5"
        if h5.exists():
            m["2-D U-Net (ours)"] = unet2d.UNet2D(h5)
        else:
            print("2-D U-Net skipped: models/unet2d/s2026.h5 not found (python -m src.unet2d train)")
        if dx:
            m["f-k 1000 m/s"] = lambda x: fk(x, dx, 1000)
    return m


def figure(x, outs, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("Input (prepared)", x)] + list(outs.items())
    t = np.arange(x.shape[1]) / FS
    if x.shape[0] >= 8:
        cols = min(3, len(panels))
        rows = int(np.ceil(len(panels) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False, sharex=True, sharey=True)
        v = np.percentile(np.abs(x), 99)
        for ax, (name, g) in zip(axes.flat, panels):
            ax.imshow(g, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, extent=[t[0], t[-1], g.shape[0], 0],
                      interpolation="none")
            ax.set_title(name, fontsize=9)
        for ax in axes.flat[len(panels):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("Channel")
        for ax in axes[-1]:
            ax.set_xlabel("Time (s)")
    else:
        fig, axes = plt.subplots(len(panels), 1, figsize=(9, 1.6 * len(panels)), sharex=True, squeeze=False)
        for ax, (name, g) in zip(axes[:, 0], panels):
            ax.plot(t, g[0], color="black" if name.startswith("Input") else "#D55E00", lw=0.6)
            ax.set_title(name + ("  (first channel)" if x.shape[0] > 1 else ""), fontsize=9, loc="left")
        axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(title + "  (same colour or amplitude scale in all panels)" if x.shape[0] >= 8 else title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--fs", type=float, help="sampling rate in Hz (needed for .npy/.npz/.h5)")
    p.add_argument("--dataset", help="array name inside .npz or .h5")
    p.add_argument("--transpose", action="store_true", help="array is stored as (time, channels)")
    p.add_argument("--dx", type=float, help="channel spacing in m (DAS only; enables the f-k filter)")
    p.add_argument("--start", type=float, help="start time in s from the beginning of the file")
    p.add_argument("--seconds", type=float, help="length in s to process (default: all)")
    p.add_argument("--out", help="output folder (default: next to the input file)")
    a = p.parse_args()

    import os

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    raw, fs = read(a.file, a.fs, a.dataset, a.transpose, a.start, a.seconds)
    print(f"input: {raw.shape[0]} channel(s), {raw.shape[1]} samples at {fs:g} Hz ({raw.shape[1] / fs:.1f} s)")
    x = prepare(raw, fs)
    nets = methods(x.shape[0], a.dx)
    outs = {}
    for name, f in nets.items():
        if name.startswith(("Bandpass", "f-k", "2-D U-Net")):
            outs[name] = f(x)
        else:
            outs[name] = windowed(f, x)
    src = Path(a.file)
    folder = Path(a.out) if a.out else src.parent
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{src.stem}_denoised"
    np.savez_compressed(stem.with_suffix(".npz"), input=x, fs=FS, **{k.replace(" ", "_"): v for k, v in outs.items()})
    figure(x, outs, stem.with_suffix(".png"), src.name)
    rms = np.sqrt(np.mean(x ** 2))
    print("\nRMS of each output relative to the input (1.0 = nothing removed):")
    for name, y in outs.items():
        print(f"  {name:22} {np.sqrt(np.mean(y ** 2)) / rms:.2f}")
    print(f"\nsaved {stem.with_suffix('.npz')} and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
