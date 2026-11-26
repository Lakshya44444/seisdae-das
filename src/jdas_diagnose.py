"""Why does jDAS do badly on our windows? Three checks.

  python -m src.jdas_diagnose                  checks 1 and 2 (data on disk, about 5 min)
  python -m src.jdas_diagnose --raw 6          also check 3: downloads 6 test events again (~1 min)
  python -m src.jdas_diagnose --sample PATH    also check 0: jDAS's own example data (examples/sample_data.npy)

0. jDAS's own example data. If our wrapper is right, the released model should
   change those data little in 1-10 Hz (they were recorded in the same experiment
   as its training data). Prints the correlation of input and output.
1. Signal without noise. Each method is applied to the clean reference of the
   19 test earthquakes (nothing added). A good denoiser keeps the earthquake:
   SI-SDR of output vs reference should be high. jDAS keeps only what can be
   predicted from neighbouring channels, so this measures how much of our
   reference is coherent from channel to channel.
2. Channel-to-channel coherence (mean zero-lag correlation of neighbours) in the
   earthquake window, for the references and for noise windows.
3. Effect of our common-mode removal. Preprocessing subtracts, at every sample,
   the median over the 96 channels (docs/data.md). On 760 m of fiber a fast
   earthquake arrival is almost the same on all channels, so the median may hold
   most of it. Check 3 downloads a few test events again, preprocesses them with
   and without that step, and reports the share of event-window energy removed
   and the coherence in both cases.
Writes results/jdas_diagnose.csv and results/figures/jdas_diagnose.png.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt

from .build import load_gather, manifest
from .das import CFG, block_rows, geometry, read_raw
from .evaluate import FIG, OUT, event_window, good_channels, original
from .methods import ROOT, SeisDAE, bandpass, coherence, fk, si_sdr


def sample_check(path):
    from . import jdas

    data = np.load(path)
    net = jdas.JDAS(jdas.PRETRAINED)
    y = jdas.chunked(net.predict, data / data.std()) * data.std()
    sos = butter(4, (1, 10), fs=50, btype="bandpass", output="sos")
    xa, ya = sosfiltfilt(sos, data, axis=1), sosfiltfilt(sos, y, axis=1)
    cc = np.mean([np.corrcoef(xa[i], ya[i])[0, 1] for i in range(len(xa))])
    print(f"0. jDAS example data {data.shape}: mean correlation input/output (1-10 Hz) {cc:.2f}, "
          f"output/input energy {np.sum(ya ** 2) / np.sum(xa ** 2):.2f}")
    return dict(check="0 example data", value=cc)


def clean_signal():
    from . import jdas

    m = original(manifest())
    ev = m[(m.kind == "strong") & (m.split == "test")]
    nz = m[(m.kind == "noise") & (m.split == "test")].head(19)
    dx = geometry()[2]
    nets = {"seisDAE DAS-trained": SeisDAE(ROOT / "models/das_seisdae.h5", ROOT / "models/das_seisdae.config"),
            "jDAS fine-tuned": jdas.JDAS(jdas.RUNS / "finetune_s2026.h5")}
    methods = {"Bandpass 1-10 Hz": lambda x: bandpass(x, (1, 10)), "f-k 1000 m/s": lambda x: fk(x, dx, 1000)}
    methods.update(nets)
    rows = []
    for k, r in enumerate(ev.itertuples()):
        s, win, ok = load_gather(r.id), event_window(r), good_channels(r)
        row = dict(event=r.id, coherence=coherence(s[ok], win))
        for name, f in methods.items():
            y = f(s)
            row[name] = float(np.nanmean(si_sdr(s[ok][:, win], y[ok][:, win])))
        rows.append(row)
        print(f"1. clean signal {k + 1}/{len(ev)}", end="\r")
    print()
    d = pd.DataFrame(rows)
    full = slice(None)
    noise_coh = [coherence(load_gather(i), full) for i in nz.id]
    print("\n1. Clean earthquake through each method (no noise added). SI-SDR of output vs reference,")
    print("   earthquake window, scored channels; mean over the 19 test events (higher = signal kept):")
    for name in methods:
        print(f"   {name:22} {d[name].mean():+6.1f} dB   (range {d[name].min():+.1f} to {d[name].max():+.1f})")
    print(f"\n2. Neighbour coherence: earthquake windows {d.coherence.mean():.2f}, "
          f"noise windows {np.mean(noise_coh):.2f}   (1 = identical neighbours, 0 = unrelated)")
    return d, ev, nets["jDAS fine-tuned"]


def without_cmr(raw):
    """das.preprocess without the common-mode step."""
    w = CFG["window"]
    x = detrend(raw, axis=-1)
    sos = butter(4, w["highpass_hz"], fs=250, btype="highpass", output="sos")
    return resample_poly(sosfiltfilt(sos, x, axis=-1), 2, 5, axis=-1)


def raw_check(ev, n, net):
    w = CFG["window"]
    pad, rows = w["pad_s"], block_rows()
    a = int(round(pad * w["fs"]))
    seconds = (w["samples"] - 1) / w["fs"] + 2 * pad + 0.04
    out, keep = [], None
    for k, r in enumerate(ev.head(n).itertuples()):
        raw = read_raw(pd.Timestamp(r.start) - pd.Timedelta(seconds=pad), seconds, rows)
        plain = without_cmr(raw)[:, a:a + w["samples"]]
        cmr = plain - np.median(plain, axis=0, keepdims=True)
        win = event_window(r)
        removed = 1 - np.sum(cmr[:, win] ** 2) / np.sum(plain[:, win] ** 2)
        j_plain, j_cmr = net(plain), net(cmr)
        row = dict(event=r.id, energy_removed_by_cmr=removed,
                   coherence_without_cmr=coherence(plain, win), coherence_with_cmr=coherence(cmr, win),
                   jdas_kept_without_cmr=float(np.nanmean(si_sdr(plain[:, win], j_plain[:, win]))),
                   jdas_kept_with_cmr=float(np.nanmean(si_sdr(cmr[:, win], j_cmr[:, win]))))
        out.append(row)
        if keep is None:
            keep = (r, plain, cmr, j_plain, j_cmr)
        print(f"3. raw {k + 1}/{n}", end="\r")
    print()
    d = pd.DataFrame(out)
    print("\n3. Common-mode removal, test events downloaded again (mean over events):")
    print(f"   share of earthquake-window energy removed by it   {100 * d.energy_removed_by_cmr.mean():5.1f} %")
    print(f"   neighbour coherence   without {d.coherence_without_cmr.mean():.2f}   with {d.coherence_with_cmr.mean():.2f}")
    print(f"   jDAS keeps the signal (SI-SDR, no noise added)   without {d.jdas_kept_without_cmr.mean():+.1f} dB"
          f"   with {d.jdas_kept_with_cmr.mean():+.1f} dB")
    return d, keep


def figure(keep):
    import matplotlib.pyplot as plt

    r, plain, cmr, j_plain, j_cmr = keep
    panels = [("Without common-mode removal", plain), ("jDAS of that", j_plain),
              ("With common-mode removal (our data)", cmr), ("jDAS of that", j_cmr)]
    t = np.arange(plain.shape[1]) / CFG["window"]["fs"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=True, constrained_layout=True)
    for ax, (title, g) in zip(axes, panels):
        v = np.percentile(np.abs(plain), 99)
        ax.imshow(g, aspect="auto", cmap="seismic", vmin=-v, vmax=v, extent=[t[0], t[-1], len(g), 0])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_xlim(5, 30)
    axes[0].set_ylabel("Channel")
    fig.suptitle(f"Test earthquake {r.id}: same colour scale in all panels")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "jdas_diagnose.png", dpi=130)
    print("figure: results/figures/jdas_diagnose.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=int, default=0, help="number of test events to download again for check 3")
    p.add_argument("--sample", help="path to jDAS examples/sample_data.npy for check 0")
    a = p.parse_args()
    if a.sample:
        sample_check(a.sample)
    d, ev, net = clean_signal()
    d.to_csv(OUT / "jdas_diagnose.csv", index=False)
    if a.raw:
        raw, keep = raw_check(ev, a.raw, net)
        raw.to_csv(OUT / "jdas_diagnose_cmr.csv", index=False)
        figure(keep)


if __name__ == "__main__":
    main()
