"""Regenerate the figures from saved test outputs without running TensorFlow."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from .data import FS, ROOT

COLORS = {
    "Scalar": "#68753c",
    "Noisy": "#777777",
    "Bandpass": "#ae671a",
    "Wavelet": "#7a6898",
    "f-k": "#276b96",
    "Zero-shot": "#9e4e52",
    "Fine-tuned": "#007b68",
    "reference": "#202728",
}


def save(fig, name, out):
    fig.savefig(out / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main(results=ROOT / "results", out=ROOT / "figures", models=ROOT / "models"):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
    )
    d = np.load(results / "example.npz")
    eid = int(d["event_id"])
    fig, ax = plt.subplots(figsize=(7.8, 3.2), layout="constrained")
    limit = np.quantile(abs(d["raw"]), 0.99)
    im = ax.imshow(
        d["raw"],
        extent=(0, 2, len(d["raw"]) - 0.5, -0.5),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax.set(
        title=f"Released P-phase clip · event {eid}",
        xlabel="Time in clip (s)",
        ylabel="Channel in aperture",
    )
    fig.colorbar(im, ax=ax, label="Recorded amplitude")
    save(fig, "02_raw_gather.png", out)

    names = [
        "reference",
        "Noisy",
        "Scalar",
        "Bandpass",
        "Wavelet",
        "f-k",
        "Zero-shot",
        "Fine-tuned",
    ]
    fig, axes = plt.subplots(
        2, 4, figsize=(12, 5), sharex=True, sharey=True, layout="constrained"
    )
    limit = np.quantile(abs(d["Noisy"]), 0.99)
    for ax, name in zip(axes.flat, names):
        ax.imshow(
            d[name],
            extent=(0, 0.7, 23.5, -0.5),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_title(name.capitalize() if name == "reference" else name)
    fig.supxlabel("Time in analysis window (s)")
    fig.supylabel("Channel in aperture")
    fig.suptitle(f"Same event and amplitude scale · {eid} · 0 dB input")
    save(fig, "03_methods.png", out)

    fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
    t = np.arange(70) / FS
    for name in ("Noisy", "reference", "Zero-shot", "Fine-tuned"):
        ax.plot(t, d[name][12], color=COLORS[name], lw=1.5, label=name)
    ax.set(
        xlabel="Time in analysis window (s)",
        ylabel="Amplitude in archive units",
        title=f"Event {eid} · channel offset 12 · 0 dB input",
    )
    ax.legend(ncol=4, fontsize=9)
    save(fig, "04_trace.png", out)

    df = pd.read_csv(results / "summary.csv")
    fig, ax = plt.subplots(figsize=(7.8, 4.2), layout="constrained")
    ax.axhline(0, ls="--", color=".5", lw=0.8)
    for method, group in df[df.method != "Noisy"].groupby("method"):
        ax.plot(group.level, group.gain_db, "o-", label=method, color=COLORS[method])
    ax.legend(ncol=3, fontsize=9)
    ax.set(
        xticks=[-5, 0, 5, 10],
        xlabel="Input SNR (dB)",
        ylabel="Mean SNR gain (dB)",
        title=f"{int(df.events.iloc[0])} held-out earthquakes · equal weight per event",
    )
    save(fig, "05_snr.png", out)
    events = pd.read_csv(results / "per_event.csv")
    matched = events[events.level == 0].pivot(
        index=["array", "event_id"], columns="method", values="gain_db"
    )
    delta = (matched["Fine-tuned"] - matched["Zero-shot"]).rename("delta")
    fig, ax = plt.subplots(figsize=(8, 2.9), layout="constrained")
    for row, (site, group) in enumerate(delta.groupby(level="array")):
        values = np.sort(group.to_numpy())
        # Rank-based offsets make overlapping markers easier to see without randomness.
        offsets = 0.12 * np.sin(np.arange(len(values)) * 2.4)
        ax.scatter(
            values, row + offsets, s=17, alpha=0.57, color=["#276b96", "#007b68"][row]
        )
        ax.plot(
            values.mean(),
            row,
            "D",
            color="#182c34",
            markersize=7,
            label="Array mean" if row == 0 else None,
        )
    ax.axvline(0, ls="--", lw=0.9, color=".45")
    ax.set(
        yticks=[0, 1],
        yticklabels=["Long Valley", "Ridgecrest"],
        xlabel="Fine-tuned minus zero-shot SNR gain (dB)",
        title="One point per held-out earthquake at 0 dB input",
    )
    ax.set_ylim(-0.55, 1.55)
    ax.legend(loc="upper right")
    save(fig, "08_paired_gain.png", out)
    loss = pd.read_csv(models / "loss.csv")
    fig, ax = plt.subplots(figsize=(7, 3.3), layout="constrained")
    for key, color in (("training_loss", "#276b96"), ("validation_loss", "#ae671a")):
        ax.plot(loss.epoch, loss[key], label=key.split("_")[0], color=color)
    ax.set(
        xlabel="Epoch", ylabel="Mask cross-entropy", title="Output-layer fine-tuning"
    )
    ax.legend()
    save(fig, "06_training.png", out)
    fig, axes = plt.subplots(
        1, 3, figsize=(9.5, 3.1), sharey=True, layout="constrained"
    )
    panels = []
    for key in ("reference", "Noisy", "Fine-tuned"):
        f, t, p = spectrogram(
            d[key][12], fs=FS, window="hann", nperseg=32, noverlap=28, nfft=64
        )
        panels.append(10 * np.log10(np.maximum(p, 1e-30)))
    vmax = np.max(panels)
    for ax, p, title in zip(axes, panels, ("Reference", "Noisy", "Fine-tuned")):
        im = ax.pcolormesh(
            t, f, p, shading="auto", vmin=vmax - 40, vmax=vmax, cmap="viridis"
        )
        ax.set(title=title, xlabel="Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axes, label="PSD (dB re 1 unit²/Hz)")
    save(fig, "07_spectrogram.png", out)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    main(args.results, args.out, args.models)
