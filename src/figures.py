"""Figures and a numbers-only summary table from results/ (no hand-typed values).

  python -m src.figures
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import FIG, OUT

ORDER = ["Noisy", "Scalar", "Bandpass", "f-k", "seisDAE (released)", "seisDAE (DAS-trained)", "seisDAE (DAS) + f-k"]
FS = 100


def gathers(path, panels, title, out):
    with np.load(path) as z:
        data = {k: z[k] for k in z.files if k != "event"}
        event = str(z["event"])
    panels = [p for p in panels if p in data]
    ref = data.get("Reference", data["Noisy"])
    clip = np.percentile(np.abs(ref), 99)
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 4), sharey=True, layout="constrained")
    for ax, name in zip(np.atleast_1d(axes), panels):
        y = data[name]
        ax.imshow(y, aspect="auto", cmap="seismic", vmin=-clip, vmax=clip,
                  extent=(0, y.shape[1] / FS, y.shape[0], 0), interpolation="none")
        ax.set_title("Recorded" if name == "Noisy" and "Reference" not in data else name, fontsize=9)
        ax.set_xlabel("Time (s)")
    np.atleast_1d(axes)[0].set_ylabel("Channel")
    fig.suptitle(f"{title} — {event}")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def levels():
    s = pd.read_csv(OUT / "synthetic_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, (q, label) in zip(axes, (("si_gain_db", "SI-SDR gain, whole window (dB)"),
                                     ("si_gain_db_event", "SI-SDR gain, earthquake window (dB)"))):
        for method in ORDER[1:]:
            g = s[s.method == method].sort_values("level")
            if g.empty:
                continue
            ax.errorbar(g.level, g[q], yerr=[g[q] - g[q + "_lo"], g[q + "_hi"] - g[q]],
                        marker="o", capsize=3, label=method)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("Input SNR (dB)")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.suptitle("Held-out test days, mean over earthquakes with 95% event bootstrap intervals")
    fig.savefig(FIG / "levels.png", dpi=160)
    plt.close(fig)


def trace():
    with np.load(OUT / "example_synthetic.npz") as z:
        ref, noisy, das = z["Reference"], z["Noisy"], z["seisDAE (DAS-trained)"]
    c = ref.shape[0] // 2
    t = np.arange(ref.shape[1]) / FS
    fig, ax = plt.subplots(3, 1, figsize=(9, 5), sharex=True, sharey=True, layout="constrained")
    for a, y, name in zip(ax, (noisy, das, ref), ("Noisy input (0 dB)", "seisDAE (DAS-trained)", "Recorded reference")):
        a.plot(t, y[c], lw=0.5, color="#16586a")
        a.set_ylabel(name, fontsize=8)
    ax[-1].set_xlabel("Time (s)")
    fig.savefig(FIG / "trace.png", dpi=160)
    plt.close(fig)


def table():
    s = pd.read_csv(OUT / "synthetic_summary.csv")
    run = json.loads((OUT / "run.json").read_text())
    lines = ["# Results (generated from results/*.csv)", "",
             f"Test earthquakes: {run['test_events']}; test noise windows: {run['test_noise_windows']}; "
             f"weak field events: {run['weak_test_events']}; validation earthquakes: {run['validation_events']}.",
             f"Frozen settings (validation days): {run['settings']}", ""]
    for level in sorted(s.level.unique()):
        g = s[s.level == level].set_index("method").reindex(ORDER).dropna(how="all").reset_index()
        lines += [f"## Input SNR {level} dB", "",
                  "| Method | Ordinary gain (dB) | SI gain (dB) [95% CI] | SI gain, event window (dB) | CC |",
                  "|---|---|---|---|---|"]
        for r in g.itertuples():
            lines.append(f"| {r.method} | {r.gain_db:.2f} | {r.si_gain_db:.2f} [{r.si_gain_db_lo:.2f}, "
                         f"{r.si_gain_db_hi:.2f}] | {r.si_gain_db_event:.2f} | {r.cc:.3f} |")
        lines.append("")
    p = pd.read_csv(OUT / "synthetic_paired.csv")
    lines += ["## Paired SI-gain differences", "", "| Comparison | Level | Mean (dB) | 95% CI | Wins / events |",
              "|---|---|---|---|---|"]
    for r in p.itertuples():
        lines.append(f"| {r.comparison} | {r.level} | {r.si_gain_db:.2f} | [{r.lo:.2f}, {r.hi:.2f}] | {r.wins}/{r.events} |")
    if (OUT / "field_summary.csv").exists():
        f = pd.read_csv(OUT / "field_summary.csv").set_index("method").reindex(ORDER).dropna(how="all").reset_index()
        lines += ["", "## Weak recorded earthquakes (no added noise, no clean reference)", "",
                  "| Method | Median channel SNR (dB) | Channels >= 10 dB | Neighbour coherence |", "|---|---|---|---|"]
        for r in f.itertuples():
            lines.append(f"| {r.method} | {r.median_snr_db:.2f} | {r.channels_10db:.2f} | {r.coherence:.3f} |")
    if (OUT / "detection_summary.csv").exists():
        d = pd.read_csv(OUT / "detection_summary.csv").set_index("method").reindex(ORDER).dropna(how="all").reset_index()
        lines += ["", "## Detection test on recorded test-day data (main result)", "",
                  f"Earthquake windows: {int(d.earthquakes.iloc[0])}; noise windows: {int(d.noise_windows.iloc[0])}. "
                  "Thresholds are set on the noise windows of each method.", "",
                  "| Method | Detected at 5% false alarm [95% CI] | Detected at 10% false alarm [95% CI] | "
                  "Median ratio, earthquakes (dB) | Median ratio, noise (dB) |", "|---|---|---|---|---|"]
        for r in d.itertuples():
            lines.append(f"| {r.method} | {r.detected_fa5:.2f} [{r.detected_fa5_lo:.2f}, {r.detected_fa5_hi:.2f}] | "
                         f"{r.detected_fa10:.2f} [{r.detected_fa10_lo:.2f}, {r.detected_fa10_hi:.2f}] | "
                         f"{r.median_earthquake_db:.2f} | {r.median_noise_db:.2f} |")
    lines += ["", f"Median CPU time per trace (ms): {run['ms_per_trace']}"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    FIG.mkdir(exist_ok=True)
    gathers(OUT / "example_synthetic.npz", ["Reference"] + ORDER[0:1] + ORDER[2:],
            "Synthetic test at 0 dB", FIG / "gather_synthetic.png")
    if (OUT / "example_field.npz").exists():
        gathers(OUT / "example_field.npz", ["Noisy", "f-k", "seisDAE (released)", "seisDAE (DAS-trained)",
                                            "seisDAE (DAS) + f-k"], "Weak recorded earthquake", FIG / "gather_field.png")
    levels()
    trace()
    table()


if __name__ == "__main__":
    main()
