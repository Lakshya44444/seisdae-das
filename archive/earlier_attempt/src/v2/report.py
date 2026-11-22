"""Generate human-readable results directly from event tables."""

from pathlib import Path
import subprocess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..data import ROOT
from .stats import OUT, FIG

REPORT = ROOT / "reports_v2"


def table(frame, digits=3):
    def fmt(x):
        return (
            f"{(0.0 if abs(x)<.5*10**(-digits) else x):.{digits}f}"
            if isinstance(x, (float, np.floating))
            else str(x)
        )

    rows = [
        list(frame.columns),
        ["---"] * len(frame.columns),
        *frame.itertuples(index=False, name=None),
    ]
    return "\n".join("| " + " | ".join(map(fmt, row)) + " |" for row in rows)


def chart(frame, series, x, y, name, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7.4, 3.6), layout="constrained")
    for label, g in frame.groupby(series, sort=False):
        ax.plot(g[x], g[y], "o-", ms=4, label=label)
        if y + "_lo" in g:
            ax.fill_between(g[x], g[y + "_lo"], g[y + "_hi"], alpha=0.09)
    ax.axhline(0, color="#8899aa", lw=0.7)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(fontsize=8, ncol=3, frameon=False)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def build():
    REPORT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    b = pd.read_csv(OUT / "baseline_summary.csv")
    sub = pd.read_csv(OUT / "strata_paired.csv")
    contrast = pd.read_csv(OUT / "magnitude_contrast.csv").iloc[0]
    s = b[b.level == 0].set_index("method")
    n = int(s.events.iloc[0])
    chart(
        b,
        "method",
        "level",
        "si_gain_db",
        "shape.png",
        "Input SNR (dB)",
        "SI-SDR gain (dB)",
    )
    chart(
        sub[sub.stratifier == "magnitude"],
        "bin",
        "level",
        "delta_db",
        "magnitude.png",
        "Input SNR (dB)",
        "Head minus zero-shot gain (dB)",
    )
    magnitude = sub[sub.level == 0][
        [
            "bin",
            "events",
            "longvalley",
            "ridgecrest",
            "delta_db",
            "delta_db_lo",
            "delta_db_hi",
        ]
    ].copy()
    magnitude.columns = ["Group", "Events", "LV", "RC", "Gain Δ", "CI low", "CI high"]
    core = s[
        [
            "gain_db",
            "si_gain_db",
            "cc",
            "peak_ratio",
            "polarity_agreement",
            "pick_recovery",
            "pick_mae_ms",
        ]
    ].reset_index()
    core.columns = [
        "Method",
        "Gain",
        "SI gain",
        "CC",
        "Peak",
        "Sign",
        "Recovery",
        "MAE (ms)",
    ]
    ref = (ROOT / "archive/v1/report.md").read_text().split("## References")[-1].strip()
    ref = ref.replace(
        "See upstream bibliography at https://github.com/JanisHe/seisDAE",
        "CTBT Science and Technology Conference. https://conferences.ctbto.org/event/7/contributions/1145/",
    )
    ref += "\n- Li et al. (2021). Rapid Response to the 2019 Ridgecrest Earthquake With Distributed Acoustic Sensing. https://doi.org/10.1029/2021AV000395"
    text = f"""# Can seisDAE recover short DAS earthquake arrivals?

## Abstract

On {n} held-out earthquakes, the fixed output-head fit changed mean SNR gain at 0 dB input from {s.loc['Zero-shot','gain_db']:.3f} to {s.loc['Fine-tuned','gain_db']:.3f} dB. Its scale-invariant gain was {s.loc['Fine-tuned','si_gain_db']:.3f} dB. A training-fitted scalar achieved {s.loc['Scalar','gain_db']:.3f} dB ordinary gain and {s.loc['Scalar','si_gain_db']:.3f} dB scale-invariant gain. The magnitude analysis found a smaller absolute benefit for the smallest catalog events. Updating the last decoder block also improved on the released model, but the original head delivered the larger improvement.

## Research question and data

Does a denoising mask trained on seismometers preserve a DAS P arrival, and does adaptation preferentially improve smaller earthquakes? DAS records axial deformation along a fiber. The reference is a relatively clear field arrival that still contains noise. The Caltech P-phase product is already common-mode suppressed and filtered to 1–10 Hz. Each source trace spans 200 samples at 100 Hz. Catalog magnitudes use the source catalog’s magnitude types.

The original split reserves the 18 largest earthquakes on each array for development and pilot work. Per array, 12 went to training, three to validation and three to an excluded pilot. The remaining test events comprise {int(s.longvalley.iloc[0])} Long Valley and {int(s.ridgecrest.iloc[0])} Ridgecrest earthquakes. Every channel of an earthquake stays in the same split. This is transfer to other earthquakes on previously seen fibers.

## Methods

Among eligible blocks of 24 consecutive physical channels, the original protocol selects the block with highest median source SNR and shift spread at most eight samples. Per-channel shifts align the windows. Zero-based samples 86–155 supply the reference, and 9–78 supply early background. One noise scale per gather sets the input SNR. The neural input centers the demeaned clip inside 6,001 samples. Only the central 70 observed samples are scored. This offers no measured minute-long earthquake context.

The seven comparators are the mixture itself, one fitted scalar, a 1–10 Hz filter, db8 wavelet shrinkage, a symmetric f–k velocity fan, released GR seisDAE, and the same network with a validation-selected output head. The original head learns 16 mask weights using soft magnitude-ratio targets. The pick is an automatic STA/LTA threshold crossing. Timing error conditions on both traces producing a pick.

SNR gain compares squared residual energy before and after filtering. SI-SDR projects each demeaned estimate onto its reference and compares projected energy to residual energy. We difference this score against the noisy input per channel. Undefined zero-energy channels are excluded and squared correlations are clipped away from zero and one at 1e−12. CC retains the sign of correlation. Peak is the estimated-to-reference local peak amplitude ratio. Sign compares the first excursion exceeding 20% of each local peak, after subtracting its five-sample pre-onset mean, over the 15 samples starting at the nominal arrival. This is a waveform proxy, without analyst polarity labels.

Each earthquake contributes one mean waveform score. Pick recovery, sign agreement and timing pool their respective counts. Percentile intervals resample whole events within each array, preserving the observed site counts over 5,000 draws. All metrics and intervals at every input level are in baseline_summary.csv.

## Waveform fidelity at 0 dB input

{table(core)}

![Scale-invariant gain across all {n} test earthquakes. Shading gives pointwise 95% event-bootstrap intervals.](../figures_v2/shape.png)

At 0 dB the highest ordinary gain was from {s.gain_db.idxmax()} and the highest SI gain from {s.si_gain_db.idxmax()}. Multiplying a waveform by a positive constant preserves its shape and pick times. Peak ratios above unity indicate that the estimated arrival window still contains excess peak amplitude relative to the field reference.

## Does fine-tuning help more for smaller events?

{table(magnitude)}

The table gives head-minus-zero-shot SNR gain at 0 dB input in magnitude and median source-SNR strata. The smallest-minus-largest contrast is {contrast.small_minus_large_db:.3f} dB (95% CI {contrast.lo:.3f} to {contrast.hi:.3f}). Both extreme magnitude bins contain only Long Valley earthquakes. Smaller events benefit in this benchmark, but the absolute gain increment is larger in the highest magnitude group. Source SNR and magnitude can covary. These descriptive differences do not isolate a causal magnitude effect. strata.csv and decoder_strata.csv give every method, level and group with intervals for gain, CC and recovery.

![Paired gain difference by magnitude across {n} earthquakes. Group sizes appear in the table. Shading gives pointwise 95% intervals.](../figures_v2/magnitude.png)
"""
    from .report_additions import experiments

    text += "\n" + experiments(table)
    text += (
        "\n## Discussion and limitations\n\nSelection favors high-SNR apertures; ground truth, manual picks and independent event-family labels are unavailable. Related earthquakes can make event-bootstrap intervals optimistic. The fibers, passband, clip length and local noise restrict external validity. Positive SNR gain alone cannot establish better detection, preserved amplitude or suitability for continuous monitoring. Order statistics and subgroup comparisons are exploratory.\n\n## References\n\nCourse-list entry: Steinberg et al. (2023), *Neural Network Based Denoising of Regional Seismic Waveforms and Impact on Analysis of North Korean Nuclear Tests*. The journal record below supplies the published title and dates.\n\n"
        + ref
        + "\n"
    )
    (REPORT / "report.md").write_text(text)
    from .release import update

    update(table)
    from .slides import build_slides

    build_slides()
    subprocess.run(
        ["pandoc", "report.md", "--resource-path=.:..", "-o", "report.docx"],
        cwd=REPORT,
        check=True,
    )
    subprocess.run(
        [
            "soffice",
            "-env:UserInstallation=file:///tmp/seisdae_report_office",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(REPORT),
            str(REPORT / "report.docx"),
        ],
        check=True,
    )


if __name__ == "__main__":
    build()
