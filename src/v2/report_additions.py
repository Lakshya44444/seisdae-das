"""Tables, plots and interpretations of the prespecified development studies."""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .stats import FIG, OUT
from ..data import ROOT


def read(name):
    return pd.read_csv(OUT / f"{name}.csv")


def at_zero(name):
    d = read(name)
    return d[d.level == 0].copy()


def interval(row, k, digits=3):
    return f"{row[k]:.{digits}f} [{row[k+'_lo']:.{digits}f}, {row[k+'_hi']:.{digits}f}]"


def experiments(table):
    final = pd.concat([at_zero("baseline_summary"), at_zero("decoder_summary")])
    variants = ["Zero-shot", "Fine-tuned", "Decoder"]
    val = at_zero("validation_summary")
    cv = at_zero("cv_summary")
    folds = at_zero("cv_folds")
    sites = at_zero("site_summary")
    sites = sites[sites.method == "Fine-tuned"].set_index("array")
    details = []
    for split, rows in [("OOF", cv), ("Test", final)]:
        for _, r in rows[rows.method.isin(variants)].iterrows():
            details.append(
                [
                    split,
                    r.method,
                    interval(r, "gain_db"),
                    interval(r, "si_gain_db"),
                    interval(r, "cc"),
                    interval(r, "pick_mae_ms", 1),
                ]
            )
    details = pd.DataFrame(
        details,
        columns=["Cohort", "Model", "Gain (dB)", "SI gain (dB)", "CC", "Pick MAE (ms)"],
    )
    paired = at_zero("decoder_paired").set_index("comparator")
    f = folds.pivot(index="method", columns="fold", values="gain_db")
    f["SD"] = f.std(axis=1)
    f.columns = ["Fold 1", "Fold 2", "Fold 3", "SD"]
    sens = at_zero("sensitivity")
    pad = sens[sens.study == "padding"].pivot(
        index="setting", columns="method", values=["gain_db", "si_gain_db"]
    )
    pad = pad.loc[["zero", "reflect", "noise", "tile", "stretch4"]]
    pad.columns = ["Head gain", "Released gain", "Head SI", "Released SI"]
    noise = sens[
        (sens.study == "noise")
        & (sens.method == "Fine-tuned")
        & (sens.setting != "continuous")
    ].copy()
    nr = at_zero("noise_ridgecrest")
    nr = nr[(nr.method == "Fine-tuned") & (nr.setting.isin(["original", "continuous"]))]
    ntable = pd.concat(
        [noise.assign(cohort="Both sites"), nr.assign(cohort="Ridgecrest")]
    )
    ntable = ntable[
        ["cohort", "setting", "events", "gain_db", "si_gain_db", "pick_mae_ms"]
    ]
    ntable.columns = ["Cohort", "Noise", "Events", "Gain (dB)", "SI (dB)", "MAE (ms)"]
    ap = sens[(sens.study == "aperture") & (sens.method == "Fine-tuned")].copy()
    ap["setting"] = ap.setting.astype(int)
    ap = ap.sort_values("setting")[
        ["setting", "gain_db", "si_gain_db", "cc", "pick_mae_ms"]
    ]
    ap.columns = ["Channels", "Gain", "SI gain", "CC", "MAE (ms)"]
    hy = sens[(sens.study == "hybrid")][
        ["method", "gain_db", "si_gain_db", "cc", "pick_mae_ms"]
    ].copy()
    hy.columns = ["Order", "Gain", "SI gain", "CC", "MAE (ms)"]
    size = read("size")
    fig, ax = plt.subplots(figsize=(7.4, 3.5), layout="constrained")
    for level, g in size.groupby("level"):
        stats = g.groupby("count").gain_db.agg(["mean", "min", "max"])
        ax.plot(stats.index, stats["mean"], "o-", label=f"{level} dB")
        ax.fill_between(stats.index, stats["min"], stats["max"], alpha=0.14)
    ax.set(
        xlabel="Training earthquakes",
        ylabel="Validation SNR gain (dB)",
        xticks=sorted(size["count"].unique()),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.savefig(FIG / "size.png", dpi=160)
    plt.close(fig)
    sizes = size[size.level == 0].groupby("count").gain_db.agg(["mean", "min", "max"])
    ranks = read("extreme_profiles")
    ends = ranks[ranks.method == "Fine-tuned"].set_index("side")
    meta = json.loads((ROOT / "data/noise_v2/continuous.json").read_text())
    fits = read("fits")
    epochs = fits[fits.fold == 0].set_index("method").epoch
    block = int(
        sum(
            np.prod(v.shape)
            for key, v in np.load(ROOT / "models/decoder_v2.npz").items()
            if not key.startswith(
                ("batch_normalization_21_2", "batch_normalization_21_3")
            )
        )
    )
    sel = val.set_index("method").gain_db
    return f"""
In the site-specific 0 dB results, head SNR gain was {interval(sites.loc['longvalley'],'gain_db')} dB for Long Valley and {interval(sites.loc['ridgecrest'],'gain_db')} dB for Ridgecrest. The pooled estimate weights earthquakes equally, giving Long Valley greater influence. site_summary.csv contains all methods and levels by site.

## How much of the decoder should adapt?

The larger fit updates the last decoder convolution, batch-normalization scale and offset, and output mask ({block} weights). The encoder, earlier decoder blocks and moving batch statistics stay fixed. Spatial features are cached with a one-frame halo. The implementation checks that cropping yields the original full-network masks at all active time frames. Adam uses a learning rate of 0.00003 and batches of 32 traces. Each training mixture contributes every fourth channel. Epoch zero is eligible; stopping uses validation mask cross-entropy with patience four. The original head uses its original optimizer and epoch budget. The final selected epochs were {epochs['Fine-tuned']} for the head and {epochs['Decoder']} for the decoder. Both reached their configured budgets. The capacity and optimizer budgets differ, so this experiment cannot isolate the effect of parameter count.

The original validation set favored the head on SNR gain at 0 dB ({sel['Fine-tuned']:.3f} vs {sel['Decoder']:.3f} dB). The diagnostic candidate was retained for reporting under the frozen protocol. The test earthquakes were known from V1. They were excluded from fitting, checkpoint selection and all development ablations. A fresh untouched external test population would be a different experiment.

### Model estimates at 0 dB input

{table(details)}

Brackets are 95% within-array event-bootstrap intervals. OOF denotes out-of-fold predictions on 30 development earthquakes. Each of three balanced outer folds holds out ten events. Of the other twenty, four stop training and sixteen fit the weights. Each development event receives one outer-fold prediction per variant and level. The final test uses separate 24-event training and six-event validation sets. At 0 dB the decoder-minus-head gain was {interval(paired.loc['Fine-tuned'],'gain_db')} dB; decoder-minus-released was {interval(paired.loc['Zero-shot'],'gain_db')} dB. More adaptation capacity did not improve the selected head under these budgets.

{table(f.reset_index().rename(columns={"method":"Model"}))}

Fold entries are mean 0 dB gains in dB. SD is the sample standard deviation across three folds. Their training sets overlap; the folds are descriptive replications. Pooled OOF intervals condition on these fitted models and omit uncertainty from an entirely new training draw.

## What changes when the model sees a different context?

{table(pad.reset_index().rename(columns={"setting":"Extension"}))}

All entries are dB on the same six validation events at 0 dB input. The two frozen checkpoints were evaluated at all four mixture levels. Reflect mirrors the clip, tile repeats it, and noise surrounds it with seeded Gaussian values matched to each observed channel’s standard deviation. Stretch4 resamples each clip to four times its length before zero padding, and reverses that resampling after inference. It changes the apparent frequencies. All other extensions score the original center. Reflect and tile increase ordinary gain while hurting scale-invariant shape on this split. The stretch result justifies testing longer measured windows in a future acquisition; it does not supply missing physical observations.

## Independent and displaced backgrounds

{table(ntable)}

The other-event condition cycles donors by event ID within the same validation site. It uses the donor's selected aperture at potentially different physical positions. The other-block condition uses the first disjoint eligible block after the target, wrapping through the event file if needed. Both use the published early-window rule. The other block shares the event and time and can retain correlated noise.

The external test uses {meta['duration_s']} seconds beginning {meta['start_utc']} from a separate SCEDC Ridgecrest recording. We fetched {meta['bytes_fetched']:,} raw sample bytes from trace rows {meta['trace_rows'][0]}–{meta['trace_rows'][1]}. We detrended, differentiated, subtracted the channel median, bandpassed at 1–10 Hz and resampled to 100 Hz. Three fixed disjoint 70-sample snippets were each mixed with one validation earthquake. The manifest records every range and digest. The excerpts are candidate backgrounds without catalog screening. The matched Ridgecrest rows show how much performance depends on a single noise source; there is only one independent continuous recording.

## Training size and aperture width

![Mean gain on six validation earthquakes versus training-set size at each input SNR. Shading spans five seeded fits; it is a range, not a confidence interval.](../figures_v2/size.png)

Each seed draws equal numbers of events per site; within-seed subsets are nested. Seeds also vary optimizer order. At 0 dB the five-fit mean rises from {sizes['mean'].iloc[0]:.3f} dB on {sizes.index[0]} training events to {sizes['mean'].iloc[-1]:.3f} dB on {sizes.index[-1]}. A fixed epoch ceiling yields more optimization steps for larger sets, so both data diversity and update count change.

{table(ap)}

These SNR and SI gains (dB) use centered nested subsets of the same selected 24-channel aperture on six validation events at 0 dB. Noise energy is recalibrated at each width. The head processes traces independently, so changes also reflect which channels were included and how much noise was mixed. The smaller apertures were not separately searched for their highest source SNR.

## Hybrid processing

{table(hy)}

At 0 dB on the six validation events, adding a head after f-k produces the largest ordinary gain among these four pipelines. Both cascades reduce SI-SDR relative to the noisy input. A lower pick MAE measures timing among matched picks and cannot establish higher detection sensitivity.

## Event-level failures

![Best and worst ten test earthquakes per method at 0 dB, from 145 candidates. The identity output is tied for every event.](../figures_v2/extremes.png)

The head’s best ten have median source SNR {ends.loc['best','median_source_snr']:.1f} dB versus {ends.loc['worst','median_source_snr']:.1f} dB in its worst ten, and median magnitude {ends.loc['best','median_magnitude']:.2f} versus {ends.loc['worst','median_magnitude']:.2f}. Their Ridgecrest counts are {int(ends.loc['best','ridgecrest'])} and {int(ends.loc['worst','ridgecrest'])}; median starting channel IDs are {ends.loc['best','median_first_channel']:.0f} and {ends.loc['worst','median_first_channel']:.0f}. Channel IDs are site-local. Pooled channel positions cannot be interpreted as a causal location effect. extremes.csv lists IDs, positions, magnitudes and shape scores for every method; ties use event-ID order. The identity baseline has no meaningful best or worst earthquake.
"""
