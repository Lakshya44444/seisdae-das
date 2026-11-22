# Can seisDAE recover short DAS earthquake arrivals?

## Abstract

On 145 held-out earthquakes, the fixed output-head fit changed mean SNR gain at 0 dB input from 0.078 to 0.397 dB. Its scale-invariant gain was 0.135 dB. A training-fitted scalar achieved 3.041 dB ordinary gain and -0.000 dB scale-invariant gain. The magnitude analysis found a smaller absolute benefit for the smallest catalog events. Updating the last decoder block also improved on the released model, but the original head delivered the larger improvement.

## Research question and data

Does a denoising mask trained on seismometers preserve a DAS P arrival, and does adaptation preferentially improve smaller earthquakes? DAS records axial deformation along a fiber. The reference is a relatively clear field arrival that still contains noise. The Caltech P-phase product is already common-mode suppressed and filtered to 1–10 Hz. Each source trace spans 200 samples at 100 Hz. Catalog magnitudes use the source catalog’s magnitude types.

The original split reserves the 18 largest earthquakes on each array for development and pilot work. Per array, 12 went to training, three to validation and three to an excluded pilot. The remaining test events comprise 133 Long Valley and 12 Ridgecrest earthquakes. Every channel of an earthquake stays in the same split. This is transfer to other earthquakes on previously seen fibers.

## Methods

Among eligible blocks of 24 consecutive physical channels, the original protocol selects the block with highest median source SNR and shift spread at most eight samples. Per-channel shifts align the windows. Zero-based samples 86–155 supply the reference, and 9–78 supply early background. One noise scale per gather sets the input SNR. The neural input centers the demeaned clip inside 6,001 samples. Only the central 70 observed samples are scored. This offers no measured minute-long earthquake context.

The seven comparators are the mixture itself, one fitted scalar, a 1–10 Hz filter, db8 wavelet shrinkage, a symmetric f–k velocity fan, released GR seisDAE, and the same network with a validation-selected output head. The original head learns 16 mask weights using soft magnitude-ratio targets. The pick is an automatic STA/LTA threshold crossing. Timing error conditions on both traces producing a pick.

SNR gain compares squared residual energy before and after filtering. SI-SDR projects each demeaned estimate onto its reference and compares projected energy to residual energy. We difference this score against the noisy input per channel. Undefined zero-energy channels are excluded and squared correlations are clipped away from zero and one at 1e−12. CC retains the sign of correlation. Peak is the estimated-to-reference local peak amplitude ratio. Sign compares the first excursion exceeding 20% of each local peak, after subtracting its five-sample pre-onset mean, over the 15 samples starting at the nominal arrival. This is a waveform proxy, without analyst polarity labels.

Each earthquake contributes one mean waveform score. Pick recovery, sign agreement and timing pool their respective counts. Percentile intervals resample whole events within each array, preserving the observed site counts over 5,000 draws. All metrics and intervals at every input level are in baseline_summary.csv.

## Waveform fidelity at 0 dB input

| Method | Gain | SI gain | CC | Peak | Sign | Recovery | MAE (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bandpass | -1.052 | -5.551 | 0.485 | 1.755 | 0.599 | 0.937 | 71.968 |
| Fine-tuned | 0.397 | 0.135 | 0.687 | 2.224 | 0.577 | 0.954 | 66.209 |
| Noisy | 0.000 | 0.000 | 0.682 | 2.274 | 0.575 | 0.950 | 69.451 |
| Scalar | 3.041 | 0.000 | 0.682 | 1.063 | 0.575 | 0.950 | 69.451 |
| Wavelet | 0.203 | -0.188 | 0.676 | 2.188 | 0.582 | 0.952 | 67.717 |
| Zero-shot | 0.078 | 0.053 | 0.684 | 2.268 | 0.576 | 0.950 | 68.967 |
| f-k | 1.738 | -0.743 | 0.661 | 1.721 | 0.583 | 0.958 | 63.392 |

![Scale-invariant gain across all 145 test earthquakes. Shading gives pointwise 95% event-bootstrap intervals.](../../figures_v2/shape.png)

At 0 dB the highest ordinary gain was from Scalar and the highest SI gain from Fine-tuned. Multiplying a waveform by a positive constant preserves its shape and pick times. Peak ratios above unity indicate that the estimated arrival window still contains excess peak amplitude relative to the field reference.

## Does fine-tuning help more for smaller events?

| Group | Events | LV | RC | Gain Δ | CI low | CI high |
| --- | --- | --- | --- | --- | --- | --- |
| M < 1.1 | 26 | 26 | 0 | 0.274 | 0.264 | 0.284 |
| 1.1 ≤ M < 1.8 | 70 | 58 | 12 | 0.307 | 0.280 | 0.341 |
| M ≥ 1.8 | 49 | 49 | 0 | 0.360 | 0.312 | 0.417 |
| <22 dB | 44 | 42 | 2 | 0.279 | 0.267 | 0.292 |
| 22–<30 dB | 53 | 47 | 6 | 0.281 | 0.271 | 0.292 |
| ≥30 dB | 48 | 44 | 4 | 0.397 | 0.337 | 0.466 |

The table gives head-minus-zero-shot SNR gain at 0 dB input in magnitude and median source-SNR strata. The smallest-minus-largest contrast is -0.087 dB (95% CI -0.144 to -0.038). Both extreme magnitude bins contain only Long Valley earthquakes. Smaller events benefit in this benchmark, but the absolute gain increment is larger in the highest magnitude group. Source SNR and magnitude can covary. These descriptive differences do not isolate a causal magnitude effect. strata.csv and decoder_strata.csv give every method, level and group with intervals for gain, CC and recovery.

![Paired gain difference by magnitude across 145 earthquakes. Group sizes appear in the table. Shading gives pointwise 95% intervals.](../../figures_v2/magnitude.png)


In the site-specific 0 dB results, head SNR gain was 0.394 [0.362, 0.429] dB for Long Valley and 0.422 [0.290, 0.644] dB for Ridgecrest. The pooled estimate weights earthquakes equally, giving Long Valley greater influence. site_summary.csv contains all methods and levels by site.

## How much of the decoder should adapt?

The larger fit updates the last decoder convolution, batch-normalization scale and offset, and output mask (608 weights). The encoder, earlier decoder blocks and moving batch statistics stay fixed. Spatial features are cached with a one-frame halo. The implementation checks that cropping yields the original full-network masks at all active time frames. Adam uses a learning rate of 0.00003 and batches of 32 traces. Each training mixture contributes every fourth channel. Epoch zero is eligible; stopping uses validation mask cross-entropy with patience four. The original head uses its original optimizer and epoch budget. The final selected epochs were 40 for the head and 16 for the decoder. Both reached their configured budgets. The capacity and optimizer budgets differ, so this experiment cannot isolate the effect of parameter count.

The original validation set favored the head on SNR gain at 0 dB (0.928 vs 0.376 dB). The diagnostic candidate was retained for reporting under the frozen protocol. The test earthquakes were known from V1. They were excluded from fitting, checkpoint selection and all development ablations. A fresh untouched external test population would be a different experiment.

### Model estimates at 0 dB input

| Cohort | Model | Gain (dB) | SI gain (dB) | CC | Pick MAE (ms) |
| --- | --- | --- | --- | --- | --- |
| OOF | Decoder | 0.293 [0.229, 0.357] | 0.263 [0.182, 0.349] | 0.745 [0.723, 0.767] | 63.5 [57.2, 70.0] |
| OOF | Fine-tuned | 0.520 [0.412, 0.631] | 0.376 [0.262, 0.497] | 0.748 [0.726, 0.771] | 62.6 [56.2, 69.2] |
| OOF | Zero-shot | 0.237 [0.183, 0.292] | 0.216 [0.147, 0.289] | 0.744 [0.722, 0.766] | 64.6 [58.2, 71.0] |
| Test | Fine-tuned | 0.397 [0.363, 0.434] | 0.135 [0.110, 0.160] | 0.687 [0.677, 0.696] | 66.2 [64.0, 68.6] |
| Test | Zero-shot | 0.078 [0.067, 0.090] | 0.053 [0.043, 0.062] | 0.684 [0.675, 0.693] | 69.0 [66.7, 71.3] |
| Test | Decoder | 0.124 [0.111, 0.140] | 0.083 [0.069, 0.095] | 0.685 [0.676, 0.694] | 68.5 [66.3, 70.9] |

Brackets are 95% within-array event-bootstrap intervals. OOF denotes out-of-fold predictions on 30 development earthquakes. Each of three balanced outer folds holds out ten events. Of the other twenty, four stop training and sixteen fit the weights. Each development event receives one outer-fold prediction per variant and level. The final test uses separate 24-event training and six-event validation sets. At 0 dB the decoder-minus-head gain was -0.272 [-0.294, -0.252] dB; decoder-minus-released was 0.047 [0.044, 0.050] dB. More adaptation capacity did not improve the selected head under these budgets.

| Model | Fold 1 | Fold 2 | Fold 3 | SD |
| --- | --- | --- | --- | --- |
| Decoder | 0.258 | 0.324 | 0.297 | 0.033 |
| Fine-tuned | 0.456 | 0.582 | 0.523 | 0.063 |
| Zero-shot | 0.208 | 0.264 | 0.240 | 0.028 |

Fold entries are mean 0 dB gains in dB. SD is the sample standard deviation across three folds. Their training sets overlap; the folds are descriptive replications. Pooled OOF intervals condition on these fitted models and omit uncertainty from an entirely new training draw.

## What changes when the model sees a different context?

| Extension | Head gain | Released gain | Head SI | Released SI |
| --- | --- | --- | --- | --- |
| zero | 0.928 | 0.274 | 0.612 | 0.289 |
| reflect | 2.158 | 1.338 | -1.168 | -3.945 |
| noise | 1.661 | 0.636 | -0.638 | -3.287 |
| tile | 1.774 | 1.142 | -1.954 | -4.812 |
| stretch4 | 2.827 | 2.447 | 0.629 | 0.764 |

All entries are dB on the same six validation events at 0 dB input. The two frozen checkpoints were evaluated at all four mixture levels. Reflect mirrors the clip, tile repeats it, and noise surrounds it with seeded Gaussian values matched to each observed channel’s standard deviation. Stretch4 resamples each clip to four times its length before zero padding, and reverses that resampling after inference. It changes the apparent frequencies. All other extensions score the original center. Reflect and tile increase ordinary gain while hurting scale-invariant shape on this split. The stretch result justifies testing longer measured windows in a future acquisition; it does not supply missing physical observations.

## Independent and displaced backgrounds

| Cohort | Noise | Events | Gain (dB) | SI (dB) | MAE (ms) |
| --- | --- | --- | --- | --- | --- |
| Both sites | original | 6 | 0.928 | 0.612 | 52.587 |
| Both sites | other_block | 6 | 0.825 | 0.068 | 56.232 |
| Both sites | other_event | 6 | 0.906 | 0.631 | 49.429 |
| Ridgecrest | continuous | 3 | 0.302 | 0.032 | 78.485 |
| Ridgecrest | original | 3 | 0.997 | 0.553 | 56.806 |

The other-event condition cycles donors by event ID within the same validation site. It uses the donor's selected aperture at potentially different physical positions. The other-block condition uses the first disjoint eligible block after the target, wrapping through the event file if needed. Both use the published early-window rule. The other block shares the event and time and can retain correlated noise.

The external test uses 30 seconds beginning 2020-06-24T17:10:00Z from a separate SCEDC Ridgecrest recording. We fetched 720,000 raw sample bytes from trace rows 400–423. We detrended, differentiated, subtracted the channel median, bandpassed at 1–10 Hz and resampled to 100 Hz. Three fixed disjoint 70-sample snippets were each mixed with one validation earthquake. The manifest records every range and digest. The excerpts are candidate backgrounds without catalog screening. The matched Ridgecrest rows show how much performance depends on a single noise source; there is only one independent continuous recording.

## Training size and aperture width

![Mean gain on six validation earthquakes versus training-set size at each input SNR. Shading spans five seeded fits; it is a range, not a confidence interval.](../../figures_v2/size.png)

Each seed draws equal numbers of events per site; within-seed subsets are nested. Seeds also vary optimizer order. At 0 dB the five-fit mean rises from 0.300 dB on 2 training events to 0.928 dB on 24. A fixed epoch ceiling yields more optimization steps for larger sets, so both data diversity and update count change.

| Channels | Gain | SI gain | CC | MAE (ms) |
| --- | --- | --- | --- | --- |
| 8 | 0.885 | 0.655 | 0.760 | 44.375 |
| 16 | 0.945 | 0.693 | 0.785 | 50.737 |
| 24 | 0.928 | 0.612 | 0.768 | 52.587 |

These SNR and SI gains (dB) use centered nested subsets of the same selected 24-channel aperture on six validation events at 0 dB. Noise energy is recalibrated at each width. The head processes traces independently, so changes also reflect which channels were included and how much noise was mixed. The smaller apertures were not separately searched for their highest source SNR.

## Hybrid processing

| Order | Gain | SI gain | CC | MAE (ms) |
| --- | --- | --- | --- | --- |
| Fine-tuned | 0.928 | 0.612 | 0.768 | 52.587 |
| f-k | 2.256 | -3.467 | 0.662 | 47.770 |
| f-k then head | 2.376 | -3.434 | 0.666 | 44.718 |
| head then f-k | 2.361 | -3.409 | 0.665 | 46.950 |

At 0 dB on the six validation events, adding a head after f-k produces the largest ordinary gain among these four pipelines. Both cascades reduce SI-SDR relative to the noisy input. A lower pick MAE measures timing among matched picks and cannot establish higher detection sensitivity.

## Event-level failures

![Best and worst ten test earthquakes per method at 0 dB, from 145 candidates. The identity output is tied for every event.](../../figures_v2/extremes.png)

The head’s best ten have median source SNR 40.8 dB versus 24.1 dB in its worst ten, and median magnitude 2.20 versus 1.25. Their Ridgecrest counts are 1 and 3; median starting channel IDs are 2724 and 2706. Channel IDs are site-local. Pooled channel positions cannot be interpreted as a causal location effect. extremes.csv lists IDs, positions, magnitudes and shape scores for every method; ties use event-ID order. The identity baseline has no meaningful best or worst earthquake.

## Discussion and limitations

Selection favors high-SNR apertures; ground truth, manual picks and independent event-family labels are unavailable. Related earthquakes can make event-bootstrap intervals optimistic. The fibers, passband, clip length and local noise restrict external validity. Positive SNR gain alone cannot establish better detection, preserved amplitude or suitability for continuous monitoring. Order statistics and subgroup comparisons are exploratory.

## References

Course-list entry: Steinberg et al. (2023), *Neural Network Based Denoising of Regional Seismic Waveforms and Impact on Analysis of North Korean Nuclear Tests*. The journal record below supplies the published title and dates.

- Steinberg et al. Deep Neural Networks Based Denoising of Regional Seismic Waveforms and Impact on Analysis of North Korean Nuclear Tests. Published online 2024, journal volume 182 (2025). https://doi.org/10.1007/s00024-024-03491-3
- Heuel & Friederich (2022). Suppression of wind turbine noise from seismological data using nonlinear thresholding and denoising autoencoder. https://doi.org/10.1007/s10950-022-10097-6
- Zhu, Mousavi & Beroza (2019). Seismic signal denoising and decomposition using deep neural networks. https://doi.org/10.1109/TGRS.2019.2926772
- Tibi et al. (2021). Deep Learning Denoising Applied to Regional Distance Seismic Data in Utah. CTBT Science and Technology Conference. https://conferences.ctbto.org/event/7/contributions/1145/
- Li, Zhu, Biondi & Zhan (2023). Earthquake focal mechanisms with distributed acoustic sensing. https://doi.org/10.1038/s41467-023-39639-3
- Li (2023). Caltech DAS P-phase dataset. https://doi.org/10.22002/n47vy-s0s65
- Mousavi et al. (2019). STanford EArthquake Dataset (STEAD). https://doi.org/10.1109/ACCESS.2019.2947848
- Li et al. (2021). Rapid Response to the 2019 Ridgecrest Earthquake With Distributed Acoustic Sensing. https://doi.org/10.1029/2021AV000395
