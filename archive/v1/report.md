# Transfer of seisDAE to fiber-optic earthquake records

## Abstract

We evaluated a seismometer-trained denoising autoencoder on 145 previously unused earthquakes recorded by distributed acoustic sensing (DAS). For each earthquake we used an early background window to corrupt a later, higher-SNR P-wave segment at four input SNRs. Fine-tuning 16 output weights increased mean SNR gain from 0.078 to 0.397 dB at 0 dB input. The gain difference favored fine-tuning for every test event, although a frequency–wavenumber filter and a fixed amplitude multiplier achieved larger reductions in squared error at this level. At 5 and 10 dB input, the fine-tuned model gave the largest mean SNR gain among the methods evaluated. A separate example shows inference on a full observed two-second clip. Short windows, residual noise in the reference and the imbalance across the two arrays limit the inference to this benchmark.

## Research question

DAS measures axial deformation along a cable. Its directional response, local coupling and spatially coherent noise differ from conventional seismometer recordings. Can a time-frequency mask learned from seismometers recover a DAS P arrival? Does updating the final mask improve recovery on smaller earthquakes?

The approach follows the seismic denoising work of Zhu et al., Tibi et al. and Heuel & Friederich. Steinberg et al. evaluated learned denoising on regional and teleseismic waveforms. We use the released seisDAE GR checkpoint and the short P-phase windows associated with the Caltech focal-mechanism study. PhaseNet-DAS helped the original authors extract the P-wave windows.

## Data and experimental design

| Property | Long Valley | Ridgecrest |
|---|---:|---:|
| Earthquakes in source archive | 151 | 30 |
| Waveform rows × samples per event | 4,670 × 200 | 4,718 × 200 |
| Sampling rate | 100 Hz | 100 Hz |
| Training / validation / earlier pilot | 12 / 3 / 3 | 12 / 3 / 3 |
| New test events and catalog magnitude range | 133, M 0.60–2.60 | 12, M 1.20–1.77 |
| Median aperture source SNR on test events | 26.51 dB | 26.65 dB |

The previous 36-event experiment used the 18 highest-magnitude events per array. We froze the resulting model and used all 145 remaining catalog events for the new comparison. Each split is independent by earthquake ID. Both arrays are represented in training and testing. The test therefore measures transfer to additional earthquakes on the same fibers. It also changes the magnitude distribution.

Every event file contains a 2 s P-phase matrix, channel SNRs and timing shifts. Catalogs supply ID, origin time, location, depth and magnitude. The parent study describes common-mode filtering and 1–10 Hz bandpass processing. The HDF5 waveform attributes record shape and sampling interval but supply no calibrated amplitude unit. Figure 1 of the workflow reproduces the upstream seismometer example; the following gather shows the released DAS product.

![Released DAS aperture](../../figures/02_raw_gather.png)

### Reference construction

Among regularly spaced, nonoverlapping candidate blocks we selected the 24 consecutive channel IDs with the highest median SNR, subject to a maximum eight-sample spread in the published shifts. Subtracting the local median shift places nearby channels on a common time axis. Samples 9–78 supply the background and 86–155 supply the arrival. Both windows are 70 samples long, with disjoint time support. The nominal arrival is derived from the clip center. We demean each row and scale the noise once per aperture to obtain −5, 0, 5 or 10 dB. All methods receive the same mixtures. Each earthquake supplies one reference and one background realization.

### Models and comparisons

The released encoder–decoder produces two softmax masks using real and imaginary STFT input planes. We used a Hann window of 99 samples, a 198-point FFT and a 50-sample hop. Zero-shot inference uses the published weights. Fine-tuning updates the 16 weights of the bias-free 1×1 output convolution. The training target is the signal-to-total magnitude ratio in active time-frequency bins. We selected six channels per training aperture and used four mixture levels. Adam used a learning rate of 0.0001 and minibatches of 2,048 bins. Validation loss reached its observed minimum at epoch 40. The model input is centered in 6,001 samples, and evaluation discards padding.

The classical methods are a forward–backward 1–10 Hz Butterworth bandpass, db8 wavelet thresholding and an f-k velocity fan with a nominal 500 m/s threshold and 10 m spacing. The f-k filter sees neighboring channels. A fourth control fits one scalar to minimize reference-energy-normalized squared error over the training mixtures. Its fitted value, 0.46750, is fixed across all test events and input levels.

![Upstream RN.BAVN reproduction](../../figures/01_reproduction.png)

The original seismometer example agreed with the upstream predictor to less than one part per million of peak recovered amplitude. The separate conventional training pilot used 24 STEAD earthquakes and 11 candidate FDSN background windows. All transfer results below use the released GR weights as their starting point. [Source-training record](../../results/source_training/README.md)

## Results

**Mean gain in SNR, in dB.** We compute one value per 24-channel aperture before averaging across earthquakes. Bold indicates the highest mean in each column.

| Method | −5 dB input | 0 dB input | 5 dB input | 10 dB input |
|---|---:|---:|---:|---:|
| Noisy | 0.00 | 0.00 | 0.00 | 0.00 |
| Scalar | **5.15** | **3.04** | -0.44 | -4.83 |
| Bandpass | 0.20 | -1.05 | -3.55 | -7.29 |
| Wavelet | 0.24 | 0.20 | 0.13 | -0.01 |
| f-k | 3.16 | 1.74 | -0.86 | -4.58 |
| Zero-shot | 0.07 | 0.08 | 0.09 | 0.11 |
| Fine-tuned | 0.37 | 0.40 | **0.42** | **0.41** |

![SNR improvement across tested levels](../../figures/05_snr.png)

At 0 dB, fine-tuning improved on zero-shot in **145 of 145 events**. The mean paired improvement was **0.319 dB** with a **95% bootstrap interval of 0.296–0.345 dB**. We resampled entire earthquakes 5,000 times within each array, preserving their observed proportions. These intervals describe event-to-event variation on the two sampled arrays. At 0 dB, the mean gains of zero-shot and fine-tuned seisDAE were 0.077 and 0.394 dB for Long Valley, and 0.086 and 0.422 dB for Ridgecrest. The corresponding f-k results were 1.764 and 1.442 dB.

![Within-event improvement for each array](../../figures/08_paired_gain.png)

| Method at 0 dB | Waveform CC | Normalized RMSE | Pick MAE (ms) | Matched / reference picks |
|---|---:|---:|---:|---:|
| Noisy | 0.682 | 1.000 | 69.5 | 3,294 / 3,466 |
| Scalar | 0.682 | 0.705 | 69.5 | 3,294 / 3,466 |
| Bandpass | 0.485 | 1.136 | 72.0 | 3,247 / 3,466 |
| Wavelet | 0.676 | 0.977 | 67.7 | 3,299 / 3,466 |
| f-k | 0.661 | 0.822 | 63.4 | 3,320 / 3,466 |
| Zero-shot | 0.684 | 0.991 | 69.0 | 3,292 / 3,466 |
| Fine-tuned | 0.687 | 0.956 | 66.2 | 3,305 / 3,466 |

CC is the mean channelwise Pearson correlation, averaged by event. Normalized RMSE divides the error norm by the reference norm per gather. A lower value indicates less relative waveform error. STA/LTA uses trailing 4- and 12-sample energy windows and a 1.8 threshold near the nominal P time. The timing statistic pools shifts only where both the reference and estimated trace produce a crossing. Reference-pick counts and recovery rates expose misses. The reported error measures agreement between automated threshold crossings.

![All methods on an identical mixture](../../figures/03_methods.png)
![Single trace against proxy reference](../../figures/04_trace.png)
![Spectra for the same example channel](../../figures/07_spectrogram.png)
![Output-layer training and validation losses](../../figures/06_training.png)

## Application to an observed clip

We applied both models directly to the original 200-sample clip for event 71123139, using the same 24 channels and no added noise. All panels share a color scale. This check shows what each method does to an observed arrival. The true underlying clean waveform is unknown, so the figure has no assigned SNR gain.

![Full observed event](../../results/field_example/observed.png)

## Discussion and limits

Fine-tuning improved SNR gain and correlation over the released model at every tested SNR. The absolute mean gain was modest. At 0 dB, pure amplitude scaling had the largest gain while leaving both Pearson correlation and pick timing unchanged. f-k increased gain more than fine-tuning at 0 dB, yet had a lower mean CC. Those differences matter when the task depends on waveform shape or first-motion timing.

The reference already contains field noise, and the short early background comes from the same recording. Highest-SNR aperture selection favors visible arrivals. Only 12 new test events are from Ridgecrest. Bootstrap intervals preserve that observed mix and assume independence between earthquake IDs within each site. Related event families could reduce the effective sample size. The scoring window is shorter than one STFT window, and short-trace edge handling also affects the classical filters. The comparison establishes performance for these processed snippets. Continuous DAS data, independent noise recordings and manual picks would test its wider applicability.

## References


- Steinberg et al. Deep Neural Networks Based Denoising of Regional Seismic Waveforms and Impact on Analysis of North Korean Nuclear Tests. Published online 2024, journal volume 182 (2025). https://doi.org/10.1007/s00024-024-03491-3
- Heuel & Friederich (2022). Suppression of wind turbine noise from seismological data using nonlinear thresholding and denoising autoencoder. https://doi.org/10.1007/s10950-022-10097-6
- Zhu, Mousavi & Beroza (2019). Seismic signal denoising and decomposition using deep neural networks. https://doi.org/10.1109/TGRS.2019.2926772
- Tibi et al. (2021). Deep Learning Denoising Applied to Regional Distance Seismic Data in Utah. See upstream bibliography at https://github.com/JanisHe/seisDAE
- Li, Zhu, Biondi & Zhan (2023). Earthquake focal mechanisms with distributed acoustic sensing. https://doi.org/10.1038/s41467-023-39639-3
- Li (2023). Caltech DAS P-phase dataset. https://doi.org/10.22002/n47vy-s0s65
- Mousavi et al. (2019). STanford EArthquake Dataset (STEAD). https://doi.org/10.1109/ACCESS.2019.2947848
