# seisDAE on Distributed Acoustic Sensing data (V4)

This document describes the fourth version (V4) of this project: why it was built, what data and methods it uses, every result it produced, and what those results do and do not show. Every number below was produced by the scripts in `src/v4/` and copied from their printed output or from the CSV files they write. `CHANGELOG.md` records each decision with its date and reason.

## 1. Summary

**Objective.** Use seisDAE, the denoising autoencoder of Heuel & Friederich used by Steinberg et al. (2023) (code: https://github.com/JanisHe/seisDAE), to denoise distributed acoustic sensing (DAS) data.

**What was done.** seisDAE was trained from scratch on 60 s windows of the public Ridgecrest DAS dataset. Training, validation and test windows come from separate days. The network was tested in three ways:

- a synthetic test, where recorded earthquakes are mixed with recorded noise;
- a field test on weak recorded earthquakes;
- a detection test on recorded earthquake and noise windows, with no added noise.

The network architecture and training code are the unchanged originals.

**Main findings (V4.2).**

1. **The released (seismometer-trained) seisDAE does not transfer to DAS.** In the synthetic test it had the lowest SI-SDR gain of all single-stage methods at every input SNR. For example, at 0 dB input SNR its mean SI-SDR gain was −8.51 dB. Only the cascade (seisDAE then f-k) scored lower, at −5 dB and above.
2. **Training on DAS helps, compared with the released model.** In the synthetic test, the DAS-trained model was 3.6–6.0 dB better in SI-SDR gain than the released model at every input SNR (paired, 14–17 of 19 events).
3. **Detection test.** The ability to separate earthquake windows from noise windows (ROC AUC) was:
   - raw data: 0.563
   - f-k filter: 0.602
   - DAS-trained seisDAE: 0.638
   - DAS-trained seisDAE followed by f-k: 0.671

   The pre-registered comparisons, DAS-trained seisDAE against the raw data and against f-k, were **not resolved at 95% confidence**. The cascade (seisDAE then f-k) was better than the raw data under both bootstrap schemes. It was better than f-k alone under the day bootstrap, but not under the window bootstrap.
4. **No improvement at low false-alarm rates.** At a 5% false-alarm rate none of the paired comparisons with the raw data (DAS-trained seisDAE, the cascade, f-k) showed significantly more detected earthquakes. The released model was not compared directly. The denoisers also raise the detection statistic on some noise windows.
5. **Waveform fidelity was not shown to improve.** Against recorded references, which still contain traffic, no method improved SI-SDR at input SNRs of −5 dB and above.

**One-sentence conclusion.** seisDAE must be retrained on DAS to be useful. A DAS-trained seisDAE, especially when followed by an f-k filter, separates earthquakes from noise somewhat better than the raw data. The evidence is moderate, part of the analysis was decided after an earlier test result was seen, and no improvement in waveform fidelity was demonstrated.

## 2. Why V4 was needed

V1–V3 (the earlier versions of this repository) used the Caltech `DASPphaseData` release. It stores 200 samples (2 s) per channel around each P pick, and V3 analysed 70-sample (0.7 s) snippets. The released seisDAE model expects 6001 samples (60 s at 100 Hz; `models/gr_mixed_stft.config`), so each snippet was placed inside 60 s of zeros. With the model's STFT (`nperseg = 99`, `nfft = 198`) such a snippet occupies 3 of 121 time frames. The network therefore saw almost no noise context. V3 reported near-identity behaviour (+0.40 dB ordinary gain and +0.14 dB SI-SDR gain at 0 dB). V4 gives seisDAE input of the length it was designed for.

## 3. Data

**DAS recordings.**
- Source: Ridgecrest DAS dataset of the Southern California Earthquake Data Center, on AWS Open Data (`s3://scedc-pds/Ridgecrest_DAS`).
- Recording period: 23 June to 29 July 2020, during the Ridgecrest aftershock sequence.
- Format: hourly SEG-Y files sampled at 250 Hz, each with 1250 channels.
- Data are read with HTTP byte ranges (`src/v4/das.py`).

The file layout (a 3600-byte header, then per channel a 240-byte header and 900000 little-endian float32 samples) reproduces the byte offsets of 24 traces downloaded and checksummed in V3. The trace headers give 2020, day 176, 17:00:00, which matches the file name `2020062417`.

**Channels.**
- `das_info.csv` gives coordinates for 1150 of the 1250 file rows. Its `index` column is the file row.
- The block used is rows **918–1013** (96 channels). The selection rule was fixed before any event was examined: centre the block in the longest run of located rows (870–1061).
- Median channel spacing: **7.95 m**. Block length: about 760 m.

**Preprocessing (per 60 s window).**
1. Linear detrend.
2. Removal of the median across the 96 channels (common mode).
3. 4th-order 0.5 Hz high-pass filter.
4. Resampling from 250 Hz to 100 Hz.

Each window is 96 × 6001 samples. No further band limiting is applied; noise suppression is left to the methods being compared.

**Earthquake catalogue.**
- Source: SCEDC FDSN event service, through ObsPy (`src/v4/catalog.py`).
- The service returned events thousands of kilometres away despite the radius parameter, so the code filters events by epicentral distance itself.
- Result: 508 events with M ≥ 1.0 within 0.6° of the block centre; 2898 events of any magnitude within 1° (used to screen noise windows); 19 events with M ≥ 3.5 within 6°.

## 4. Dataset

**Earthquake windows.**
- The 240 highest-ranked candidates (ranked by magnitude − 1.5 log10 distance) and the remaining test-day events with M 0.5–1.6 were downloaded. The configuration allows up to 120 such events; 46 were available.
- The predicted P arrival (straight ray, 6.0 km/s) is placed at sample 1000 (10 s). In inspected examples the observed P arrives about 1–1.5 s later; the event is still inside the window.
- 273 earthquake windows were downloaded. Hour files that are missing from the archive were skipped.

**Noise windows.** Random 60 s windows with no catalogued event (any magnitude, within 1°) from 60 s before to the end of the window, and no regional M ≥ 3.5 event in the preceding 300 s. There are 6 per day, plus 168 extra windows on test days (Section 6.4).

**Splits.** Day number since 23 June, modulo 5: 0, 1, 2 → train; 3 → validation; 4 → test. The sets never share a day.

**Which windows count as earthquakes (V4.2 rule).** Traffic on the road beside the fiber produces slow, slanted arrivals with large energy. An energy-based SNR cannot tell traffic from an earthquake. V4.2 therefore measures the SNR after an f-k fan filter that keeps apparent velocities above 1000 m/s. Traffic crosses the block at tens of m/s; a local earthquake crosses it at km/s.

A window is an earthquake ("strong") if at least 30% of its channels reach this fast-component SNR ≥ 10 dB. The SNR is the power in the event window over the power in the 1–9 s pre-P window. The filter is used only for this decision; every method receives the unfiltered gather.

`tests/test_v4.py::test_fast_snr_ignores_traffic` shows why this matters. A synthetic traffic arrival and a synthetic earthquake both have 11.3 dB energy SNR, but the fast-component SNR is 0.1 dB for the traffic and 15.4 dB for the earthquake.

| Rule | Train / validation / test earthquakes | Training traces |
|---|---|---|
| V4.0: energy SNR, ≥30% of channels ≥10 dB | 46 / 8 / 24 | 2142 |
| V4.2: fast-component SNR, ≥30% of channels ≥10 dB | 35 / 6 / 19 | 1894 (count reported by `--stats`) |

Training data are single-channel traces whose selection SNR is ≥10 dB, from train-day earthquakes, plus every fourth channel of the 132 train-day noise windows.

## 5. Model and training

The architecture and training code are the original seisDAE ones. The trained model's configuration is identical to that of the released model (`python -m src.v4.collect` printed no differences):

| Setting | Value |
|---|---|
| Input | STFT of a 6001-sample trace: 100 frequency bins × 121 time frames × 2 (real and imaginary parts), normalised by its maximum |
| Network | U-Net, depth 6, 8 filters in the first level, 3 × 3 kernels, stride 2, dropout 0.1, no bias |
| Output | A mask between 0 and 1 for each time-frequency cell; mask × STFT, followed by the inverse STFT, gives the denoised trace |
| Loss, optimiser | Binary cross-entropy, Adam |
| Data augmentation | On: signal and noise traces are mixed at random SNR during training |

- **Training.** From scratch (random initial weights), not fine-tuned. Pinned original trainer (`scripts/upstream_train.py`, upstream commit `26c9ba0`), 100 epochs, batch size 16, on a laptop CPU.
- **V3 comparison.** V3, in contrast, fine-tuned only the final layer (16 or 608 parameters) of the released model.
- **Software.** Python 3.13 with TensorFlow 2.20 and tf-keras 2.20 (`requirements-py313.txt`); the pinned TensorFlow 2.16.2 has no Python 3.13 build.
- **Inference check.** The inference code used in the evaluation reproduces the original `predict` function to a relative peak error of 9.5 × 10⁻⁸ (released model) and 1.5 × 10⁻⁷ (V4.2 DAS model).

## 6. Evaluation design

### 6.1 Methods compared

| Method | What it does |
|---|---|
| Noisy | No processing |
| Scalar | Multiplies the input by a constant. A control: it shows how much "gain" pure attenuation earns under the ordinary metric |
| Bandpass | Butterworth band-pass filter |
| f-k | Fan filter that keeps apparent velocities above a threshold, using all 96 channels together |
| seisDAE (released) | The released model (`gr_mixed_stft`), applied channel by channel |
| seisDAE (DAS-trained) | The model trained here, applied channel by channel |
| seisDAE (DAS) + f-k | DAS-trained seisDAE followed by the f-k filter: single-channel denoising plus multichannel coherence |

**Settings** were chosen on validation days only, then frozen:
- Rule: the bandpass, f-k and cascade settings maximise the mean validation SI-SDR gain; the scalar maximises the mean ordinary gain.
- Frozen values: bandpass 0.5–45 Hz, f-k 250 m/s, cascade f-k 250 m/s, scalar 0.8.

Both baseline choices lie at the weak end of their grids. The bandpass picked its widest band. The f-k filter picked the lowest velocity of the grid (250, 500, 1000, 2000 m/s), which keeps 83% of the energy of white noise. See limitation 3 in Section 9.

### 6.2 Synthetic test

- **Mixing.** Each of the 19 test earthquakes is mixed with 2 of the 42 original test-day noise gathers at −10, −5, 0, 5 and 10 dB. One scale factor is applied per gather.
- **Reference.** The recorded earthquake gather.
- **Scored channels.** Only channels whose reference reaches the 10 dB selection SNR.
- **Ordinary gain:** reduction of the squared error, in dB.
- **SI-SDR gain:** the mean per-channel change in scale-invariant signal-to-distortion ratio. Pure scaling gives 0 dB.
- **CC:** the mean correlation with the reference.
- **Intervals:** 95% event-bootstrap intervals.

### 6.3 Field test

For 16 weak test-day earthquakes, recorded as they are: median channel SNR, fraction of channels with SNR ≥ 10 dB, and neighbouring-channel coherence. There is no reference, so these are descriptive only (Section 9).

### 6.4 Detection test (main V4.2 test)

- **Positives:** every downloaded test-day earthquake window without an overlapping catalogued event, 73 windows. Most are small: 50 have M < 1.5.
- **Negatives:** 210 test-day noise windows. 42 were original, and 168 were added later under a plan fixed before the run (CHANGELOG).
- **Statistic:** the median over channels of the power ratio, in dB, between the window from 0.5 s before to 20 s after the predicted P and the 1–9 s pre-P window. It does not depend on overall scale.
- **Measures:**
  - the ROC curve and its area (AUC), the probability that an earthquake window scores higher than a noise window;
  - the fraction of earthquakes detected when each method's threshold is set to a 5% or 10% false-alarm rate on that method's own noise windows.
- **Uncertainty:** a paired bootstrap with 5000 draws that resamples earthquake and noise windows. A day-block bootstrap, which resamples the 7 test days, is a robustness check.

## 7. Results

### 7.1 V4.0 (first run, energy-based selection)

At 0 dB input SNR, the mean SI-SDR gain was −3.47 dB for DAS-trained seisDAE, −3.65 dB for f-k, −0.71 dB for bandpass and −9.91 dB for the released model. DAS-trained seisDAE beat the released model at every input SNR (paired +6.2 to +7.1 dB, 20–22 of 24 events).

Inspecting the figures afterwards showed test references dominated by traffic. This led to the V4.2 selection rule. V4.0 had no detection test.

### 7.2 V4.2 synthetic test (19 test earthquakes, 42 noise gathers)

SI-SDR gain in dB [95% CI]:

| Input SNR | Bandpass | f-k | seisDAE (released) | seisDAE (DAS-trained) | seisDAE (DAS) + f-k |
|---|---|---|---|---|---|
| −10 dB | −0.28 [−0.45, −0.14] | −2.84 [−4.24, −1.65] | −5.28 [−9.01, −1.37] | **+0.74 [−1.60, 3.00]** | −4.78 [−8.83, −0.83] |
| −5 dB | −0.50 | −3.33 | −6.28 | −1.62 [−4.46, 1.05] | −7.37 |
| 0 dB | −0.82 | −4.14 | −8.51 | −4.37 [−7.77, −1.21] | −10.29 |
| 5 dB | −1.42 | −5.54 | −11.44 | −7.64 | −13.76 |
| 10 dB | −2.37 | −7.76 | −14.87 | −11.24 | −17.74 |

The scalar control has an SI-SDR gain of 0 by construction. Its ordinary gain was 1.89, 1.78, 1.48, 0.74 and −0.83 dB at −10 to 10 dB.

Paired SI-SDR differences [95% CI] (wins out of 19 events):

| Comparison | −10 dB | 0 dB | 10 dB |
|---|---|---|---|
| DAS-trained minus released | +6.01 [3.77, 8.22] (17) | +4.14 [2.42, 5.80] (15) | +3.62 [1.00, 6.14] (14) |
| DAS-trained minus f-k | +3.58 [2.00, 5.21] (17) | −0.23 [−2.79, 2.33] (10) | −3.49 [−7.01, 0.05] (9) |
| Cascade minus DAS-trained | −5.52 [−7.62, −3.50] (2) | −5.92 [−8.26, −3.74] (0) | −6.50 [−8.87, −4.39] (0) |

### 7.3 V4.2 field test (16 weak test-day earthquakes, descriptive)

| Method | Median channel SNR (dB) | Channels ≥ 10 dB | Neighbour coherence |
|---|---|---|---|
| Noisy | 2.92 | 0.17 | 0.217 |
| Bandpass | 2.94 | 0.18 | 0.216 |
| f-k | 4.46 | 0.22 | 0.427 |
| seisDAE (released) | 12.81 | 0.56 | 0.327 |
| seisDAE (DAS-trained) | 9.50 | 0.47 | 0.229 |
| seisDAE (DAS) + f-k | 12.72 | 0.63 | 0.594 |

### 7.4 V4.2 detection test (73 earthquake windows, 210 noise windows)

Window bootstrap, 95% CI:

| Method | AUC [95% CI] | Detected at 5% false alarm | Detected at 10% false alarm |
|---|---|---|---|
| Noisy | 0.563 [0.483, 0.641] | 0.096 | 0.192 |
| Bandpass | 0.566 [0.485, 0.644] | 0.110 | 0.164 |
| f-k | 0.602 [0.525, 0.678] | 0.123 | 0.178 |
| seisDAE (released) | 0.601 [0.524, 0.675] | 0.151 | 0.260 |
| seisDAE (DAS-trained) | 0.638 [0.561, 0.714] | 0.082 | 0.233 |
| seisDAE (DAS) + f-k | 0.671 [0.599, 0.742] | 0.027 | 0.219 |

The median detection statistic on noise windows was 1.23 dB for the raw data and 5.61 dB after DAS-trained seisDAE. The denoiser raises the statistic on noise as well as on earthquakes.

Paired AUC differences:

| Comparison | Difference | Window bootstrap 95% CI | Day bootstrap 95% CI |
|---|---|---|---|
| **DAS-trained minus Noisy (pre-registered)** | +0.075 | [−0.007, 0.155] | [−0.002, 0.183] |
| **DAS-trained minus f-k (pre-registered)** | +0.036 | [−0.049, 0.121] | [−0.029, 0.138] |
| DAS-trained minus released | +0.037 | [−0.040, 0.118] | [−0.005, 0.095] |
| Cascade minus Noisy | +0.108 | [0.017, 0.200] | [0.047, 0.166] |
| Cascade minus f-k | +0.070 | [−0.002, 0.143] | [0.038, 0.109] |
| f-k minus Noisy | +0.038 | [−0.007, 0.085] | [−0.005, 0.087] |

The day bootstrap has only 7 clusters. Bootstrap intervals from so few clusters can be too narrow, so it is a robustness check, not the main analysis.

With the original 42 noise windows the detection rate of DAS-trained seisDAE at 5% false alarm was 0.23. With 210 noise windows it was 0.08. The earlier, larger effect came from the small noise sample.

Exploratory split, not planned in advance, using 42 noise windows (AUC): M < 1.5 (50 events): Noisy 0.52, DAS-trained 0.59, cascade 0.67. M ≥ 1.5 (23 events): Noisy 0.71, DAS-trained 0.88, cascade 0.79.

### 7.5 Figures and files

All of these are in `results_v4/` on the run machine:

- **`summary.md`:** the generated tables.
- **`roc.png`:** ROC curves.
- **`detection_auc.csv`, `detection_paired.csv`, `detection_paired_byday.csv`:** detection statistics.
- **`synthetic_*.csv`, `field_*.csv`:** per-event scores.
- **`figures_v4/`:** example gathers, the SI-SDR gain against input SNR (`levels.png`) and an example trace.

## 8. What the results support

| Claim | Supported? |
|---|---|
| Released seisDAE should not be applied to DAS without retraining | Yes. It had the lowest SI-SDR gain of the single-stage methods at every SNR, and its detection AUC (0.601) was no better than plain f-k (0.602). |
| Retraining on DAS improves seisDAE compared with the released model | Yes for waveform fidelity (paired +3.6 to +6.0 dB). Not resolved for detection (AUC +0.037, CI includes 0). |
| DAS-trained seisDAE separates earthquakes from noise better than the raw data | A positive trend (AUC 0.638 vs 0.563). Not resolved at 95% confidence (CI −0.007 to 0.155). |
| DAS-trained seisDAE followed by f-k separates them better than the raw data | Yes (AUC 0.671; +0.108, CI 0.017 to 0.200; day bootstrap 0.047 to 0.166). This was a secondary comparison. |
| A method detects more earthquakes than the raw data at a low (5%) false-alarm rate | Not shown. None of the paired comparisons (DAS-trained, cascade, f-k) resolved a difference; released seisDAE was not compared directly. |
| Any method improves waveform fidelity against recorded references at input SNR ≥ −5 dB | No |
| DAS-trained seisDAE improves SI-SDR at −10 dB | Not resolved (+0.74, CI −1.60 to 3.00). It was better than f-k at −10 dB (+3.58, CI 2.00 to 5.21). |

## 9. Limitations

1. **Test days were reused.** The same test days served V4.0. The V4.2 selection rule and the detection test were introduced after the V4.0 test scores had been seen. Only the extension to 210 noise windows was planned before its result. V4.2 is therefore not an untouched held-out evaluation.
2. **Few, small, clustered events.** There are 6 validation earthquakes, 19 synthetic-test earthquakes and 73 detection-test earthquakes, 50 of them below M1.5, over 7 test days. Events and traffic cluster in time, which the window bootstrap does not model.
3. **Weakly tuned baselines.** Baseline settings were chosen with synthetic SI-SDR against references that contain traffic. That criterion favours weak filtering: the f-k velocity sits at the edge of its grid. Neither the baselines nor seisDAE were tuned for detection.
4. **Imperfect references.** Synthetic references are recorded earthquakes that still contain traffic, so SI-SDR penalises removal of that traffic.
5. **Field-test measures are not quality measures.** Suppressing the pre-event window raises SNR, and an f-k filter raises neighbouring-channel coherence by construction.
6. **Biased selection.** Earthquake selection uses an f-k criterion, so the chosen references hold less slow energy than unselected windows.
7. **Trainer validation split.** The original trainer draws its validation set at trace level, so neighbouring channels of one event can fall in both its training and validation sets. This affects only checkpoint selection, not the tests.
8. **Uncatalogued events.** Noise windows are screened only for catalogued events.
9. **One fiber.** All data come from one fiber section of about 760 m, recorded over 37 days.
10. **Software versions.** TensorFlow 2.20 instead of the pinned 2.16.2. The inference adapter was verified against the original code (Section 5).
11. **Example-figure choice.** The `median_snr_db` column in `manifest.csv` is the original energy-based SNR. It is used only to choose example figures.

## 10. Decision history

| Step | What changed | Why |
|---|---|---|
| V4.0 | 60 s Ridgecrest windows, energy-SNR selection, training from scratch | Give seisDAE its intended input |
| V4.1 (rejected) | Common-mode removal switched off | Hypothesis that it removed earthquake energy. Three re-downloaded events kept the same SNR (18.3→18.4, 19.4→19.4, 5.2→5.3 dB), so the change was reverted |
| V4.2 | Fast-component (f-k) SNR selection, new training, detection test | Traffic had been labelled as earthquakes |
| V4.2 + noise | 168 extra test-day noise windows; analysis fixed beforehand | Detection intervals were too wide with 42 noise windows |

## 11. Reproducing V4.2

On a laptop with Python 3.13 (`requirements-py313.txt`) and a checkout of seisDAE at commit `26c9ba0` in `vendor/seisDAE`:

```
python -m src.v4.das --inspect
python -m src.v4.catalog
python -m src.v4.build --download
python -m src.v4.build --reclassify
python -m src.v4.build --export
python scripts/upstream_train.py vendor/seisDAE --signals data/v4/upstream/signal --noise data/v4/upstream/noise --count 1894 --batch 16 --epochs 100 --threads 8 --name das_ridgecrest_v42
python -m src.v4.collect vendor/seisDAE --name das_ridgecrest_v42
python -m src.v4.build --extra-noise
python -m src.v4.evaluate --upstream vendor/seisDAE
python -m src.v4.figures
python -m src.v4.detection_stats
```

The full download is roughly 3–5 GB and takes several hours. Cached windows are reused if a step is restarted. Offline tests:

```
python -m unittest tests.test_v4
```

## 12. Files

| Path | Content |
|---|---|
| `src/v4/config.json` | Every setting |
| `src/v4/das.py` | SEG-Y reader, preprocessing, geometry |
| `src/v4/catalog.py` | Catalogue download with distance filter |
| `src/v4/build.py` | Window download, selection, splits, export, extra noise windows |
| `src/v4/methods.py` | Bandpass, f-k, seisDAE inference, scores |
| `src/v4/evaluate.py` | Validation selection, synthetic, field and detection tests |
| `src/v4/detection_stats.py` | ROC, AUC, paired and day bootstraps |
| `src/v4/figures.py`, `src/v4/peek.py` | Figures and the generated summary |
| `src/v4/collect.py` | Copies the trained model and compares its configuration |
| `scripts/upstream_train.py` | Runs the pinned original trainer |
| `tests/test_v4.py` | Offline tests |
| `CHANGELOG.md` | Dated record of every decision and result |

## 13. References

- Steinberg, A., et al. (2023). *Neural Network Based Denoising of Regional Seismic Waveforms and Impact on Analysis of North Korean Nuclear Tests.* Code: https://github.com/JanisHe/seisDAE.
- Atterholt, J., et al. (2022). A unified wavefield-partitioning approach for distributed acoustic sensing. *Geophysical Journal International*, 228, 1410–1418. https://doi.org/10.1093/gji/ggab407
- Lapins, S., et al. (2024). DAS-N2N: machine learning distributed acoustic sensing (DAS) signal denoising without clean data. *Geophysical Journal International*, 236, 1026–1041. https://doi.org/10.1093/gji/ggad460
- van den Ende, M., Lior, I., Ampuero, J.-P., Sladen, A., Ferrari, A., & Richard, C. A self-supervised deep learning approach for blind denoising and waveform coherence enhancement in distributed acoustic sensing data (jDAS). https://github.com/martijnende/jDAS
- Southern California Earthquake Data Center, Ridgecrest DAS dataset, `s3://scedc-pds/Ridgecrest_DAS`.
