# Denoising DAS data with seisDAE

**Objective.** Use seisDAE (Steinberg et al., 2023; code https://github.com/JanisHe/seisDAE) to denoise distributed acoustic sensing (DAS) data.

**What we did.** We trained the original seisDAE network from scratch on 60 s windows of the public Ridgecrest DAS dataset (California, 2020). We then compared it with standard filters on test days that were not used for training or for tuning the final settings. The test days were, however, reused across versions of this pipeline (see Limitations).

## Result in short

**Detection test:** separating recorded earthquake windows (73) from noise windows (210) on test days. Settings were chosen on validation days by detection AUC.

| Method | ROC AUC [95% CI] | AUC, events with M ≥ 1.5 only (23 events, exploratory) [95% CI] |
|---|---|---|
| Raw data | 0.563 [0.483, 0.641] | 0.698 [0.569, 0.817] |
| Bandpass 1–10 Hz | 0.633 [0.558, 0.708] | 0.811 [0.688, 0.914] |
| f-k filter, 250 m/s | 0.602 [0.525, 0.678] | 0.671 [0.540, 0.790] |
| seisDAE, released model | 0.601 [0.524, 0.675] | 0.824 [0.718, 0.910] |
| seisDAE trained on DAS | 0.638 [0.561, 0.714] | 0.849 [0.759, 0.920] |
| seisDAE trained on DAS + f-k 250 m/s | 0.671 [0.599, 0.742] | 0.759 [0.653, 0.850] |

No method is a clear winner. The intervals overlap, and the paired differences below include zero. The M ≥ 1.5 column rests on 23 events.

**Synthetic test:** waveform shape at 0 dB input SNR, as SI-SDR gain in dB (0 = unchanged). Raw 0, bandpass 0.5–45 Hz −0.82, f-k −4.14, released seisDAE −8.51, seisDAE trained on DAS −4.37 [−7.77, −1.21], cascade −10.29.

1. **Released seisDAE does not transfer to DAS waveforms.** It had the lowest SI-SDR gain of the single-stage methods at every input SNR.
2. **Training on DAS helps compared with the released model.** The DAS-trained model was 3.6–6.0 dB better in SI-SDR gain at every input SNR (paired, 14–17 of 19 test events).
3. **For detection, seisDAE trained on DAS is no better than a simple tuned bandpass.** AUC differences [95% CI]:
   - tuned 1–10 Hz bandpass minus raw data: +0.070 [0.022, 0.120];
   - seisDAE trained on DAS minus tuned bandpass: +0.005 [−0.068, 0.077];
   - seisDAE + f-k minus tuned bandpass: +0.038 [−0.040, 0.118];
   - for the M ≥ 1.5 events, seisDAE trained on DAS minus tuned bandpass: +0.038 [−0.084, 0.169].
4. **Small events are at chance.** 50 of the 73 test earthquakes are below M1.5. For them the raw data score AUC 0.50, and no method separates them reliably.
5. **Not shown.** No method improved waveform shape at input SNR ≥ −5 dB against the recorded references.

**Conclusion.** The released seisDAE should not be used on DAS data without retraining. Retrained on DAS, it detects earthquakes about as well as a 1–10 Hz bandpass filter chosen on validation data, and not better. Neither was shown to improve waveform fidelity.

A possible reason, which we have **not tested**, is that seisDAE processes each channel on its own and does not use the coherence across neighbouring DAS channels.

The full numbers are in `results/` (`summary.md`, `detection_*.csv`, `detection_retuned_*.csv`, `detection_sensitivity_test.csv`). The figures are in `results/roc.png` and `results/figures/`.

## Method

1. **Data.** Ridgecrest DAS (SCEDC, AWS Open Data): 250 Hz, 1250 channels, 23 June – 29 July 2020. We use 96 neighbouring channels (rows 918–1013, 7.95 m spacing, about 760 m of fiber).
2. **Windows.** Each window is 60 s, resampled to 100 Hz (96 × 6001 samples), after detrending, common-mode removal and a 0.5 Hz high-pass filter.
   - Earthquake windows come from the SCEDC catalogue: origin time, location and magnitude, ranked by magnitude and distance.
   - Noise windows are random times with no catalogued earthquake.
3. **Choosing clean training earthquakes.** Cars on the road next to the fiber give strong, slow, slanted signals. An earthquake reaches all channels almost at once. To train seisDAE and to build synthetic-test references, we kept only earthquake windows where at least 30% of channels have SNR ≥ 10 dB after an f-k filter that keeps apparent velocities above 1000 m/s, which removes traffic. Every method receives unfiltered data.
   - **The detection test does not use this f-k criterion.** Its earthquakes are all downloaded catalogue events on test days without another catalogued event in the window.
   - The training signals were still chosen with an f-k criterion, so seisDAE may have learned to favour fast, f-k-like signals.
4. **Splits.** By day: day number modulo 5 gives 0, 1, 2 → train; 3 → validation; 4 → test. The result is 35 / 6 / 19 earthquakes, and 132 / 42 / 42 noise windows (plus 168 extra test noise windows for the detection test).
5. **Model.** seisDAE U-Net exactly as released: STFT input of 100 × 121 × 2, depth 6, 8 base filters, 0.1 dropout, binary cross-entropy loss, Adam optimiser. We trained it **from scratch** with the original training code: 100 epochs, batch size 16, laptop CPU. The trained model's configuration is identical to the released one.
6. **Baselines.** All settings are chosen on validation days only.
   - Synthetic test: SI-SDR picked a 0.5–45 Hz bandpass and f-k at 250 m/s.
   - Detection test (`src/detection_tuning.py`): AUC picked a 1–10 Hz bandpass, f-k at 250 m/s and a cascade at 250 m/s. On validation, every f-k setting scored at or below the raw data.
   - Every grid setting is also reported on the test days (`results/detection_sensitivity_test.csv`). The f-k velocity that was not chosen, 1000 m/s, would have scored 0.640 on test.
7. **Tests.**
   - **Synthetic:** 19 test earthquakes mixed with recorded test-day noise at −10 to 10 dB. Scored with SI-SDR (insensitive to scale) against the recorded earthquake.
   - **Detection (main test):** 73 recorded test-day earthquakes and 210 noise windows, with no mixing.
     - **Score:** each method's output gets one number per window. That number is the median over the 96 channels of the power ratio, in dB, between the window from 0.5 s before to 20 s after sample 1000 (10 s) and the 1–9 s window.
     - **Sample 1000:** in an earthquake window this is the predicted P arrival, from the catalogue origin time and a 6 km/s straight-ray travel time. In a noise window it is the same fixed 10 s mark, with no event.
     - **Known-time test:** this is a known-time detection test, not blind detection.
     - **Bias:** the score rewards any method that removes energy outside the earthquake's frequency band, so it measures filtering as much as denoising.
     - **AUC:** the probability that a random earthquake window scores higher than a random noise window. Bootstrap intervals are computed over windows and over days.
   - **Pictures:** before/after examples are in `results/figures/` (`gather_synthetic.png`, `gather_field.png`, `trace.png`, `examples.png`).

## Limitations

- **Test days were reused.** The test days were first used by an earlier version of this pipeline. The earthquake-labelling rule and the detection test were decided after that first result was seen. Only the addition of 168 noise windows was planned before its result.
- **Small samples.** 6 validation, 19 synthetic-test and 73 detection-test earthquakes (50 of them below M1.5), over 7 test days.
- **Baseline tuning matters.** With the synthetic SI-SDR criterion (references contain traffic) the bandpass stayed almost open (0.5–45 Hz), and seisDAE then looked better than the baselines. With detection-based tuning on validation days (1–10 Hz), the advantage disappeared. Both versions are reported.
- **Unreliable tuning.** The validation days hold only 36 earthquake windows (6 strong). The validation choice of f-k velocity (250 m/s) was not the best on test (1000 m/s scored 0.640), so the chosen settings are not reliable. `python -m src.detection_tuning --tune-on non-test` re-tunes bandpass and f-k on all 30 non-test days, with a leave-one-day-out count of how often each setting wins.
- **One training run.** seisDAE was trained once, from scratch, on 35 earthquakes. The spread between training runs (seeds) was not measured.
- **Proxy metrics.** Detection AUC is a proxy. `python -m src.fk_energy` adds a reference-free measure: how much slow (traffic) energy a method removes compared with fast (earthquake) energy kept.
- **Many comparisons.** Six methods, several tests and a magnitude split were compared. A single interval that barely excludes zero should be read as suggestive.
- **Noisy references.** Synthetic references are recorded earthquakes that still contain traffic, so negative SI-SDR gains partly reflect removed traffic.
- **One fiber.** All data come from one fiber section, recorded over 37 days.

Every decision, with its date and reason, including the earlier results, is in [`docs/decision_log.md`](docs/decision_log.md).

## Repository layout

```
src/
  config.json        all settings
  das.py             read Ridgecrest SEG-Y windows, preprocessing
  catalog.py         earthquake catalogue (SCEDC)
  build.py           download windows, label, split, export training traces
  methods.py         bandpass, f-k, seisDAE inference, scores
  evaluate.py        synthetic and detection tests
  detection_stats.py ROC, AUC, bootstrap, AUC by magnitude
  detection_tuning.py re-tune baselines by detection AUC (validation or all non-test days)
  fk_energy.py       slow (traffic) energy removed vs fast (earthquake) energy kept
  examples.py        before/after pictures of recorded windows
  figures.py         figures and results/summary.md
  collect.py         copy the trained model into models/
  peek.py            quick-look plot of one window
scripts/train_seisdae.py   runs the original seisDAE trainer
models/                    released seisDAE model (+ trained DAS model)
tests/                     offline tests
results/                   final results
docs/decision_log.md       dated record of every decision and result
archive/earlier_attempt/   first attempt on 2 s Caltech DAS snippets (not used)
```

## How to run

Tested on a Windows laptop, CPU only, with Python 3.13.2, TensorFlow 2.20 and tf-keras 2.20 (`pip install -r requirements.txt`). The pinned seisDAE code targets older TensorFlow and works here through tf-keras. Clone seisDAE into `vendor/seisDAE` and check out commit `26c9ba0`.

Rough cost on that laptop:

| Step | Time | Disk |
|---|---|---|
| Download | several hours (byte-range requests to AWS) | about 3 GB in `data/ridgecrest/` |
| Training (100 epochs) | 30–80 min (17–47 s per epoch) | |
| Evaluation | about 30 min | |
| `detection_tuning` | 20–40 min | |

`--count` in the training command is the number of signal traces printed by `python -m src.build --export` (1894 in our run).

```
python -m src.das --inspect
python -m src.catalog
python -m src.build --download
python -m src.build --reclassify
python -m src.build --export
python scripts/train_seisdae.py vendor/seisDAE --signals data/ridgecrest/upstream/signal --noise data/ridgecrest/upstream/noise --count 1894 --batch 16 --epochs 100 --threads 8 --name das_ridgecrest_v42
python -m src.collect vendor/seisDAE --name das_ridgecrest_v42
python -m src.build --extra-noise
python -m src.evaluate --upstream vendor/seisDAE
python -m src.figures
python -m src.detection_stats
python -m src.detection_tuning
python -m src.detection_tuning --tune-on non-test
python -m src.fk_energy
python -m src.examples
```

To run the offline tests: `python -m unittest discover -s tests`.

## References

- Steinberg, A., et al. (2023). *Neural Network Based Denoising of Regional Seismic Waveforms and Impact on Analysis of North Korean Nuclear Tests.* Code: https://github.com/JanisHe/seisDAE
- Atterholt, J., et al. (2022). A unified wavefield-partitioning approach for distributed acoustic sensing. *Geophys. J. Int.*, 228, 1410–1418. https://doi.org/10.1093/gji/ggab407
- Lapins, S., et al. (2024). DAS-N2N: machine learning DAS signal denoising without clean data. *Geophys. J. Int.*, 236, 1026–1041. https://doi.org/10.1093/gji/ggad460
- Li, Z., Shen, Z., Yang, Y., Williams, E., Wang, X., & Zhan, Z. (2021). Rapid response to the 2019 Ridgecrest earthquake with distributed acoustic sensing. *AGU Advances*, 2, e2021AV000395. https://doi.org/10.1029/2021AV000395
