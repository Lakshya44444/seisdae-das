# Denoising DAS data with seisDAE

**Objective.** Use seisDAE (Steinberg et al., 2023; code https://github.com/JanisHe/seisDAE) to denoise distributed acoustic sensing (DAS) data.

**What we did.** We trained the original seisDAE network from scratch on 60 s windows of the public Ridgecrest DAS dataset (California, 2020). We then compared it with standard filters on test days that were never used for training.

## Result in short

| Measure | Raw data | Bandpass | f-k filter | seisDAE (released) | seisDAE (trained on DAS) | seisDAE (DAS) + f-k |
|---|---|---|---|---|---|---|
| Earthquake vs noise separation, ROC AUC [95% CI] | 0.563 [0.48, 0.64] | 0.566 | 0.602 [0.53, 0.68] | 0.601 | **0.638** [0.56, 0.71] | **0.671** [0.60, 0.74] |
| Waveform shape at 0 dB input SNR, SI-SDR gain in dB [95% CI] | 0 | −0.82 | −4.14 | −8.51 | −4.37 [−7.77, −1.21] | −10.29 |

1. **Released seisDAE does not transfer to DAS.** It had the lowest SI-SDR gain of the single-stage methods at every input SNR. Only the cascade scored lower, at −5 dB and above.
2. **Training on DAS helps compared with the released model.** The DAS-trained model was 3.6–6.0 dB better in SI-SDR gain at every input SNR (paired, 14–17 of 19 test events).
3. **Earthquake detection** (AUC differences, 95% CI):
   - seisDAE trained on DAS against raw data: +0.075 [−0.007, 0.155]. **Not significant.**
   - seisDAE followed by an f-k filter against raw data: +0.108 [0.017, 0.200]. This is **suggestive only**: the lower bound is close to zero, and it is one of several comparisons.
   - Most test earthquakes are very small (50 of 73 below M1.5), and for them the raw data are at chance (AUC 0.50). In an exploratory split (not planned in advance), for the 23 events with M ≥ 1.5, seisDAE trained on DAS reached AUC 0.849 against 0.698 for the raw data (+0.151 [0.030, 0.281]) and 0.671 for f-k (+0.179 [0.052, 0.312]). See `results/detection_by_magnitude.csv`.
4. **Not shown.** No improvement in waveform shape at input SNR ≥ −5 dB, and no gain in detections at a low (5%) false-alarm rate.

The full numbers are in `results/summary.md` and `results/detection_*.csv`. The ROC curves are in `results/roc.png`.

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
6. **Baselines.** Bandpass and f-k settings were chosen on validation days only, by the synthetic-test criterion (SI-SDR): 0.5–45 Hz and 250 m/s. `src/detection_tuning.py` re-tunes them on validation days by detection AUC, and reports every setting on the test days.
7. **Tests.**
   - **Synthetic:** 19 test earthquakes mixed with recorded test-day noise at −10 to 10 dB. Scored with SI-SDR (insensitive to scale) against the recorded earthquake.
   - **Detection (main test):** 73 recorded test-day earthquakes and 210 noise windows, with no mixing.
     - **Score:** each method's output gets one number per window. That number is the median over the 96 channels of the power ratio, in dB, between the 20.5 s after the predicted P arrival and the 1–9 s before it.
     - **AUC:** the probability that a random earthquake window scores higher than a random noise window. Bootstrap intervals are computed over windows and over days.
   - **Pictures:** before/after examples are in `results/figures/` (`gather_synthetic.png`, `gather_field.png`, `trace.png`, `examples.png`).

## Limitations

- **Test days were reused.** The test days were first used by an earlier version of this pipeline. The earthquake-labelling rule and the detection test were decided after that first result was seen. Only the addition of 168 noise windows was planned before its result.
- **Small samples.** 6 validation, 19 synthetic-test and 73 detection-test earthquakes (50 of them below M1.5), over 7 test days.
- **Weak baselines.** The baselines were tuned on validation days with the synthetic SI-SDR criterion, whose references still contain traffic. This favoured weak filtering (the f-k velocity sits at the edge of its grid). The detection-based re-tuning addresses this.
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
  detection_tuning.py re-tune baselines by detection AUC on validation days
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

Requires Python 3.13 and `pip install -r requirements.txt`. Then clone seisDAE into `vendor/seisDAE` and check out commit `26c9ba0`.

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
python -m src.examples
```

To run the offline tests: `python -m unittest discover -s tests`.

## References

- Steinberg, A., et al. (2023). *Neural Network Based Denoising of Regional Seismic Waveforms and Impact on Analysis of North Korean Nuclear Tests.* Code: https://github.com/JanisHe/seisDAE
- Atterholt, J., et al. (2022). A unified wavefield-partitioning approach for distributed acoustic sensing. *Geophys. J. Int.*, 228, 1410–1418. https://doi.org/10.1093/gji/ggab407
- Lapins, S., et al. (2024). DAS-N2N: machine learning DAS signal denoising without clean data. *Geophys. J. Int.*, 236, 1026–1041. https://doi.org/10.1093/gji/ggad460
- Li, Z., Shen, Z., Yang, Y., Williams, E., Wang, X., & Zhan, Z. (2021). Rapid response to the 2019 Ridgecrest earthquake with distributed acoustic sensing. *AGU Advances*, 2, e2021AV000395. https://doi.org/10.1029/2021AV000395
