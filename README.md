# Denoising DAS data with seisDAE

**Objective.** Use seisDAE (Steinberg et al., 2023; code https://github.com/JanisHe/seisDAE) to denoise distributed acoustic sensing (DAS) data.

**What we did.** We retrained the original seisDAE network on 60 s windows of the public Ridgecrest DAS dataset (California, 2020), both from scratch and by fine-tuning the released model, with three seeds each. We then compared it with standard filters on test days that were not used for training or for tuning the final settings. The test days were, however, reused across versions of this pipeline (see Limitations).

## Description

DAS turns an optical fiber into thousands of closely spaced strain sensors. Roadside fiber records earthquakes, but also strong traffic noise, noise common to all channels and badly coupled channels. seisDAE is a U-Net denoising autoencoder that works on the time–frequency (STFT) image of one trace and predicts which parts are signal; it was built and trained for seismometer data. This project asks whether it also works on DAS, and tests the answer honestly.

- **Data.** 37 days (23 June – 29 July 2020) of the public Ridgecrest DAS archive (SCEDC, AWS Open Data), 96 neighbouring channels (about 760 m of fiber), 60 s earthquake windows from the SCEDC catalogue and random noise windows. Windows are split by day into train, validation and test days. Only the needed bytes are read from the hourly SEG-Y files with HTTP range requests. File format, channel mapping, labelling and window counts: [`docs/data.md`](docs/data.md).
- **Models.** The released seisDAE model; seisDAE retrained from scratch on DAS; and the released model fine-tuned on DAS. Both retrained versions use the original seisDAE code unchanged, with three random seeds each.
- **Baselines.** A bandpass filter and an f-k (apparent-velocity) filter, with settings chosen without test days.
- **Tests.** Waveform shape: SI-SDR on test earthquakes mixed with recorded test noise at −10 to 10 dB. Detection: ROC AUC on 73 recorded test earthquake windows against 210 test noise windows. Traffic removal: how much slow (traffic) and fast (earthquake) f-k energy each method keeps. All comparisons are paired, with 95% bootstrap intervals.
- **Everything is recorded.** Every result table is in `results/`; every decision, including those made after a result was seen, is dated in [`docs/decision_log.md`](docs/decision_log.md); the full write-up is in [`docs/results.md`](docs/results.md).

## TL;DR

- **Released seisDAE does not transfer to DAS.** Used as released, it distorts DAS waveforms more than any single filter (SI-SDR), and its detection AUC (0.601) is below that of the tuned filters (difference not significant).
- **Retraining on DAS clearly helps waveform shape, less clearly detection.** In SI-SDR, all six retrained runs (3 from scratch, 3 fine-tuned) beat the released model by 3.2–7.4 dB at every input SNR, all significant. In detection, the average AUC rose from 0.601 to 0.632–0.640, but no single run is significantly better than the released model.
- **At very low SNR, retrained seisDAE beats the f-k filter.** At −10 dB input SNR all six retrained runs had higher SI-SDR gain than f-k (paired intervals exclude zero), and their gain over the unprocessed input was positive (+0.3 to +1.0 dB, not significant).
- **Otherwise it is only as good as a tuned bandpass filter.** For detection it is level with a 1–10 Hz bandpass and an f-k filter (AUC about 0.63–0.64), not better. In SI-SDR it never beat the bandpass, and above 0 dB it was worse.
- **The spread between training seeds (about 0.04 AUC) is as large as the differences between methods.** Every retrained run leads to the same conclusion.

## Key numbers

Test days. Detection: 73 earthquake and 210 noise windows; settings of the filters chosen on the 29 non-test days. SI-SDR: 19 test earthquakes mixed with recorded noise; gain in dB (0 = unchanged input). Ranges are over the three seeds.

| Method | Detection ROC AUC | SI-SDR gain at 0 dB | SI-SDR gain at −10 dB |
|---|---|---|---|
| Raw data | 0.563 | 0 | 0 |
| Bandpass (detection: 1–10 Hz; SI-SDR: 0.5–45 Hz) | 0.633 | −0.82 | −0.28 |
| f-k filter (detection: 1000 m/s; SI-SDR: 250 m/s) | 0.640 | −4.14 | −2.84 |
| seisDAE, released | 0.601 | −8.51 | −5.28 |
| seisDAE, from scratch (3 seeds) | 0.620–0.662 (mean 0.640) | −4.89 to −3.60 | +0.33 to +0.83 |
| seisDAE, fine-tuned (3 seeds) | 0.604–0.648 (mean 0.632) | −2.95 to −2.38 | +0.55 to +1.03 |

Confidence intervals, paired comparisons, magnitude bins, the traffic-energy measure and the pictures: [`docs/results.md`](docs/results.md).

**Conclusion.** We compared the released, the from-scratch and the fine-tuned seisDAE, each retrained with three seeds. The released seisDAE should not be used on DAS data without retraining. Retrained on DAS, from scratch or by fine-tuning, it detects earthquakes about as well as a 1–10 Hz bandpass filter chosen on the non-test days, and not better; the run-to-run spread is as large as any difference between methods. In SI-SDR, fine-tuning did better than training from scratch, and every retrained run beat f-k at −10 dB, but no retrained run beat the bandpass, and none improved significantly on the unprocessed input.

A possible reason, which we have **not tested**, is that seisDAE processes each channel on its own and does not use the coherence across neighbouring DAS channels.

## Method

1. **Data.** Ridgecrest DAS (SCEDC, AWS Open Data): 250 Hz, 1250 channels, 23 June – 29 July 2020. We use 96 neighbouring channels (rows 918–1013, 7.95 m spacing, about 760 m of fiber).
2. **Windows.** Each window is 60 s, resampled to 100 Hz (96 × 6001 samples), after detrending, common-mode removal and a 0.5 Hz high-pass filter.
   - Earthquake windows come from the SCEDC catalogue: origin time, location and magnitude, ranked by magnitude and distance.
   - Noise windows are random times with no catalogued earthquake.
3. **Choosing clean training earthquakes.** Cars on the road next to the fiber give strong, slow, slanted signals. An earthquake reaches all channels almost at once. To train seisDAE and to build synthetic-test references, we kept only earthquake windows where at least 30% of channels have SNR ≥ 10 dB after an f-k filter that keeps apparent velocities above 1000 m/s, which removes traffic. Every method receives unfiltered data.
   - **The detection test does not use this f-k criterion.** Its earthquakes are all downloaded catalogue events on test days without another catalogued event in the window.
   - The training signals were still chosen with an f-k criterion, so seisDAE may have learned to favour fast, f-k-like signals.
4. **Splits.** By day: day number modulo 5 gives 0, 1, 2 → train; 3 → validation; 4 → test.
   - **Two kinds of earthquake counts.** The clean-labelled earthquakes (step 3) are 35 / 6 / 19 (train / validation / test). They are used for training and for the synthetic test. The detection test and the detection tuning use **all** downloaded catalogue earthquake windows instead: 36 on the 7 validation days, 165 on the 29 non-test days, 73 on the 7 test days. So "6 validation earthquakes" and "36 validation earthquake windows" refer to the same days with two different labelling rules.
   - Noise windows: 132 / 42 / 42, plus 168 extra test noise windows for the detection test (210 in total).
5. **Model.** seisDAE U-Net exactly as released: STFT input of 100 × 121 × 2, depth 6, 8 base filters, 0.1 dropout, binary cross-entropy loss, Adam optimiser. We trained it **from scratch** with the original training code: 100 epochs, batch size 16, laptop CPU. The trained model's configuration is identical to the released one. As a variant, we also **fine-tuned** the released model on the same traces with the original retraining code (its defaults; batch size 16). Each was run with seeds 2026, 2027 and 2028.
   - The fine-tuned model keeps the released architecture (loaded from the `.h5` file). The original retraining code leaves the architecture fields of its `.config` file empty (`None`). Inference reads only the STFT settings, which are filled in.
6. **Baselines.** No setting is chosen on test days.
   - Synthetic test: SI-SDR on validation days picked a 0.5–45 Hz bandpass and f-k at 250 m/s.
   - Detection test: `src/detection_tuning.py --tune-on non-test` uses AUC on the 29 non-test days (bandpass and f-k have no trained parameters). It picked a 1–10 Hz bandpass, f-k at 1000 m/s and a cascade at 250 m/s. Each choice won in 29 of 29 leave-one-day-out folds.
   - The cascade choice is optimistic, because seisDAE saw part of the train-day data.
   - Tuning on the 7 validation days alone had picked f-k at 250 m/s, which was not stable.
   - Every grid setting is also reported on the test days (`results/detection_sensitivity_test_nontest.csv`).
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
- **Baseline tuning matters.** Tuned by synthetic SI-SDR on validation days (references contain traffic), the bandpass stayed almost open (0.5–45 Hz), and seisDAE then looked better than the baselines in detection. Tuned by detection AUC on the 29 non-test days (1–10 Hz bandpass, f-k 1000 m/s), the baselines are level with seisDAE. The detection numbers above use the second, detection-based tuning. Both versions are in `results/`.
- **The f-k choice was switched after test results were seen.** Detection tuning on the 7 validation days alone first chose f-k at 250 m/s. The sensitivity table on test had already shown 1000 m/s at 0.640 before the re-tuning, and the validation choice not being the best on test is what prompted the re-tuning. The re-tuning itself used no test data: it used the 29 non-test days, chose 1000 m/s, and that choice won all 29 leave-one-day-out folds. The bandpass choice (1–10 Hz) was the same in both tunings.
- **Noisy channels.** Single noisy channels are not masked. They survive seisDAE, and an f-k filter smears them into coherent artefacts.
- **Training runs.** Three seeds each for training from scratch and for fine-tuning, all on the same 35 training earthquakes. The field, f-k energy and picture analyses use only the first from-scratch run (seed 2026). Three seeds give a rough spread, not a precise one.
- **Training length.** The from-scratch runs use a fixed 100 epochs. In the seed 2027 run the best validation loss came at epoch 93, so longer training might still help a little. Fine-tuning stopped early (seed 2026: epoch 21).
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
  variants.py        released vs from-scratch vs fine-tuned seisDAE, several seeds (detection)
  synthetic_seeds.py the same variants on the synthetic SI-SDR test
  examples.py        before/after pictures of recorded windows
  figures.py         figures and results/summary.md
  collect.py         copy the trained model into models/
  peek.py            quick-look plot of one window
scripts/train_seisdae.py   runs the original seisDAE trainer
models/                    released seisDAE model (+ trained DAS model)
tests/                     offline tests
results/                   final results
docs/data.md               Ridgecrest data format, channel mapping, windows and splits
docs/results.md            detailed results (all tables and intervals)
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

Variants (fine-tuned released model, and more seeds):

```
python scripts/train_seisdae.py vendor/seisDAE --signals data/ridgecrest/upstream/signal --noise data/ridgecrest/upstream/noise --count 1894 --batch 16 --threads 8 --seed 2026 --retrain-from models/gr_mixed_stft.h5 --name das_finetune_s2026
python -m src.collect vendor/seisDAE --name das_finetune_s2026 --as variants/finetune_s2026
python -m src.variants
python -m src.synthetic_seeds
```

To run the offline tests: `python -m unittest discover -s tests`.

## References

- Steinberg, A., et al. (2023). *Neural Network Based Denoising of Regional Seismic Waveforms and Impact on Analysis of North Korean Nuclear Tests.* Code: https://github.com/JanisHe/seisDAE
- Atterholt, J., et al. (2022). A unified wavefield-partitioning approach for distributed acoustic sensing. *Geophys. J. Int.*, 228, 1410–1418. https://doi.org/10.1093/gji/ggab407
- Lapins, S., et al. (2024). DAS-N2N: machine learning DAS signal denoising without clean data. *Geophys. J. Int.*, 236, 1026–1041. https://doi.org/10.1093/gji/ggad460
- Li, Z., Shen, Z., Yang, Y., Williams, E., Wang, X., & Zhan, Z. (2021). Rapid response to the 2019 Ridgecrest earthquake with distributed acoustic sensing. *AGU Advances*, 2, e2021AV000395. https://doi.org/10.1029/2021AV000395
