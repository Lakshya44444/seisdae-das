# Denoising DAS data with seisDAE

**Objective.** Use seisDAE (Steinberg et al., 2023; code https://github.com/JanisHe/seisDAE) to denoise distributed acoustic sensing (DAS) data.

**What we did.** We trained the original seisDAE network from scratch on 60 s windows of the public Ridgecrest DAS dataset (California, 2020). We then compared it with standard filters on test days that were not used for training or for tuning the final settings. The test days were, however, reused across versions of this pipeline (see Limitations).

## TL;DR

- **Released seisDAE does not transfer to DAS.** Used as released, it distorts DAS waveforms more than any single filter, and its detection AUC (0.601) is below that of the tuned filters (difference not significant).
- **Retraining on DAS helps compared with the released model.** In the synthetic test all six retrained runs were 3.2–7.4 dB better in SI-SDR gain at every input SNR. In detection, the average AUC rose from 0.601 to 0.640 (from scratch) and 0.632 (fine-tuned); one fine-tuned seed (0.604) did not improve.
- **Retrained seisDAE is only as good as a tuned bandpass filter.** For detection it is level with a 1–10 Hz bandpass and an f-k filter (AUC about 0.63–0.64), not better. In waveform shape it never beat the bandpass.
- **The spread between training seeds (about 0.04 AUC) is as large as the differences between methods.** Six retrained runs (3 from scratch, 3 fine-tuned) all lead to the same conclusion.

Details follow.

## Results

**Detection test:** separating recorded earthquake windows (73) from noise windows (210) on test days. Settings were chosen by detection AUC on the 29 non-test days (165 earthquake windows, 174 noise windows). Each choice won in all 29 leave-one-day-out folds.

| Method | ROC AUC [95% CI] | AUC, events with M ≥ 1.5 only (23 events, exploratory) [95% CI] |
|---|---|---|
| Raw data | 0.563 [0.483, 0.641] | 0.698 [0.569, 0.817] |
| Bandpass 1–10 Hz | 0.633 [0.558, 0.708] | 0.811 [0.688, 0.914] |
| f-k filter, 1000 m/s | 0.640 [0.565, 0.714] | 0.691 [0.564, 0.804] |
| seisDAE, released model | 0.601 [0.524, 0.675] | 0.824 [0.718, 0.910] |
| seisDAE trained on DAS | 0.638 [0.561, 0.714] | 0.849 [0.759, 0.920] |
| seisDAE trained on DAS + f-k 250 m/s | 0.671 [0.599, 0.742] | 0.759 [0.653, 0.850] |

No method is a clear winner. The intervals overlap, and the paired differences below include zero. The M ≥ 1.5 column rests on 23 events.

**Synthetic test:** waveform shape at 0 dB input SNR, as SI-SDR gain in dB (0 = unchanged). Raw 0, bandpass 0.5–45 Hz −0.82, f-k −4.14, released seisDAE −8.51, seisDAE trained on DAS from scratch −4.37 [−7.77, −1.21] (seeds 2027, 2028: −3.60, −4.89), fine-tuned −2.95, −2.41, −2.38 (seeds 2026–2028), cascade −10.29. All seeds: `results/synthetic_seeds_*.csv`.

1. **Released seisDAE does not transfer to DAS waveforms.** It had the lowest SI-SDR gain of the single-stage methods at every input SNR.
2. **Training on DAS helps compared with the released model.** All six retrained runs were better than the released model in SI-SDR gain at every input SNR: from scratch by 3.2–6.1 dB, fine-tuned by 5.4–7.4 dB. Every paired 95% interval excludes zero (13–19 of 19 test events per comparison).
   - **Fine-tuning gave better waveform shape than training from scratch.** At every input SNR from −5 to 10 dB, all three fine-tuned runs scored above all three from-scratch runs (at 0 dB: −2.4 to −3.0 dB versus −3.6 to −4.9 dB). This was not tested with a paired interval, and the two used different training budgets.
   - **Against the filters.** At −10 dB input SNR all six runs beat the f-k filter (paired intervals exclude zero); at −5 dB the three fine-tuned runs and one from-scratch run do; at 0 dB and above none does. No run beat the 0.5–45 Hz bandpass with an interval excluding zero, and at 5 and 10 dB all six were worse than it.
3. **For detection, seisDAE trained on DAS is no better than a tuned bandpass or a tuned f-k filter.** All three gain about +0.07 AUC over the raw data. AUC differences [95% CI]:
   - tuned bandpass (1–10 Hz) minus raw data: +0.070 [0.022, 0.120];
   - tuned f-k (1000 m/s) minus raw data: +0.077 [0.012, 0.145];
   - seisDAE trained on DAS minus tuned bandpass: +0.005 [−0.068, 0.077];
   - seisDAE trained on DAS minus tuned f-k: −0.002 [−0.088, 0.086];
   - seisDAE + f-k minus tuned bandpass: +0.038 [−0.040, 0.118];
   - seisDAE + f-k minus tuned f-k: +0.032 [−0.032, 0.096];
   - for the M ≥ 1.5 events only, seisDAE trained on DAS minus tuned bandpass: +0.038 [−0.084, 0.169].
4. **Small events are at chance.** 50 of the 73 test earthquakes are below M1.5. For them the raw data score AUC 0.50, and no method separates them reliably.
5. **Not shown.** No method improved waveform shape at input SNR ≥ −5 dB against the recorded references.
6. **Traffic versus earthquake energy (reference-free, `results/fk_energy_summary.csv`).** In the 73 test earthquake windows:
   - DAS-trained seisDAE kept a median 11% of fast (> 1000 m/s) and 17% of slow (< 250 m/s) f-k energy. Its selectivity was −1.7 dB [−2.7, −0.8]: it lowers both and does not preferentially remove traffic.
   - The 1–10 Hz bandpass scored −8.5 dB, because it also removes fast energy above 10 Hz.
   - f-k (+26.0 dB) and the cascade (+16.0 dB) score high by construction.
   - The fast region also contains fast noise, so this is an indication, not a clean signal measure.
7. **What the pictures show (`results/figures/examples.png`, 3 earthquakes and 2 noise windows).**
   - In earthquake windows, the DAS-trained seisDAE output still contains most of the traffic.
   - In noise windows, seisDAE suppressed most of the noise, including traffic, but kept noisy single channels.
   - Applying f-k after seisDAE spread the energy of those noisy channels into coherent-looking artefacts. This is a likely cause of the cascade's false alarms. It is an observation from two windows, not a test.

8. **Released, from scratch and fine-tuned, over several seeds (`results/variants_*.csv`).** Same detection test (73 earthquake, 210 noise windows). No setting of any variant was chosen on validation or test days. Fine-tuning continued training of the released model on the same DAS training traces, with the original retraining code and its defaults (50 epochs, early stopping with patience 5). Every run is listed.

   | Model | Seed | ROC AUC [95% CI] | minus tuned bandpass [95% CI] |
   |---|---|---|---|
   | Released | – | 0.601 [0.524, 0.675] | −0.033 [−0.079, 0.013] |
   | From scratch | 2026 | 0.638 [0.561, 0.714] | +0.005 [−0.068, 0.077] |
   | From scratch | 2027 | 0.662 [0.588, 0.735] | +0.028 [−0.042, 0.100] |
   | From scratch | 2028 | 0.620 [0.543, 0.695] | −0.014 [−0.083, 0.059] |
   | Fine-tuned | 2026 | 0.643 [0.563, 0.721] | +0.010 [−0.045, 0.067] |
   | Fine-tuned | 2027 | 0.604 [0.528, 0.679] | −0.029 [−0.086, 0.028] |
   | Fine-tuned | 2028 | 0.648 [0.574, 0.721] | +0.014 [−0.040, 0.072] |

   - Mean over seeds: from scratch 0.640 (range 0.620–0.662), fine-tuned 0.632 (range 0.604–0.648). Tuned bandpass 0.633, tuned f-k 0.640.
   - The spread between seeds (about 0.04 AUC) is larger than the difference between the two training strategies (0.008) and than the difference from the tuned filters.
   - No run differs from the tuned bandpass with a 95% interval that excludes zero.
   - This is not an equal-budget comparison. From-scratch runs train for a fixed 100 epochs; fine-tuning uses the original retraining defaults with early stopping (seed 2026 stopped at epoch 21).
   - Fine-tuning did not improve on training from scratch here. Starting from the released weights did not avoid the seed spread either (one fine-tuned run, 0.604, is level with the released model).
   - The best single run (from scratch, seed 2027, 0.662) is not used as the headline number, because it was not chosen in advance.

**Conclusion.** We compared the released, the from-scratch and the fine-tuned seisDAE, each retrained with three seeds. The released seisDAE should not be used on DAS data without retraining. Retrained on DAS, from scratch or by fine-tuning, it detects earthquakes about as well as a 1–10 Hz bandpass filter chosen on validation data, and not better; the run-to-run spread is as large as any difference between methods. Fine-tuning gave better waveform shape than training from scratch, but no retrained run beat the bandpass filter in waveform shape, and none improved on the unprocessed input at input SNR ≥ −5 dB.

A possible reason, which we have **not tested**, is that seisDAE processes each channel on its own and does not use the coherence across neighbouring DAS channels.

The full numbers are in `results/` (`summary.md`, `detection_*.csv`, `detection_retuned_*.csv`, `detection_sensitivity_test.csv`, `variants_*.csv`). The figures are in `results/roc.png` and `results/figures/`.

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
