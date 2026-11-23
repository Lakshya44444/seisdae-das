# Detailed results

All numbers on the test days. The short version, method and limitations are in the [README](../README.md); every decision and its date are in [`decision_log.md`](decision_log.md).


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
   - **In SI-SDR (waveform shape), fine-tuning did better than training from scratch.** At every input SNR from −5 to 10 dB, all three fine-tuned runs scored above all three from-scratch runs (at 0 dB: −2.4 to −3.0 dB versus −3.6 to −4.9 dB). This was not tested with a paired interval, and the two used different training budgets.
   - **Against the filters.** At −10 dB input SNR all six runs beat the f-k filter (paired intervals exclude zero); at −5 dB the three fine-tuned runs and one from-scratch run do; at 0 dB and above none does. No run beat the 0.5–45 Hz bandpass with an interval excluding zero, and at 5 and 10 dB all six were worse than it.
3. **For detection, seisDAE trained on DAS is no better than a tuned bandpass or a tuned f-k filter.** All three gain about +0.07 AUC over the raw data. AUC differences [95% CI]:
   - tuned bandpass (1–10 Hz) minus raw data: +0.070 [0.022, 0.120];
   - tuned f-k (1000 m/s) minus raw data: +0.077 [0.012, 0.145];
   - seisDAE trained on DAS minus tuned bandpass: +0.005 [−0.068, 0.077];
   - seisDAE trained on DAS minus tuned f-k: −0.002 [−0.088, 0.086];
   - seisDAE + f-k minus tuned bandpass: +0.038 [−0.040, 0.118];
   - seisDAE + f-k minus tuned f-k: +0.032 [−0.032, 0.096];
   - for the M ≥ 1.5 events only, seisDAE trained on DAS minus tuned bandpass: +0.038 [−0.084, 0.169].
4. **Small events are near chance for seisDAE.** 50 of the 73 test earthquakes are below M1.5. For them the raw data score AUC 0.501, the bandpass 0.551 [0.468, 0.637] and from-scratch seisDAE 0.541 [0.454, 0.630]; only the methods with an f-k step stay above chance (f-k 0.616 [0.531, 0.699], cascade 0.631 [0.545, 0.714]) (`results/detection_retuned_by_magnitude_nontest.csv`).
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

   | Model | Seed | ROC AUC [95% CI] | minus tuned bandpass [95% CI] | minus released [95% CI] |
   |---|---|---|---|---|
   | Released | – | 0.601 [0.524, 0.675] | −0.033 [−0.079, 0.013] | – |
   | From scratch | 2026 | 0.638 [0.561, 0.714] | +0.005 [−0.068, 0.077] | +0.037 [−0.040, 0.118] |
   | From scratch | 2027 | 0.662 [0.588, 0.735] | +0.028 [−0.042, 0.100] | +0.061 [−0.011, 0.135] |
   | From scratch | 2028 | 0.620 [0.543, 0.695] | −0.014 [−0.083, 0.059] | +0.019 [−0.056, 0.095] |
   | Fine-tuned | 2026 | 0.643 [0.563, 0.721] | +0.010 [−0.045, 0.067] | +0.042 [−0.016, 0.102] |
   | Fine-tuned | 2027 | 0.604 [0.528, 0.679] | −0.029 [−0.086, 0.028] | +0.003 [−0.055, 0.063] |
   | Fine-tuned | 2028 | 0.648 [0.574, 0.721] | +0.014 [−0.040, 0.072] | +0.047 [−0.008, 0.103] |

   - Mean over seeds: from scratch 0.640 (range 0.620–0.662), fine-tuned 0.632 (range 0.604–0.648). Tuned bandpass 0.633, tuned f-k 0.640.
   - The spread between seeds (about 0.04 AUC) is larger than the difference between the two training strategies (0.008) and than the difference from the tuned filters.
   - No run differs from the tuned bandpass, or from the released model, with a 95% interval that excludes zero. In detection, retraining raised the average AUC (by 0.039 from scratch, 0.031 fine-tuned), but no single run is significantly better than the released model.
   - This is not an equal-budget comparison. From-scratch runs train for a fixed 100 epochs; fine-tuning uses the original retraining defaults with early stopping (seed 2026 stopped at epoch 21).
   - In detection AUC, fine-tuning did not improve on training from scratch. Starting from the released weights did not avoid the seed spread either (one fine-tuned run, 0.604, is level with the released model).
   - The best single run (from scratch, seed 2027, 0.662) is not used as the headline number, because it was not chosen in advance.

The full numbers are in `results/` (`summary.md`, `detection_*.csv`, `detection_retuned_*.csv`, `detection_sensitivity_test.csv`, `variants_*.csv`, `synthetic_seeds_*.csv`). The figures are in `results/roc.png` and `results/figures/`.
