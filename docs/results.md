# Detailed results

All DAS numbers are on the test days. Items 9–11 were added after the first review (26 November 2026). The short version, method and limitations are in the [README](../README.md); every decision and its date are in [`decision_log.md`](decision_log.md).


**Detection test:** separating recorded earthquake windows (73) from noise windows (210) on test days. Settings were chosen by detection AUC on the 29 non-test days (165 earthquake windows, 174 noise windows). Each choice won in all 29 leave-one-day-out folds.

| Method | ROC AUC [95% CI] | AUC, events with M ≥ 1.5 only (23 events, exploratory) [95% CI] |
|---|---|---|
| Raw data | 0.563 [0.483, 0.641] | 0.698 [0.569, 0.817] |
| Bandpass 1–10 Hz | 0.633 [0.558, 0.708] | 0.811 [0.688, 0.914] |
| f-k filter, 1000 m/s | 0.640 [0.565, 0.714] | 0.691 [0.564, 0.804] |
| seisDAE, released model | 0.601 [0.524, 0.675] | 0.824 [0.718, 0.910] |
| seisDAE trained on DAS | 0.638 [0.561, 0.714] | 0.849 [0.759, 0.920] |
| seisDAE trained on DAS + f-k 250 m/s | 0.671 [0.599, 0.742] | 0.759 [0.653, 0.850] |

No method is a clear winner. The intervals overlap, and the paired differences between the denoising methods below include zero (bandpass and f-k are, however, significantly above the raw data). The M ≥ 1.5 column rests on 23 events.

**Synthetic test:** waveform shape at 0 dB input SNR, as SI-SDR gain in dB (0 = unchanged). Raw 0, bandpass 0.5–45 Hz −0.82, f-k −4.14, released seisDAE −8.51, seisDAE trained on DAS from scratch −4.37 [−7.77, −1.21] (seeds 2027, 2028: −3.60, −4.89), fine-tuned −2.95, −2.41, −2.38 (seeds 2026–2028), cascade −10.29. All seeds: `results/synthetic_seeds_*.csv`.

1. **Released seisDAE does not transfer to DAS waveforms.** Among the seisDAE models and the filters it had the lowest SI-SDR gain at every input SNR; the jDAS versions added later (item 10) are lower still.
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

9. **Released seisDAE on public seismometer data (`src/public_check.py`, `results/public_check_*.csv`, `results/figures/public_check_examples.png`).** Released model, unchanged. Land: IU.ANMO.00.BHZ, 2016–2020, 100 records. Ocean bottom: Cascadia Initiative 7D BHZ, Aug 2013 – Jun 2014, 164 records from 87 events (77 events with two records, 10 with one) at 19 stations. IU.ANMO.00 recorded with a Geotech KS-54000 at 20 samples/s until 9 July 2018 (51 of the 100 land records, which therefore hold nothing above 10 Hz) and with a Streckeisen STS-6A at 40 samples/s afterwards; the 7D BHZ channels are broadband (Trillium Compact or Güralp CMG-3T) at 40–50 samples/s. No instrument response is removed. Events M ≥ 4 at 1–8° (USGS), windows aligned on the iasp91 P time, 100 Hz.

   | | Land (IU.ANMO) | Ocean bottom (7D) |
   |---|---|---|
   | Real-record SNR gain, seisDAE | +14.88 dB [13.20, 16.51], 95/100 improved | +12.22 dB [10.37, 14.03], 155/164 improved |
   | Real-record SNR gain, bandpass 1–10 Hz | +3.58 dB [2.88, 4.28], 87/100 | +2.00 dB [1.43, 2.59], 120/164 |
   | Synthetic SI-SDR gain at 0 dB, seisDAE / bandpass | −0.31 / −0.02 (42 references) | +5.15 / +4.53 (11 references) |
   | seisDAE minus bandpass, synthetic, significant at | +5 dB (+3.02) and +10 dB (+4.77); worse but not significant at −10 dB (−2.38 [−4.53, +0.04]) | +10 dB only (+3.89 [1.57, 6.13]) |

   - The real-record SNR gain also rewards removing pre-event noise, so the synthetic test is shown beside it.
   - Conclusion: on the data type it was built for, the released seisDAE clearly works (large real-record gain; in the synthetic test better than the bandpass at high input SNR, level or worse at low input SNR, e.g. −2.38 dB [−4.53, +0.04] on land at −10 dB). The real-record gain needs the control in item 13. Its failure on DAS is therefore a change of data type, not a fault of the model or of our code.

10. **jDAS (`src/jdas.py`, `results/synthetic_jdas_*.csv`, `results/jdas_*`).** Released network rebuilt layer by layer; `python -m src.jdas check` gives a relative difference of 0 to the released file. Fine-tuned and trained from scratch on train-day windows (self-supervised, 40 epochs, seed 2026); validation loss fell from 0.157 to 0.140 (fine-tuned) and from 0.177 to 0.140 (from scratch).

    | jDAS version | Detection AUC | SI-SDR gain at 0 dB |
    |---|---|---|
    | Released | 0.533 [0.449, 0.614] | −28.72 [−34.1, −22.9] |
    | Released + bandpass 1–10 Hz | 0.571 [0.491, 0.652] | −30.39 |
    | Fine-tuned | 0.557 [0.478, 0.637] | −8.74 [−11.5, −6.2] |
    | Fine-tuned + bandpass 1–10 Hz | 0.604 [0.524, 0.683] | −14.18 |
    | From scratch | 0.544 [0.463, 0.626] | −8.92 [−11.5, −6.6] |
    | From scratch + bandpass 1–10 Hz | 0.618 [0.539, 0.695] | −14.02 |

    - Every jDAS version is below the synthetic-test bandpass and f-k at every input SNR (all paired intervals exclude zero). Without the post-filter, every version is significantly below the detection bandpass; with it, the fine-tuned version still is (−0.030 [−0.061, −0.001]), the others are not significantly different.
    - The 1–10 Hz bandpass after jDAS lowers SI-SDR because the references carry energy outside 1–10 Hz: on clean references (no noise added) a 1–10 Hz bandpass alone scores −3.6 dB.
    - **Diagnosis (`src/jdas_diagnose.py`).**
      - The 96-channel common-mode removal in our preprocessing is not the cause. On 6 test events downloaded again, it removed 1.5% of the earthquake-window energy, and neighbour coherence was 0.70 without it and 0.69 with it. jDAS kept the clean signal equally in both cases (SI-SDR +6.1 and +5.9 dB, no noise added). This repeats the earlier check of 22 November (V4.1).
      - On the 19 clean test references (no noise added), on the scored channels only, the output kept the signal with SI-SDR: DAS-trained seisDAE +7.3 dB, jDAS fine-tuned +0.1 dB, bandpass 1–10 Hz −3.6 dB, f-k 1000 m/s −4.2 dB. The coherence printed in that check (0.38) was computed between scored channels, which are not always neighbours; the 0.70 above, over all 96 channels, is the meaningful value.
      - Reading: jDAS keeps only what neighbouring channels can predict. With earthquake coherence of about 0.7 between neighbours and traffic noise that is itself coherent, it removes part of the earthquake and keeps much of the noise.

11. **Our 2-D U-Net (`src/unet2d.py`, `results/synthetic_unet2d_*.csv`, `results/unet2d_*`, `results/final_*.csv`).**
    - **Choice of version on validation days only** (`results/unet2d_validation.csv`; rule fixed before: highest mean SI-SDR gain): v1 −2.32 dB, v3 −3.29 dB, v2 −3.34 dB. v1 was chosen and then trained with two more seeds. (In this validation table the filters are the detection settings, 1–10 Hz and 1000 m/s, which is why they score far lower there than in the test table.)
    - **Test days, v1, mean over 3 seeds per event:** SI-SDR gain −0.23, −0.57, −1.16, −2.30, −4.32 dB at −10, −5, 0, 5, 10 dB. Paired against the others: see the README table. In short, better than f-k at every level (19/19 events) and than released seisDAE and jDAS; better than retrained seisDAE at +5 and +10 dB; level with the 0.5–45 Hz bandpass at −10 to 0 dB, below it at +5 and +10 dB.
    - **Single seeds** at 0 dB: −1.25, −1.03, −1.19 dB (seeds 2026–2028). Detection AUC 0.581, 0.585, 0.579; each run is significantly below the 1–10 Hz bandpass (by 0.049–0.054).
    - **v2 and v3 on test:** SI-SDR at 0 dB −2.17 and −2.09 dB; AUC 0.575 and 0.586. Both below v1, as on validation.
    - Training: about 15–22 s per epoch on a laptop CPU; early stopping after 12–37 epochs (v1: 31–37, v2: 12, v3: 14). Two runs had one diverging epoch (v3 epoch 8, seed 2027 epoch 32); the saved weights are from the best epoch before it.

12. **U-Net v4 and onset SNR (`src/onset_snr.py`, `results/onset_snr_*.csv`, 27 November).**
    - v4 (v1 with traffic-free targets): first run diverged (bug, see decision log); second run: validation AUC 0.565 and mean SI-SDR −3.86 dB against 0.585 and −2.32 dB for v1, so v1 stays. Test: AUC 0.580; SI-SDR gain −1.99, −2.51, −3.40, −5.08, −7.79 dB.
    - Onset SNR (Steinberg et al., 2025), 73 recorded test earthquakes / 210 noise windows, minus the 1–10 Hz bandpass (17.20 dB): seisDAE released +1.76 [+1.10, +2.48] (58/73 higher), seisDAE fine-tuned s2026 +1.16 [+0.66, +1.70] (49/73), f-k +0.39 (n.s.), U-Net v1 −0.49 to −1.59, jDAS −1.19. AUC: bandpass 0.616, seisDAE released 0.668, fine-tuned 0.680 (differences +0.052 and +0.064, not significant).

The full numbers are in `results/` (`summary.md`, `detection_*.csv`, `detection_retuned_*.csv`, `detection_sensitivity_test.csv`, `variants_*.csv`, `synthetic_seeds_*.csv`). The figures are in `results/roc.png` and `results/figures/`.
