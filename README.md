# Denoising DAS data with seisDAE, jDAS and a 2-D U-Net

**Objective.** Use seisDAE (Steinberg et al., 2025; code https://github.com/JanisHe/seisDAE) to denoise distributed acoustic sensing (DAS) data.

**What we did.** We retrained the original seisDAE network on 60 s windows of the public Ridgecrest DAS dataset (California, 2020), both from scratch and by fine-tuning the released model, with three seeds each. After a first review we added (i) a check of the released seisDAE on public land and ocean-bottom seismometer data, (ii) jDAS, a published self-supervised multi-channel DAS denoiser, and (iii) our own supervised 2-D U-Net that sees all channels at once. Everything was compared with standard filters on test days that were not used for training or for choosing settings. The test days were, however, reused across versions of this pipeline (see Limitations).

## Description

DAS turns an optical fiber into thousands of closely spaced strain sensors. Roadside fiber records earthquakes, but also strong traffic noise, noise common to all channels and badly coupled channels. seisDAE is a U-Net denoising autoencoder that works on the time–frequency (STFT) image of one trace and predicts which parts are signal; it was built and trained for seismometer data. This project asks whether it also works on DAS, and tests the answer honestly.

- **Data.** 37 days (23 June – 29 July 2020) of the public Ridgecrest DAS archive (SCEDC, AWS Open Data), 96 neighbouring channels (about 760 m of fiber), 60 s earthquake windows from the SCEDC catalogue and random noise windows. Windows are split by day into train, validation and test days. Only the needed bytes are read from the hourly SEG-Y files with HTTP range requests. File format, channel mapping, labelling and window counts: [`docs/data.md`](docs/data.md).
- **Models.** The released seisDAE model; seisDAE retrained from scratch on DAS; the released model fine-tuned on DAS (both with the original seisDAE code unchanged, three seeds each); jDAS (released, fine-tuned, from scratch); and our supervised 2-D U-Net (three seeds).
- **Baselines.** A bandpass filter and an f-k (apparent-velocity) filter, with settings chosen without test days.
- **Tests.** Waveform shape: SI-SDR on test earthquakes mixed with recorded test noise at −10 to 10 dB. Detection: ROC AUC on 73 recorded test earthquake windows against 210 test noise windows. Traffic removal: how much slow (traffic) and fast (earthquake) f-k energy each method keeps. All comparisons are paired, with 95% bootstrap intervals.
- **Everything is recorded.** Every result table is in `results/`; every decision, including those made after a result was seen, is dated in [`docs/decision_log.md`](docs/decision_log.md); the full write-up is in [`docs/results.md`](docs/results.md).

## TL;DR

- **The released seisDAE works on seismometers, not on DAS.** On public broadband data it raised the SNR of real records by +14.9 dB [13.2, 16.5] on land (IU.ANMO, 100 records) and +12.2 dB [10.4, 14.0] on the ocean bottom (Cascadia 7D, 164 records); a 1–10 Hz bandpass gave +3.6 and +2.0 dB. On DAS the same model distorts waveforms more than any filter (SI-SDR gain −8.5 dB at 0 dB input SNR). So the failure on DAS is a change of data type (strain on a roadside fiber), not a broken model or code.
- **Retraining seisDAE on DAS helps waveform shape, not detection.** All six retrained runs beat the released model in SI-SDR at every input SNR; in detection none is significantly better than the released model or than a tuned bandpass.
- **jDAS fails on this array.** Released, fine-tuned or trained from scratch, it lowered SI-SDR at every input SNR (by 6 to 15 dB after fine-tuning, 26 to 36 dB as released) and its detection AUC (0.53–0.56) is below the bandpass. Measured reason: neighbouring channels correlate only about 0.70 during earthquakes, and the main noise (traffic) is itself coherent, so the assumption behind jDAS does not hold here.
- **Our 2-D U-Net is the best neural model on waveform shape, but not on detection.** Averaged over three seeds it beats the f-k filter at every input SNR (+2.6 to +3.4 dB, 19 of 19 events), the released seisDAE (+5 to +11 dB) and jDAS; above 0 dB it also beats retrained seisDAE. At −10 to 0 dB it is level with the bandpass (differences −0.3 to +0.05 dB, not significant); above that the bandpass is better. Its detection AUC (0.58) is below the bandpass (0.63).
- **No method beat a simple bandpass with a significant margin** on this 760 m urban fiber section, in either test.

## Key numbers

Test days. Detection: ROC AUC, 73 earthquake and 210 noise windows. Synthetic: SI-SDR gain in dB (0 = unchanged input), 19 test earthquakes mixed with recorded test noise; for several seeds the gain is averaged per event over the seeds. Baselines were tuned without test days: for detection a 1–10 Hz bandpass and f-k at 1000 m/s, for the synthetic test a 0.5–45 Hz bandpass and f-k at 250 m/s. Brackets: 95% bootstrap intervals. Built by `python -m src.final_table` from `results/`.

| Method | Detection AUC | SI-SDR gain at −10 dB | at 0 dB | at +10 dB |
|---|---|---|---|---|
| Raw data | 0.563 | 0 | 0 | 0 |
| Bandpass | 0.633 | −0.28 [−0.5, −0.1] | −0.82 [−1.3, −0.4] | −2.37 [−3.5, −1.4] |
| f-k filter | 0.640 | −2.84 [−4.2, −1.6] | −4.14 [−6.0, −2.5] | −7.76 [−10.3, −5.4] |
| seisDAE, released | 0.601 | −5.28 [−9.0, −1.4] | −8.51 [−11.9, −5.0] | −14.87 [−18.1, −11.5] |
| seisDAE, from scratch (3 seeds) | 0.620–0.662 | +0.63 [−1.7, +3.0] | −4.29 [−7.8, −1.1] | −10.90 [−15.1, −6.9] |
| seisDAE, fine-tuned (3 seeds) | 0.604–0.648 | +0.79 [−1.4, +3.1] | −2.58 [−5.1, −0.1] | −8.04 [−10.9, −5.4] |
| jDAS, released | 0.533 | −25.54 [−30.8, −20.1] | −28.72 [−34.1, −22.9] | −35.62 [−41.2, −29.7] |
| jDAS, fine-tuned | 0.557 | −5.94 [−8.0, −4.0] | −8.74 [−11.5, −6.2] | −14.88 [−18.1, −11.9] |
| jDAS, from scratch | 0.544 | −5.75 [−7.6, −4.0] | −8.92 [−11.5, −6.6] | −15.16 [−18.1, −12.4] |
| **2-D U-Net v1, ours (3 seeds)** | 0.579–0.585 | −0.23 [−1.0, +0.5] | −1.16 [−2.3, −0.1] | −4.32 [−6.1, −2.7] |

U-Net v1 minus the others, paired per event (dB; * = 95% interval excludes zero):

| minus | −10 dB | −5 dB | 0 dB | +5 dB | +10 dB |
|---|---|---|---|---|---|
| Bandpass | +0.05 | −0.07 | −0.33 | −0.88* | −1.95* |
| f-k | +2.61* | +2.76* | +2.98* | +3.24* | +3.43* |
| seisDAE released | +5.05* | +5.71* | +7.35* | +9.14* | +10.54* |
| seisDAE fine-tuned | −1.02 | +0.10 | +1.42 | +2.68* | +3.72* |
| jDAS fine-tuned | +5.71* | +6.39* | +7.58* | +9.08* | +10.56* |

Public seismometer check, the "+ bandpass" variants of jDAS, U-Net versions v2 and v3, all intervals and the earlier seisDAE details: [`docs/results.md`](docs/results.md).

**Conclusion.** The released seisDAE is sound on the seismometer data it was built for, but should not be used on DAS data without retraining. Of the three neural approaches we tried on DAS, single-channel (seisDAE), self-supervised multi-channel (jDAS) and supervised multi-channel (our 2-D U-Net), the 2-D U-Net preserves waveforms best and is the only one that matches a tuned bandpass at low SNR, but none beats the bandpass significantly, and in detection none beats it at all. jDAS fails for a measured physical reason (limited channel-to-channel coherence of earthquakes and coherent traffic noise on this short array). On this data set a tuned bandpass remains the method to beat.

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
7. **Models added after review.**
   - **Public seismometer check** (`src/public_check.py`): released seisDAE on IU.ANMO (land, 2016–2020) and Cascadia Initiative 7D ocean-bottom BHZ (2013–2014), M ≥ 4 events at 1–8°, from IRIS/USGS via ObsPy. Real-record SNR gain (0–20 s after the predicted P over 1–9 s before), and a synthetic test with high-SNR records mixed with noise windows of other records.
   - **jDAS** (van den Ende et al., 2023; `src/jdas.py`): the released network, rebuilt layer by layer (output identical to the released file), applied at 50 Hz on 11-channel blocks. Fine-tuned and trained from scratch, self-supervised, on train-day windows only (40 epochs, early stopping on validation days). Also scored with a 1–10 Hz bandpass after it, as in the jDAS example.
   - **2-D U-Net, ours** (`src/unet2d.py`): 3 levels, pooling 2 (channels) × 4 (time), 16–128 filters, 1.1 M parameters. Supervised: random 48-channel × 10.24 s crops of train-day earthquakes plus train-day noise at −10 to +10 dB, target the recorded earthquake, MSE loss, early stopping on validation days. Versions v2 (test-like mixing, 25% noise-only samples) and v3 (v2 predicting the noise) were compared with v1 on validation days only, by a rule fixed beforehand (highest mean SI-SDR gain); v1 was chosen. All versions are reported on test.
8. **Tests.**
   - **Synthetic:** 19 test earthquakes mixed with recorded test-day noise at −10 to 10 dB. Scored with SI-SDR (insensitive to scale) against the recorded earthquake.
   - **Detection (main test):** 73 recorded test-day earthquakes and 210 noise windows, with no mixing.
     - **Score:** each method's output gets one number per window. That number is the median over the 96 channels of the power ratio, in dB, between the window from 0.5 s before to 20 s after sample 1000 (10 s) and the 1–9 s window.
     - **Sample 1000:** in an earthquake window this is the predicted P arrival, from the catalogue origin time and a 6 km/s straight-ray travel time. In a noise window it is the same fixed 10 s mark, with no event.
     - **Known-time test:** this is a known-time detection test, not blind detection.
     - **Bias:** the score rewards any method that removes energy outside the earthquake's frequency band, so it measures filtering as much as denoising.
     - **AUC:** the probability that a random earthquake window scores higher than a random noise window. Bootstrap intervals are computed over windows and over days.
   - **Pictures:** before/after examples are in `results/figures/` (`gather_synthetic.png`, `gather_field.png`, `trace.png`, `examples.png`).

## Limitations

- **Noisy references cap the synthetic test.** The reference earthquakes are recordings and contain their own noise. A method that removes that noise is scored as distorting the signal, which is a likely reason why every method's gain falls as the input SNR rises.
- **Validation set for the U-Net choice is small** (6 earthquakes); the three versions differed by about 1 dB there. U-Net training at learning rate 1e-3 was occasionally unstable (v3 and seed 2027 had one diverging epoch; the saved model is the best one before it).
- **jDAS was run at 50 Hz** (the released model's rate), which removes content above 25 Hz, and with overlapping, cross-faded blocks instead of the original non-overlapping ones. Only one seed each for jDAS fine-tuning and training from scratch.
- **Public check:** one land station; only 11 high-SNR ocean-bottom references; the real-record SNR gain also rewards removing pre-event noise; the comparison bandpass (1–10 Hz) was not tuned.
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
  jdas.py            jDAS (multi-channel, self-supervised): released weights, fine-tuning, same tests
  unet2d.py          our supervised 2-D U-Net (channels x time), trained on train-day mixtures
  jdas_diagnose.py   why jDAS fails here: clean-signal test, coherence, common-mode removal
  public_check.py    released seisDAE on public land and ocean-bottom seismometer data
  denoise_file.py    denoise your own file (miniSEED/SAC/SEG-Y, .npy, .h5) with every method
  paper_figures.py   all figures of the paper (results/figures/paper/)
  final_table.py     final comparison of every method, read from results/*.csv
  quick_check.py     1-2 minute check that the repository works on a new computer
  synthetic_seeds.py the same variants on the synthetic SI-SDR test
  examples.py        before/after pictures of recorded windows
  figures.py         figures and results/summary.md
  collect.py         copy the trained model into models/
  peek.py            quick-look plot of one window
scripts/train_seisdae.py   runs the original seisDAE trainer
models/                    released seisDAE model, trained DAS model, released jDAS model
tests/                     offline tests
results/                   final results
docs/data.md               Ridgecrest data format, channel mapping, windows and splits
docs/results.md            detailed results (all tables and intervals)
docs/decision_log.md       dated record of every decision and result
archive/earlier_attempt/   first attempt on 2 s Caltech DAS snippets (not used)
```

## How to run

There are three levels. Levels 1 and 2 take minutes and need no Ridgecrest download; level 3 repeats the whole study.

| Level | What it shows | Time (CPU laptop) | Download |
|---|---|---|---|
| 1. Quick check | code, tests and all three neural models work; saved results load | 1–2 min | none |
| 2. Public seismometer check | released seisDAE on land (IU.ANMO) and ocean-bottom (7D) broadband data | minutes (land download took about 5 min in our run) | small, from IRIS/EarthScope |
| 3. Full run | every DAS result in this README | many hours | about 3 GB from AWS |

### Setup (once)

Tested on Windows 11, CPU only, Python 3.13.2, TensorFlow 2.20, tf-keras 2.20. A GitHub ZIP or a clone does **not** contain the data (`data/`) or the seisDAE trainer (`vendor/`); both are fetched by the commands below.

Windows (Command Prompt, inside the unzipped folder):
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Linux / macOS:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Run every command from the repository folder (the one that contains `src/`), with the environment active.

### Level 1: quick check

```
python -m src.quick_check
```
Prints PASS or FAIL for: package versions, the offline tests (`tests/`), loading and running the released seisDAE, the DAS-trained seisDAE and the released jDAS (jDAS is also compared with its original file: difference 0), and the main saved results. The last line should read `... of ... checks passed`. On a fresh computer step 5 says the Ridgecrest windows are not downloaded; that is expected.

The tables of this README can be printed again from the saved CSV files, without data:
```
python -m src.variants --stats-only
python -m src.final_table
```

### Level 2: public seismometer check

```
python -m src.public_check download land
python -m src.public_check download obs
python -m src.public_check evaluate
```
Needs internet (IRIS and USGS web services). Events are drawn with a fixed seed, but the archives can change, so counts may differ slightly from `results/public_check_*.csv`.

### Level 3: full run

Clone the seisDAE trainer (only needed to train seisDAE):
```
git clone https://github.com/JanisHe/seisDAE vendor/seisDAE
git -C vendor/seisDAE checkout 26c9ba0
```

Rough cost on the laptop:

| Step | Time | Disk |
|---|---|---|
| Download | several hours (byte-range requests to AWS) | about 3 GB in `data/ridgecrest/` |
| seisDAE training (100 epochs) | 30–80 min (17–47 s per epoch) | |
| Evaluation | about 30 min | |
| `detection_tuning` | 20–40 min | |

`--count` in the training command is the number of signal traces printed by `python -m src.build --export` (1894 in our run).

Data and seisDAE:
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

seisDAE variants (fine-tuned released model, more seeds):
```
python scripts/train_seisdae.py vendor/seisDAE --signals data/ridgecrest/upstream/signal --noise data/ridgecrest/upstream/noise --count 1894 --batch 16 --threads 8 --seed 2026 --retrain-from models/gr_mixed_stft.h5 --name das_finetune_s2026
python -m src.collect vendor/seisDAE --name das_finetune_s2026 --as variants/finetune_s2026
python -m src.variants
python -m src.synthetic_seeds
```

jDAS (needs the steps above, because it adds rows to the same tables):
```
python -m src.jdas check
python -m src.jdas train
python -m src.jdas train --scratch
python -m src.jdas evaluate
```

Our 2-D U-Net (same tests):
```
python -m src.unet2d train
python -m src.unet2d evaluate
```

### Your own data

```
python -m src.denoise_file record.mseed
python -m src.denoise_file gather.npy --fs 250 --dx 8 --seconds 60
```
Reads any ObsPy format (sampling rate from the headers) or a `(channels, time)` array (`.npy`, `.npz`, `.h5`, give `--fs`). Runs the bandpass, both seisDAE models and, with 8 or more channels, the 2-D U-Net (and f-k if `--dx` is given); saves all outputs and a before/after figure next to the input. The DAS models were trained on one Ridgecrest fiber section and may need retraining for another array. For long DAS records use `--start`/`--seconds` to keep memory small.

### If something fails

| Message | Fix |
|---|---|
| `No module named ...` | environment not active, or `pip install -r requirements.txt` not run |
| `No such file ... data/ridgecrest/...` | the command needs level 3 data: run `python -m src.build --download` first |
| `No such file ... vendor/seisDAE` | clone seisDAE as shown under level 3 |
| `FDSNNoDataException` during `public_check download` | normal for a few stations without data; those records are skipped |
| TensorFlow deprecation warnings | harmless |

## References

- Steinberg, A., Gaebler, P., Hartmann, G., Lehr, J., & Pilger, C. (2025). Deep neural networks based denoising of regional seismic waveforms and impact on analysis of North Korean nuclear tests. *Pure and Applied Geophysics*, 182, 4977–4997 (published online 2024). https://doi.org/10.1007/s00024-024-03491-3 Code: https://github.com/JanisHe/seisDAE
- Atterholt, J., et al. (2022). A unified wavefield-partitioning approach for distributed acoustic sensing. *Geophys. J. Int.*, 228, 1410–1418. https://doi.org/10.1093/gji/ggab407
- Lapins, S., et al. (2024). DAS-N2N: machine learning DAS signal denoising without clean data. *Geophys. J. Int.*, 236, 1026–1041. https://doi.org/10.1093/gji/ggad460
- van den Ende, M., Lior, I., Ampuero, J.-P., Sladen, A., Ferrari, A., & Richard, C. (2023). A self-supervised deep learning approach for blind denoising and waveform coherence enhancement in distributed acoustic sensing data. *IEEE Trans. Neural Netw. Learn. Syst.*, 34(7), 3371–3384. https://doi.org/10.1109/TNNLS.2021.3132832 Code: https://github.com/martijnende/jDAS
- Li, Z., Shen, Z., Yang, Y., Williams, E., Wang, X., & Zhan, Z. (2021). Rapid response to the 2019 Ridgecrest earthquake with distributed acoustic sensing. *AGU Advances*, 2, e2021AV000395. https://doi.org/10.1029/2021AV000395
