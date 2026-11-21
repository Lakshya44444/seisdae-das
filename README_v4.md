# V4: seisDAE trained and tested on 60 s Ridgecrest DAS windows

## Why V4 exists

V1–V3 used the Caltech `DASPphaseData` release. That release stores 200 samples (2 s) per channel around each P pick. The analysed snippet is 70 samples (0.7 s). The released seisDAE GR model expects 6,001 samples (60 s, `models/gr_mixed_stft.config`). V3 therefore placed each 0.7 s snippet in the middle of 60 s of zeros.

With the model's STFT (`nperseg = 99`, `nfft = 198`), such a snippet touches only 3 of the 121 time frames. The noise around the signal is not visible to the network, which is what it needs to estimate a mask. The near-identity behaviour reported in V3 (+0.40 dB ordinary, +0.14 dB SI gain at 0 dB) is consistent with this. It is a limit of that dataset, not a test of the method under its intended input.

V4 keeps the method and the honest-metric design of V3, but changes the input data:

- **Data.** Continuous Ridgecrest DAS data from the SCEDC AWS Open Dataset: 250 Hz, hourly SEG-Y files, 23 June to 29 July 2020. Windows are 60 s (6,001 samples at 100 Hz) and contain real DAS noise before and after each earthquake.
- **Training.** seisDAE is trained from scratch on DAS earthquake and DAS noise traces. Training uses the pinned original trainer (`scripts/upstream_train.py`, upstream commit `26c9ba0`) with the published architecture and default parameter file.
- **Separation.** Train, validation and test sets never share a day. The day number since 23 June, modulo 5, decides the set: 0, 1, 2 train; 3 validation; 4 test.
- **Frozen settings.** All tunable settings are chosen on validation days and frozen before the test is run. These are the bandpass corners, the f-k velocity, the f-k velocity in the cascade, and the scalar control.
- **Comparison.** Methods compared: no processing, scalar control, bandpass, f-k, released seisDAE, DAS-trained seisDAE, and DAS-trained seisDAE followed by f-k. The last one is the DAS-specific extension: it adds the spatial coherence that single-trace seisDAE ignores.
- **Two tests.**
  1. Synthetic: recorded strong test earthquakes plus real test-day noise at −10, −5, 0, 5 and 10 dB.
  2. Field: weak recorded test earthquakes denoised as recorded. There is no clean reference for these events, so only SNR and neighbouring-channel coherence are reported.

No V4 numbers exist until the steps below have been run. Every table in `results_v4/summary.md` is generated from CSV files by `src/v4/figures.py`.

## Assumptions checked in step 1

These are the assumptions about the archive, and how each is checked:

| Assumption | How it is checked |
|---|---|
| Byte layout: 3,600-byte file header, then per channel a 240-byte trace header and 900,000 float32 little-endian samples | Reproduces the V3 byte offsets of 24 checksummed traces (`tests/test_v4.py`). Step 1 also checks that the file size divides exactly. |
| Each hour file starts on the UTC hour named in its file name | Step 1 prints the time stored in the trace headers. |
| Row *i* of the SEG-Y file is row *i* of `das_info.csv` | Step 1 prints both counts and the das_info columns. |
| Channel block: rows 400–495 (96 channels) | V3 used rows 400–423. If step 1 shows gaps or bad channels, change `channels.first_row` in `src/v4/config.json`. |

If any check fails, fix `src/v4/config.json` (or the code) before continuing.

## Steps (laptop, CPU only)

Run all commands from the repository root. Python 3.12 is required by the pinned TensorFlow.

**0. Environment (once).** Create a virtual environment, install `requirements.txt`, and check out the original code:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
git clone https://github.com/JanisHe/seisDAE.git vendor/seisDAE
git -C vendor/seisDAE checkout 26c9ba0fd70c532f0fb1a612d816cd32475b5036
```

On Linux or macOS, activate with `source .venv/bin/activate` instead.

**1. Inspect the archive**

```
python -m src.v4.das --inspect
```

**2. Earthquake catalogue (SCEDC FDSN, via ObsPy)**

```
python -m src.v4.catalog
```

**3. Smoke test with a few windows**

```
python -m src.v4.build --download --limit 3
```

**4. Full download (resumable; interrupted runs reuse cached windows)**

```
python -m src.v4.build --download
```

**5. Export the training traces**

```
python -m src.v4.build --export
```

**6. Train.** Set `--count` to the signal-trace count printed by step 5. Run one epoch first to measure the time per epoch, then rerun with the full number of epochs:

```
python scripts/upstream_train.py vendor/seisDAE --signals data/v4/upstream/signal --noise data/v4/upstream/noise --count COUNT --batch 16 --epochs 1 --name das_ridgecrest
```

**7. Copy the trained model into models/ and compare its config with the released model**

```
python -m src.v4.collect vendor/seisDAE --name das_ridgecrest
```

**8. Evaluate.** This also cross-checks the inference adapter against the original `predict`:

```
python -m src.v4.evaluate --upstream vendor/seisDAE
```

**9. Figures and the generated summary**

```
python -m src.v4.figures
```

## Rough download size

This is arithmetic, not a measurement. A window covers 96 channels × 64 s × 250 Hz × 4 bytes, about 6.1 MB. The default config fetches up to 570 earthquake candidates plus up to 296 noise windows, which is at most about 5.3 GB of transfer. The compressed cache under `data/v4/gathers` is smaller. To reduce the download, lower `strong_candidates`, `noise_windows_per_day` or `channels.count`.

## Reporting rules

- Do not change any setting after looking at test results. If a change is needed, rerun selection on validation days, then rerun the test once, and record the change in CHANGELOG.md.
- Report every method at every level, including negative gains. The ordinary gain rewards attenuation; the Scalar row shows by how much. Treat SI gain and CC as the main shape measures.
- The synthetic reference is a recorded strong earthquake. It still contains its own background noise, so the measured gains are bounded by that noise.
- All windows come from one fiber and one 37-day deployment. Generalization to other arrays is not tested.
- The catalogue screens noise windows only for catalogued events. Uncatalogued microearthquakes may remain in the noise windows.

## Files

| Path | Content |
|---|---|
| `src/v4/config.json` | Every V4 setting. |
| `src/v4/das.py` | SEG-Y byte-range reader, preprocessing (common-mode removal, 0.5 Hz high-pass, 250 → 100 Hz), geometry. |
| `src/v4/catalog.py` | SCEDC catalogues used for selection and noise screening. |
| `src/v4/build.py` | Window selection, SNR classes, day splits, export for the original trainer. |
| `src/v4/methods.py` | Bandpass, f-k, seisDAE inference, scores. |
| `src/v4/evaluate.py` | Validation-only selection, synthetic and field tests, bootstrap intervals. |
| `src/v4/figures.py` | Figures in `figures_v4/` and the generated `results_v4/summary.md`. |
| `tests/test_v4.py` | Offline tests: byte offsets, splits, export format, mixing, scale-invariance, f-k, STFT identity. |
