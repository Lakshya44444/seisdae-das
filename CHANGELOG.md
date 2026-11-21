# Study revision log

The released benchmark is retained under `results/`. Commands below run from the repository root.

- **Baseline.** `python -m unittest discover -s tests -v` passed all seven original tests. `python -m src.verify ../../seisdae-das/runs/notebook_test` compared the earlier rerun against the packaged event, summary, site, paired and waveform outputs.
- **T1.** `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/*.ipynb` encountered a kernel socket permission error in this environment. `python scripts/execute_local.py` executes the same cells with IPython in process. Notebook 07 saves method waveforms to a transient directory for the additional metrics.
- **T2.** `python -m src.v2.stats` computes event bootstrap intervals within each site and stratum and a direct small-minus-large contrast. Bin boundaries use metadata and avoid sparse pooled bins.
- **T3.** `python -m src.v2.baseline` evaluates the seven original outputs against the same field reference, adding SI-SDR gain, arrival peak ratio and first-excursion agreement.
- **T4.** `python -m src.v2.report` generates Markdown from CSVs, renders figures, exports Word with Pandoc and renders it as PDF with LibreOffice. The initial Pandoc deck was replaced with an editable ten-slide python-pptx deck in the final rendering.
- **T5.** `python -m src.v2.training` reproduces the shipped head and fits the last decoder block. It uses three array-balanced outer folds, separate inner stopping events and epoch-zero eligibility. `python -m src.v2.final` scores the locked decoder once on the original test split. The benchmark test set had already been reported in V1. No new choice uses its metrics.
- **T6.** `python -m src.v2.sensitivity` compares zero, reflect, Gaussian noise, periodic tile and fourfold stretch on the six original validation events. The released and shipped-head checks use a shared forward pass.
- **T7.** `python -m src.v2.noise_data --raw /tmp/v2raw/CaltechDATA` extracts disjoint channel blocks. `python -m src.v2.noise_data --continuous` fetched the documented byte ranges. The sensitivity run tests same-array event swaps, other blocks, and a separate Ridgecrest recording; its matched Ridgecrest table uses the same three events throughout.
- **T8.** `python -m src.v2.training --sizes` fits the head on balanced nested subsets. The five seeds vary both training subsamples and optimizer order, including the full-size fits. The sensitivity command also evaluates centered nested aperture widths and recalibrates input SNR for each width.
- **T9.** The same sensitivity run evaluates both compositions of f-k and the fixed head on identical validation mixtures.
- **T10.** `python -m src.v2.failures` ranks best/worst held-out events for each method at the prespecified 0 dB reference level; tie flags and profile summaries accompany the plot.
- **T11.** A new from-scratch STEAD/FDSN experiment and full-network tuning were optional and were not run. The original pilot and scripts are preserved.

Run verification: `bash run_all.sh` completed with all 16 tests passing, 4,060 original event rows and original example waveforms matching, and all 26 V2 CSVs matching the independent rerun. The 10-slide PowerPoint and eight-page PDF were rendered for inspection. The viva sheet rendered on one page.

<!-- results:start -->

## Measured outputs

| Task | Measured result |
|---|---|
| T1 | 7 notebooks, 19 executed code cells, 17 saved output blocks. |
| T2 | Magnitude counts 26, 70, 49; small-minus-large -0.087 dB, 95% CI [-0.144, -0.038]. |
| T3 | Head SI gain 0.135 dB at 0 dB on 145 test events; scalar gain is zero on this scale. |
| T4 | Markdown, Word, PDF and an editable ten-slide presentation. |
| T5 | OOF head/decoder ordinary gains 0.520/0.293 dB. Test decoder minus head -0.272 [-0.294, -0.252] dB. |
| T6 | Validation head zero vs reflect ordinary gain 0.928/2.158 dB and SI gain 0.612/-1.168 dB. |
| T7 | Matched Ridgecrest original/external head SI gain 0.553/0.032 dB. |
| T8 | Five-seed mean head gains from 0.300 to 0.928 dB across training sizes. |
| T9 | f-k → head gain/SI gain 2.376/-3.434 dB on validation. |
| T10 | 160 ranked rows, 8 methods. |
| T11 | Optional larger from-scratch run not performed. |

All gains quoted in this table refer to 0 dB input. Run `bash run_all.sh` for the full sequence. `python -m src.verify runs/reproduced_v2 --v2` compares the independent rerun's scientific values to the packaged tables with absolute/relative tolerance 0.0002. Legacy outputs use 0.00001; elapsed CPU times are excluded.

<!-- results:end -->

# V3 course release

Commands run from the repository root, using the pinned CPU environment and seed 2026. The original numerical release remains in results/ and results_v2/. Historical documents are archived.

| Task | Reproduction command | Observed result or decision |
|---|---|---|
| Baseline | `python -m unittest discover -s tests -v`; `python -m src.verify runs/notebook_test`; `python -m src.verify runs/reproduced_v2 --v2` | The 16 original tests passed; the earlier independent rerun matched 4,060 event rows, example waveforms and 26 V2 tables. |
| A1 | `python -m src.v3.experiments`; `python -m src.v3.boundary` | Padded/tuned f-k: 1.380 dB ordinary and 0.760 dB SI gain on the 145 fixed test events at 0 dB. Includes median and SVD controls. |
| A2 | `python -m src.v3.experiments` | Perfect and sign-flipped signals hit the 120 dB finite SI ceiling. The 78-cluster head-minus-released ordinary CI was 0.294–0.347 dB. |
| A3 | `python -m src.v3.stretch` | Three development folds selected factors 2 (head) and 4 (released); respective held-out SI gains were 0.269 and 0.256 dB at 0 dB. |
| A4 | `python -m src.v3.multi_data`; `python -m src.v3.crossfit` | Five outer folds, 181 earthquakes, two apertures; the full-size OOF head gave 0.132 / 0.094 dB ordinary / SI gain at 0 dB. |
| A5 | `python -m src.v3.objective` | 144 matched updates for 16 or 608 parameters. Decoder NMSE gave 0.281 / 0.251 dB on development OOF predictions at 0 dB. |
| A6 | `python -m src.v3.noise --fetch`; `python -m src.v3.noise` | Three SCEDC source hours, three draws each, twelve Ridgecrest targets. Mean external head SI gain was 0.159 dB; matched original was 0.151 dB. |
| A7 | `python -m src.v3.mechanism`; `python -m src.v3.stead` | Original energy-weighted mask 0.997 on the example. Compared DAS spectra with the station example and the 24-trace STEAD pilot. |
| A8 | `python -m src.v3.sensitivity` | High-source-SNR subset, 23 events: head SI gain 0.256 dB. Stack proxy and alternate early window also reported by site. |
| A9 | `python -m src.v3.scratch_model` | Scratch U-Net, 1,681 parameters, 200 updates per fold. OOF ordinary gain 2.820 dB at 0 dB; SI gain 0.100 dB with interval crossing zero. Larger conventional retraining remains future work. |
| Report, talk, study aids | `make assets`; `make check` | One final report in three formats, 14 main slides, six backups, notes and timed script, 40 viva questions and reviewer critique. Instructor was not supplied. |
| Notebooks | `jupyter nbconvert --to notebook --execute --inplace notebooks/0[89]*.ipynb notebooks/1[012]*.ipynb` or `python scripts/execute_local.py notebooks/0[89]*.ipynb notebooks/1[012]*.ipynb` | Standard kernel startup was denied permission to open a socket. All five new notebooks executed in-process with saved outputs. |
| Checks | `make test check`; `python scripts/verify_v3.py ACTUAL EXPECTED` | All 22 tests passed. Checked the source-derived README, viva, report, actual slide text, speaker notes, script and embedded images. |

## Final Windows source release — 10 November 2026

The previous V3 archive ended at `src/v3/mechanism.py`. Its preserved tables and documents provided a reference for reconstruction. This release supplies 17 V3 implementation modules, a package initializer, the experiment configuration, document templates and regression tests. Historical entries above describe the earlier run.

| Check or change | Command | Observed result |
|---|---|---|
| Original benchmark | `bash run_v2.sh` | Re-executed the seven original notebooks. Verified 4,060 event-method-SNR rows, example waveforms and 26 V2 tables. |
| Original station prediction | `python -m src.reproduce --upstream PATH_TO_PINNED_SEISDAE` | Measured relative peak discrepancy of 3.2423567e-07 against upstream commit 26c9ba0. |
| Reconstructed V3 experiments | `python -m src.v3.stretch`, `noise`, `sensitivity`, `objective`, `scratch_model` (same module prefix) | All were executed. Comparing all 31 V3 CSVs with the prior release, 27 matched within 2e-4 relative and absolute tolerance after sorting by keys. The four changed tables are event and summary CSVs for the two newly trained experiments. |
| Matched-budget objective training | `python -m src.v3.objective` | Decoder waveform NMSE gave 0.280 dB ordinary and 0.251 dB SI gain in development out-of-fold predictions at 0 dB. |
| Scratch U-Net | `python -m src.v3.scratch_model` | The refit gave 3.333 dB ordinary and 0.729 dB SI gain at 0 dB on 181 out-of-fold events. The SI interval was [0.619, 0.829] dB. Architecture, seeds, RMS normalization, sampling and optimizer settings are explicit in source. The earlier run reported 2.820 and 0.100 dB. Use the refit values with this implementation. |
| Tests and publications | `make test assets check` | 24 tests passed. Summary checks recompute six bootstrap tables and the independent-noise means. Report, slide text, speaker notes, embedded figures, viva and talk script are checked against generated content. |
| Release packaging | `python scripts/package_release.py --output ../seisdae-das-final.zip` | The builder checks the V3 inventory, Windows paths, ZIP integrity and extracted SHA-256 digests. It excludes caches, downloads, nested ZIPs and redundant build outputs. |
| Fresh scientific rerun | `SEISDAE_JOBS=4 make reproduce`; after an interrupted session the remaining V2 and V3 module invocations were resumed in the same fresh working copy. `python -m src.verify NEW/results_v3 --v3 --expected RELEASE/results_v3` | All 31 V3 CSV tables and both numerical NPZ archives matched the released values. The pinned STEAD reference spectrum was separately rebuilt from its verified source. Both training experiments were independently refitted. All twelve notebooks executed successfully. |
| Original data extraction | `python -m src.v3.multi_data`; `python -m src.data --raw data/raw/CaltechDATA --out runs/prepared-from-source`; `python -m src.v3.noise --fetch`; `python -m src.v3.stead` | Source TAR verified by its published MD5. Both prepared-aperture NPZs and both primary metadata tables matched. Three downloaded SCEDC cuts reproduced the packaged arrays exactly. Regenerated STEAD spectrum CSV matched byte-for-byte. |
| Delivered copy | `python scripts/package_release.py --check`; `make test check smoke` | The downloaded ZIP matched the built ZIP byte-for-byte. Its extracted contents passed the package check, all 24 tests, the document consistency check and the smoke check on Linux. Native Windows execution, a Docker build and hosted GitHub Actions were not exercised. |

## V4 pipeline (code only, not yet run) — 21 November 2026

| Change | Reason |
|---|---|
| New `src/v4/` pipeline on continuous Ridgecrest DAS (SCEDC AWS), 60 s / 6,001-sample windows, day-separated train/validation/test | The V1–V3 input was a 0.7 s snippet zero-padded to 60 s; it touches 3 of 121 STFT frames of the released model. |
| seisDAE trained from scratch on DAS signal and noise with the pinned original trainer | V3 adapted at most 608 parameters of a station model. |
| `scripts/upstream_train.py --name` | Lets the DAS model be saved under its own name. |
| `tests/test_v4.py` (11 offline tests) | Verifies byte offsets against V3 downloads, splits, export format, mixing, SI invariance, f-k and STFT identity. |

No V4 results are reported here. Run the steps in README_v4.md; `results_v4/summary.md` is generated from the CSV files.

### V4 download size — 21 November 2026

During the first full download, candidates ranked 200–237 (M1.0–1.9) were almost all below the selection SNR. To save time, `strong_candidates` was lowered from 450 to 240 and `noise_windows_per_day` from 8 to 6. Both are data-volume settings. No validation or test score had been computed when they were changed.
