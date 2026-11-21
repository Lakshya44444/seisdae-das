# Data card

**Target windows.** Jiaxuan Li’s CC0 Caltech DAS P-phase release contains 181 event files. The bundled prepared archive holds high-source-SNR reference, early background, raw samples and nominal arrivals for a contiguous aperture per event. The second-aperture archive records the first eligible disjoint block after the first, wrapping if necessary. Its manifest gives both starting HDF5 rows. The targets already contain noise and were filtered by the source study.

**Independent background.** SCEDC supplies the three public Ridgecrest hourly source objects. The UTC timestamp, trace rows, byte offsets and SHA-256 digests are in `data/noise_v2/continuous.json` and `data/noise_v3/*/continuous.json`. Every target reuses three fixed cuts from each recording. Samples within a recording are associated. These excerpts were not independently certified event-free.

**Conventional data.** The released RN.BAVN example verifies the upstream adapter. The conventional spectral comparison uses the exact 24 source IDs in `results/source_training/stead_manifest.csv`. Its derived spectra and source hash are shipped; the normal benchmark can run offline. To reconstruct the spectra from the source HDF5, run `python -m src.v3.stead`. The selected STEAD pilot is distinct from the training population of the released GR checkpoint.

**Splits.** Fixed-test experiments, development OOF experiments and the separate all-event cross-fit answer different questions. Never use an all-event fold’s weights to predict the original fixed test and call it external validation. See `results_v3/crossfit_roles.csv`, `results_v2/folds.csv` and `data/prepared/events.csv`.
