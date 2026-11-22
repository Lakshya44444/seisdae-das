# Verification record

The expanded benchmark was run on Linux CPU with Python 3.12, NumPy 1.26.4, SciPy 1.14.1, TensorFlow CPU 2.16.2 and tf-keras 2.16.0.

| Check | Measured outcome |
|---|---|
| Download and preparation | The CaltechDATA tar matched published MD5 `c50468d83401790c9e78f0f21d0798fe`. All 181 catalog IDs produced valid 24-channel apertures. The arrays for the original 36 events matched the earlier package sample for sample. |
| Frozen checkpoint | The GR model SHA-256 matched the pinned upstream artifact. The current RN.BAVN comparison yielded a maximum absolute error of 3.25 × 10⁻⁷ or less relative to peak upstream output. See `reproduction.json`. |
| Independent refit | Training on the 24 original earthquakes reproduced every stored final-layer weight exactly. The selected epoch was 40. The largest absolute loss difference across all epochs was below 10⁻¹⁴. |
| Full rerun | The seven notebooks passed format validation and their code cells ran in sequence. An independent evaluation using the notebook-refitted weights reproduced all 4,060 event/method/SNR rows, 28 pooled rows, 56 array-specific rows, 12 paired comparisons and the saved example waveforms within the verifier tolerances. |
| Independent metric calculation | For the plotted test event at 0 dB, gain computed from vector norms, CC computed with per-channel correlation matrices and RMSE computed from residual samples matched the CSV for all seven methods. |
| Original field input | Direct HDF5 input and the bundled excerpt for event 71123139 returned identical observed arrays and identical estimates for both checkpoints. No artificial noise was added. |
| Regression tests | Seven tests passed, covering split membership, all four target SNRs, filtering, STFT reconstruction, source-checkpoint identity, training-only scalar fitting and event-level aggregation. |

To repeat the numerical comparison, run `python -m src.evaluate --out runs/check`, then `python -m src.verify runs/check`. The verifier permits relative and absolute tolerances of 10⁻⁵ for floating-point outputs. Execution times depend on hardware and system load.
