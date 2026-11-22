# Results (generated from results/*.csv)

Test earthquakes: 19; test noise windows: 42; weak field events: 16; validation earthquakes: 6.
Frozen settings (validation days): {'Bandpass': [0.5, 45], 'Scalar': 0.8, 'f-k': 250, 'seisDAE (DAS) + f-k': 250, 'Noisy': None}

## Input SNR -10 dB

| Method | Ordinary gain (dB) | SI gain (dB) [95% CI] | SI gain, event window (dB) | CC |
|---|---|---|---|---|
| Noisy | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.407 |
| Scalar | 1.89 | -0.00 [-0.00, 0.00] | 0.00 | 0.407 |
| Bandpass | 0.66 | -0.28 [-0.45, -0.14] | -0.48 | 0.402 |
| f-k | 3.51 | -2.84 [-4.24, -1.65] | -4.14 | 0.327 |
| seisDAE (released) | 6.75 | -5.28 [-9.01, -1.37] | -5.83 | 0.365 |
| seisDAE (DAS-trained) | 8.68 | 0.74 [-1.60, 3.00] | -1.74 | 0.455 |
| seisDAE (DAS) + f-k | 9.60 | -4.78 [-8.83, -0.83] | -7.38 | 0.332 |

## Input SNR -5 dB

| Method | Ordinary gain (dB) | SI gain (dB) [95% CI] | SI gain, event window (dB) | CC |
|---|---|---|---|---|
| Noisy | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.556 |
| Scalar | 1.78 | -0.00 [-0.00, -0.00] | 0.00 | 0.556 |
| Bandpass | 0.56 | -0.50 [-0.80, -0.26] | -0.90 | 0.550 |
| f-k | 2.22 | -3.33 [-4.97, -1.94] | -5.03 | 0.470 |
| seisDAE (released) | 4.97 | -6.28 [-9.80, -2.61] | -8.16 | 0.437 |
| seisDAE (DAS-trained) | 5.44 | -1.62 [-4.46, 1.05] | -4.17 | 0.523 |
| seisDAE (DAS) + f-k | 5.46 | -7.37 [-12.00, -2.91] | -10.38 | 0.394 |

## Input SNR 0 dB

| Method | Ordinary gain (dB) | SI gain (dB) [95% CI] | SI gain, event window (dB) | CC |
|---|---|---|---|---|
| Noisy | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.698 |
| Scalar | 1.48 | -0.00 [-0.00, 0.00] | -0.00 | 0.698 |
| Bandpass | 0.30 | -0.82 [-1.34, -0.44] | -1.60 | 0.691 |
| f-k | 0.02 | -4.14 [-6.02, -2.50] | -6.56 | 0.609 |
| seisDAE (released) | 2.30 | -8.51 [-11.86, -5.01] | -11.21 | 0.493 |
| seisDAE (DAS-trained) | 1.64 | -4.37 [-7.77, -1.21] | -7.00 | 0.581 |
| seisDAE (DAS) + f-k | 1.02 | -10.29 [-15.51, -5.31] | -13.53 | 0.448 |

## Input SNR 5 dB

| Method | Ordinary gain (dB) | SI gain (dB) [95% CI] | SI gain, event window (dB) | CC |
|---|---|---|---|---|
| Noisy | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.806 |
| Scalar | 0.74 | 0.00 [0.00, 0.00] | 0.00 | 0.806 |
| Bandpass | -0.36 | -1.42 [-2.20, -0.81] | -2.76 | 0.798 |
| f-k | -3.09 | -5.54 [-7.74, -3.56] | -8.81 | 0.714 |
| seisDAE (released) | -1.26 | -11.44 [-14.68, -8.07] | -14.51 | 0.536 |
| seisDAE (DAS-trained) | -2.55 | -7.64 [-11.58, -3.85] | -10.28 | 0.622 |
| seisDAE (DAS) + f-k | -3.62 | -13.76 [-19.56, -8.46] | -17.23 | 0.486 |

## Input SNR 10 dB

| Method | Ordinary gain (dB) | SI gain (dB) [95% CI] | SI gain, event window (dB) | CC |
|---|---|---|---|---|
| Noisy | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.876 |
| Scalar | -0.83 | 0.00 [-0.00, 0.00] | 0.00 | 0.876 |
| Bandpass | -1.71 | -2.37 [-3.50, -1.44] | -4.47 | 0.867 |
| f-k | -6.92 | -7.76 [-10.29, -5.41] | -11.77 | 0.779 |
| seisDAE (released) | -5.46 | -14.87 [-18.10, -11.54] | -18.25 | 0.569 |
| seisDAE (DAS-trained) | -7.00 | -11.24 [-15.63, -6.98] | -13.90 | 0.651 |
| seisDAE (DAS) + f-k | -8.39 | -17.74 [-23.87, -12.06] | -21.32 | 0.509 |

## Paired SI-gain differences

| Comparison | Level | Mean (dB) | 95% CI | Wins / events |
|---|---|---|---|---|
| seisDAE (DAS-trained) minus seisDAE (released) | -10 | 6.01 | [3.77, 8.22] | 17/19 |
| seisDAE (DAS-trained) minus seisDAE (released) | -5 | 4.66 | [2.89, 6.30] | 16/19 |
| seisDAE (DAS-trained) minus seisDAE (released) | 0 | 4.14 | [2.42, 5.80] | 15/19 |
| seisDAE (DAS-trained) minus seisDAE (released) | 5 | 3.80 | [1.68, 5.88] | 15/19 |
| seisDAE (DAS-trained) minus seisDAE (released) | 10 | 3.62 | [1.00, 6.14] | 14/19 |
| seisDAE (DAS-trained) minus f-k | -10 | 3.58 | [2.00, 5.21] | 17/19 |
| seisDAE (DAS-trained) minus f-k | -5 | 1.71 | [-0.30, 3.75] | 10/19 |
| seisDAE (DAS-trained) minus f-k | 0 | -0.23 | [-2.79, 2.33] | 10/19 |
| seisDAE (DAS-trained) minus f-k | 5 | -2.10 | [-5.24, 1.04] | 9/19 |
| seisDAE (DAS-trained) minus f-k | 10 | -3.49 | [-7.01, 0.05] | 9/19 |
| seisDAE (DAS) + f-k minus f-k | -10 | -1.94 | [-4.91, 1.10] | 6/19 |
| seisDAE (DAS) + f-k minus f-k | -5 | -4.04 | [-7.51, -0.63] | 7/19 |
| seisDAE (DAS) + f-k minus f-k | 0 | -6.15 | [-10.14, -2.43] | 5/19 |
| seisDAE (DAS) + f-k minus f-k | 5 | -8.22 | [-12.55, -4.16] | 5/19 |
| seisDAE (DAS) + f-k minus f-k | 10 | -9.98 | [-14.47, -5.74] | 4/19 |
| seisDAE (DAS) + f-k minus seisDAE (DAS-trained) | -10 | -5.52 | [-7.62, -3.50] | 2/19 |
| seisDAE (DAS) + f-k minus seisDAE (DAS-trained) | -5 | -5.75 | [-8.05, -3.60] | 1/19 |
| seisDAE (DAS) + f-k minus seisDAE (DAS-trained) | 0 | -5.92 | [-8.26, -3.74] | 0/19 |
| seisDAE (DAS) + f-k minus seisDAE (DAS-trained) | 5 | -6.12 | [-8.44, -4.01] | 0/19 |
| seisDAE (DAS) + f-k minus seisDAE (DAS-trained) | 10 | -6.50 | [-8.87, -4.39] | 0/19 |

## Weak recorded earthquakes (no added noise, no clean reference)

| Method | Median channel SNR (dB) | Channels >= 10 dB | Neighbour coherence |
|---|---|---|---|
| Noisy | 2.92 | 0.17 | 0.217 |
| Bandpass | 2.94 | 0.18 | 0.216 |
| f-k | 4.46 | 0.22 | 0.427 |
| seisDAE (released) | 12.81 | 0.56 | 0.327 |
| seisDAE (DAS-trained) | 9.50 | 0.47 | 0.229 |
| seisDAE (DAS) + f-k | 12.72 | 0.63 | 0.594 |

## Detection test on recorded test-day data (main result)

Earthquake windows: 73; noise windows: 210. Thresholds are set on the noise windows of each method.

| Method | Detected at 5% false alarm [95% CI] | Detected at 10% false alarm [95% CI] | Median ratio, earthquakes (dB) | Median ratio, noise (dB) |
|---|---|---|---|---|
| Noisy | 0.10 [0.04, 0.16] | 0.19 [0.11, 0.29] | 2.36 | 1.23 |
| Bandpass | 0.11 [0.04, 0.19] | 0.16 [0.08, 0.26] | 2.36 | 1.22 |
| f-k | 0.12 [0.05, 0.21] | 0.18 [0.10, 0.27] | 3.03 | 1.27 |
| seisDAE (released) | 0.15 [0.07, 0.23] | 0.26 [0.16, 0.37] | 14.99 | 10.85 |
| seisDAE (DAS-trained) | 0.08 [0.03, 0.15] | 0.23 [0.14, 0.33] | 10.03 | 5.61 |
| seisDAE (DAS) + f-k | 0.03 [0.00, 0.07] | 0.22 [0.14, 0.32] | 11.34 | 4.17 |

Median CPU time per trace (ms): {'Bandpass': 0.1013208335886399, 'f-k': 1.5196531245843896, 'seisDAE (released)': 2.4754333335295087, 'seisDAE (DAS-trained)': 2.4981760416267207}
