# How to explain the experiment

Start with a sound-check analogy. Turning down a noisy microphone reduces its loudness. It may bring the recording closer to a quiet reference in squared error while leaving every wiggle in the same place. That is exactly what the scalar baseline tests. An earthquake analyst also cares about the wiggles, their timing and their sign.

## From ground motion to a model input

**DAS** uses a laser interrogator to sense changes in backscattered light along a fiber. Its channels measure deformation along the cable. Direction, coupling and gauge length matter. A vertical station instrument at one location measures a different physical quantity. Adjacent DAS channels share a wavefront and often share noise. A channel is a useful spatial sample, not an independent earthquake.

**Aperture and moveout.** An aperture is a contiguous group of channels. Moveout is the change in arrival time along the group. We use published timing shifts and favor a block with relatively clear arrivals. This improves the surrogate reference and biases the sample toward easy arrivals. The released observations already underwent bandpass and common-mode removal.

**Reference and noise.** A late P window contains a real arrival plus unknown background. An early window supplies added noise. One scale per aperture fixes the energy ratio without destroying relative channel amplitudes. The model receives the mixture. Only the experimenter sees the target. A synthetic mixture of two recorded windows has controlled added noise but no perfectly clean truth.

**STFT and mask.** The short-time Fourier transform represents a trace as overlapping windows with complex frequency coefficients. Magnitude measures local strength; phase helps reconstruct timing. An autoencoder compresses features and decodes a mask. Multiplying the noisy complex coefficients by the signal mask suppresses selected bins. The inverse STFT reconstructs a waveform. A mask near one in high-energy bins changes little even if low-energy bins are strongly suppressed.

**Padding and stretching.** The checkpoint needs a much longer input than the measured clip. Zeros preserve the measured center. Reflection and tiling impose artificial continuity. Gaussian extension adds unrelated context. Stretching resamples time and changes apparent frequencies. Its inverse restores output length but cannot create physical observations.

## What each comparator contributes

| Method | What it does | What can go wrong |
|---|---|---|
| Identity | Returns the mixture. | Retains all the added background. |
| Scalar | Multiplies everything by a training-fitted constant. | Improves squared error without improving shape. |
| Bandpass | Accepts a frequency interval. | Refilters already filtered data; short-window boundaries distort the result. |
| Wavelet | Shrinks small detail coefficients. | An estimated noise threshold can remove real transients. |
| f–k fan | Uses temporal frequency and spatial wavenumber to reject some apparent velocities. | DAS noise can be coherent; real wave energy can lie outside the fan. |
| Truncated SVD | Keeps a few dominant spatial–temporal modes. | A strong coherent nuisance can dominate those modes. |
| Median stack | Uses a robust across-channel waveform as a shared estimate. | Different amplitudes or residual moveout can erase genuine differences. |
| Zero-shot | Uses the released station-trained parameters. | Sensor response, clip duration and noise statistics differ. |
| Head | Updates the final linear map to signal/noise mask logits. | A tiny head cannot change the frozen representation. |
| Decoder | Also adapts the final decoder block. | More weights can require different optimization and more independent data. |

## What the scores actually measure

**Ordinary SNR gain** is ten times the base-ten logarithm of input squared residual energy divided by output squared residual energy. Positive is better against this target. **SI-SDR** projects a demeaned estimate onto the target before scoring residual distortion. Its gain is the change from the same mixture. Rescaling and global sign reversal leave it unchanged. **Signed correlation** distinguishes a reversed waveform. **Peak ratio** compares local peak magnitudes; one is the target. **First-excursion sign** compares the direction of local threshold crossings. Noise or a shifted crossing can flip it. It is a proxy, not an analyst first motion.

**Picking.** STA/LTA compares short- and long-window energy. Recovery counts reference triggers also recovered in the output. Timing MAE conditions on matched triggers. A filter can shift which traces enter this denominator. A detection study would also require continuous negative examples and false-alarm rates.

## How to defend an inference

**Event split.** Put every channel and every SNR of one earthquake in one split. A channelwise random split would leak almost the same waveform into training and testing. **Validation** selects settings. **Out-of-fold predictions** score development events using weights fitted and stopped elsewhere. The previously reported test benchmark is a frozen same-fiber cohort, not a new acquisition.

**Bootstrap.** Resample earthquakes within each array and keep all their channels together. Pair competing methods on each earthquake before resampling a difference. **Cluster bootstrap** goes further by resampling time-and-distance groups together. It is sensitive to the chosen grouping and is not a catalog of confirmed aftershock families.

**Read the study by opening the scalar waveform figure first, then the metric comparison, then the magnitude intervals.** Ask which scores improve, by how much, compared with which baseline, and on how many independent recordings. The report appendix links each result to its event table.

## Quick glossary

| Term | Meaning in this study |
|---|---|
| P arrival | First compressional-wave arrival in the supplied event window. |
| Gauge length | Fiber distance over which deformation is measured. |
| Common mode | A component shared across many channels. |
| Checkpoint | Saved neural-network parameters. |
| Logit | A model score before conversion to mask probabilities. |
| SI-SDR | Signal-to-distortion ratio after optimal rescaling of the reference. |
| OOF | Out-of-fold predictions from a model that excluded the scored event. |
| U-Net | Encoder and decoder joined by skip connections that retain local detail. |
| Receptive field | Input samples that can influence a particular output. |
| Cluster | A connected same-site group under the prespecified time and distance thresholds. |
