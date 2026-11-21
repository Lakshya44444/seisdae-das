# Ten-minute talk and examiner prompts

Main talk: ten minutes including pointing and pauses. Backup slides are outside the clock.

## 0:00–0:34 · Can a station denoiser hear earthquakes through fiber?

Open on the real gather. Each row is a different place along the same fiber. Follow the dark and bright patches across the rows. The question is whether a network trained on a different sensor can preserve the first arriving wave. The unit of evidence in this talk is the earthquake. Point to the coherent arrival on the right. Return to this image when explaining why neighboring channels are associated observations. Remember: 145 test events.

**Backup answers**

- **What is the scientific question?** Does a station-trained denoiser recover a recorded fiber P arrival, and which comparisons support that claim?

- **Which part meets the reproduction requirement?** The released station example is run through the local adapter and compared against the upstream predictor.

- **Why would a station model transfer at all?** Both sensors see seismic wave energy with time-frequency structure. Sensor response and noise can still differ.

- **What physical quantity does DAS sense?** Axial deformation along the fiber, with sensitivity dependent on direction and coupling.

- **What result would change your conclusion?** Consistent held-out improvements in shape, amplitude and detection across unseen fibers and independent days would support broader deployment.

## 0:34–1:14 · One cable samples motion at thousands of points.

An interrogator sends laser light down the cable. Changes in the returning light encode deformation along successive fiber sections. A station measures ground motion at one location. The two instruments sense different quantities; nearby fiber channels can share both earthquake energy and noise. Trace the optical path on screen. A bend in the cable changes its directional sensitivity, so similar ground motion can produce different fiber amplitudes. Remember: 24 adjacent channels per selected aperture.

**Backup answers**

- **What physical quantity does DAS sense?** Axial deformation along the fiber, with sensitivity dependent on direction and coupling.

- **Why align adjacent channels?** A coherent arrival moves along the aperture. Correcting timing shifts makes spatial comparisons more meaningful.

- **Why select a high-SNR aperture?** It gives a more interpretable field reference, at the cost of selecting relatively visible signals.

- **Why can a bandpass hurt?** The source is already bandlimited, arrival windows are short and filter boundaries affect the center.

- **Why a median stack?** It is a simple robust estimate of a common arrival; it sacrifices channel-specific waveform structure.

## 1:14–1:54 · Can a station model recover a small arrival?

The assignment has two parts. First reproduce the released example to establish that inference is wired correctly. Then use openly available DAS data to test transfer. We distinguish replication of one prediction from replication of the original paper's entire scientific conclusion. Ask the audience to distinguish two possible successes: reproducing a station prediction, and recovering a previously unseen fiber earthquake. One can succeed while the other fails. Remember: one reproduced station example.

**Backup answers**

- **What is the scientific question?** Does a station-trained denoiser recover a recorded fiber P arrival, and which comparisons support that claim?

- **Did you reproduce the whole paper?** No. The implementation check covers one published example, not the paper’s full population or endpoints.

- **Why would a station model transfer at all?** Both sensors see seismic wave energy with time-frequency structure. Sensor response and noise can still differ.

- **Is the test independent?** Its event IDs are disjoint from fitting, but it shares fibers, may share families, and its earlier results were known.

- **What result would change your conclusion?** Consistent held-out improvements in shape, amplitude and detection across unseen fibers and independent days would support broader deployment.

## 1:54–2:38 · The released station example is reproducible.

These are the measured station input and reconstructed output using the released checkpoint. The archived upstream comparison divided the largest absolute difference by the peak upstream prediction. This checks the adapter, sampling and reconstruction. It covers one trace; it says nothing yet about transfer to fiber. Use the dashed pick to orient the audience. The error measures agreement between two implementations. It does not measure how closely either output matches an unknown clean signal. Remember: one upstream predictor comparison.

**Backup answers**

- **Which part meets the reproduction requirement?** The released station example is run through the local adapter and compared against the upstream predictor.

- **Did you reproduce the whole paper?** No. The implementation check covers one published example, not the paper’s full population or endpoints.

- **What is the STFT?** A sequence of overlapping Fourier transforms that localize frequency content in time.

- **What exactly does the model predict?** Two time-frequency mask probabilities. The signal mask multiplies the noisy complex coefficients.

- **Why keep the complex phase?** It carries timing information necessary to reconstruct a time-domain waveform.

## 2:38–3:30 · Each earthquake supplies one controlled recovery task.

Read this pipeline left to right. The later window is our relatively clear field target. An earlier window supplies background. We scale it once for the whole aperture and add it back. Only observed center samples are scored. The same event never crosses from fitting into testing. Pause at the crop. The model input is long enough for the network architecture; the actual evidence remains the short recorded center. The pilot set is excluded. Remember: 70 recorded samples scored.

**Backup answers**

- **Why align adjacent channels?** A coherent arrival moves along the aperture. Correcting timing shifts makes spatial comparisons more meaningful.

- **Why select a high-SNR aperture?** It gives a more interpretable field reference, at the cost of selecting relatively visible signals.

- **Is your reference clean?** No. It is a recorded P window with residual background. We measure recovery of that window.

- **Why mix with one scale per gather?** It sets aperture energy SNR while retaining relative channel strengths.

- **Why not split by channel?** Closely related channels would reach both training and testing.

## 3:30–4:27 · Turning down the volume earns a gain.

These traces use the same amplitude scale. The fitted multiplier preserves every turning point and every relative time. Ordinary gain improves because squared error rewards attenuation in a noisy mixture. Scale-invariant gain removes that degree of freedom. This is why the rest of the talk uses both scores. Point to a shared peak, then to a shared zero crossing. Neither has moved. A metric that rewards this example needs a second measure before anyone claims better waveform recovery. Remember: zero SI gain for a scalar.

**Backup answers**

- **Why is the scalar better?** It can reduce squared residual energy through attenuation at low input SNR, even though waveform shape is identical.

- **How can SI-SDR miss a sign error?** The optimal projection coefficient can be negative. Signed correlation and first-excursion checks complement it.

- **Why report peak ratio?** Scale invariance discards amplitude information; the peak ratio retains a local amplitude diagnostic.

- **Why distinguish gain from picking?** Residual energy, shape and onset timing answer different scientific questions.

- **Is pick recovery the detection recall?** No. Every input already contains a selected arrival, and the reference is another automatic pick.

## 4:27–5:20 · Changing the metric changes the method ranking.

A circle shows ordinary gain, and a diamond shows scale-invariant gain. Compare how far apart the two markers are for the scalar and the spatial filter. The head gives a small positive improvement on both. The original bandpass loses shape agreement. The development-tuned f–k filter improves both scores and changes the spatial comparison. Follow the f-k row from circle to diamond. It illustrates a genuine physical trade-off. Later, the revised controls change this comparison by changing boundary treatment and the velocity setting. Remember: 0 dB input.

**Backup answers**

- **Why is the scalar better?** It can reduce squared residual energy through attenuation at low input SNR, even though waveform shape is identical.

- **How can SI-SDR miss a sign error?** The optimal projection coefficient can be negative. Signed correlation and first-excursion checks complement it.

- **Why can a bandpass hurt?** The source is already bandlimited, arrival windows are short and filter boundaries affect the center.

- **Why try an f-k filter?** It uses spatial coherence through apparent velocity, which a single-trace network cannot directly exploit.

- **What is an SVD denoiser?** A low-rank approximation retaining dominant modes across channels and time. Strong noise can also be low rank.

## 5:20–6:00 · Shape recovery depends on the starting noise level.

Read the horizontal axis as the input signal-to-noise ratio. The scalar stays at zero scale-invariant gain. F-k behaves differently as the mixture becomes clearer. The adapted head remains near the identity baseline in absolute terms. The complete tables include the other filters and event intervals. Trace one line from left to right. A gain is relative to a different noisy input at each level. A negative value means the filter increased the particular measured distortion. Remember: four input levels.

**Backup answers**

- **Why is the scalar better?** It can reduce squared residual energy through attenuation at low input SNR, even though waveform shape is identical.

- **How can SI-SDR miss a sign error?** The optimal projection coefficient can be negative. Signed correlation and first-excursion checks complement it.

- **Why can a bandpass hurt?** The source is already bandlimited, arrival windows are short and filter boundaries affect the center.

- **Why distinguish gain from picking?** Residual energy, shape and onset timing answer different scientific questions.

- **Does faster picking guarantee better detection?** No. Lower timing error among matched picks can coexist with misses and false alarms.

## 6:00–6:49 · Smaller events gained slightly less from adaptation.

The bars are paired differences between the head and the released network, not total denoising gain. The intervals resample earthquakes within array. Both outer magnitude bins contain only Long Valley events. Source SNR and magnitude can covary, so this is a descriptive comparison within this sample. Read the two extremes first, then the middle group. Whiskers are uncertainties of group means. The direct difference uses its own resampling calculation; visual overlap is not that test. Remember: two extreme Long Valley bins.

**Backup answers**

- **Why select a high-SNR aperture?** It gives a more interpretable field reference, at the cost of selecting relatively visible signals.

- **Why bootstrap by event?** Channels from one earthquake are correlated. Treating them as separate independent samples understates uncertainty.

- **Are your clusters known families?** No. They are operational groups connected by fixed proximity thresholds.

- **Why hold the magnitude question to one comparison?** One prespecified contrast is easier to interpret than many uncorrected post-hoc subgroup comparisons.

- **Do smaller earthquakes benefit more?** The original head helped both extreme bins, but its increment was smaller in the lowest bin. Both extremes are Long Valley only.

## 6:49–7:32 · More trainable weights delivered less gain here.

All three values use identical test mixtures. The larger variant updates the last decoder block as well as the head. Under the original budgets, the smaller adaptation wins. That does not show that extra parameters inherently hurt. Separate the architecture from the optimizer and its number of updates. Say what a fairer budget means: equal numbers of gradient updates on the same traces, plus separate validation decisions. The follow-up experiment is described in the report. Remember: 16 head weights.

**Backup answers**

- **What does the head learn?** A shared linear mapping from frozen final features to the two mask logits.

- **Why did the decoder lose?** It underperformed under the particular training and stopping budgets. The comparison cannot isolate capacity from optimization.

- **Why not train from scratch?** We trained a compact residual U-Net on two apertures per event and scored held-out events in five folds. Its different training population limits comparison with the fixed-test model.

- **Is the test independent?** Its event IDs are disjoint from fitting, but it shares fibers, may share families, and its earlier results were known.

- **What happens inside CV?** Fit on one subset, stop on a separate subset, and predict the held-out outer events.

## 7:32–8:24 · Stretching and background choice change the answer.

Stretching changes apparent frequency. Only development folds chose the factor, separately for each model. The independent-noise panel controls the target earthquake and changes the background source. Interpret conditioning carefully: multiple windows from one recording share its noise regime. Keep the denominators separate. The stretch panel scores controlled same-event mixtures. The external panel combines target earthquakes with separately recorded backgrounds. These are distinct questions. Remember: three external recording hours.

**Backup answers**

- **Does the pre-P window guarantee noise independence?** No. It shares both the acquisition and the event time; the external-noise tests probe this weakness.

- **Why pad so much?** The released neural architecture expects a fixed input length much greater than the real window.

- **Does stretching add information?** No. It changes sampling and the apparent frequencies presented to the network.

- **Can the external-noise result generalize?** Only to the sampled origins and target set. Multiple snippets from one recording share its conditions.

- **What is the decisive next dataset?** Longer recordings on unseen fibers with several independent noise days and analyst-checked arrivals.

## 8:24–9:07 · The mask mostly passes the energetic bins.

The plots show the released signal mask only on frames overlapping the actual observation. The numerical summary weights each bin by the input energy. Large weights near one explain how the prediction can be almost the input on this example. This is a mechanism illustration on one event, not population-wide attribution. Use the color bar: larger values mean the input coefficient passes. Low-energy bins can have striking colors and still contribute very little to the final time-domain signal. Remember: one example aperture.

**Backup answers**

- **What is the STFT?** A sequence of overlapping Fourier transforms that localize frequency content in time.

- **What exactly does the model predict?** Two time-frequency mask probabilities. The signal mask multiplies the noisy complex coefficients.

- **Why keep the complex phase?** It carries timing information necessary to reconstruct a time-domain waveform.

- **What does the head learn?** A shared linear mapping from frozen final features to the two mask logits.

- **Why did the decoder lose?** It underperformed under the particular training and stopping budgets. The comparison cannot isolate capacity from optimization.

## 9:07–9:38 · The evidence stops at these measured windows.

We can defend the measured differences on these mixtures, same-site earthquakes and explicitly specified metrics. We do not have independent analyst labels or continuous-stream false-alarm rates. A deployment claim needs longer data, additional fibers and independent noise days. If asked about deployment, name the missing evaluation: long untouched noise intervals, a false-alarm metric and an independent array or acquisition campaign. Remember: two fibers.

**Backup answers**

- **Is your reference clean?** No. It is a recorded P window with residual background. We measure recovery of that window.

- **Can sign agreement validate a focal mechanism?** No. This automated threshold proxy lacks analyst polarity labels and is sensitive to noise and timing.

- **Is pick recovery the detection recall?** No. Every input already contains a selected arrival, and the reference is another automatic pick.

- **Is the test independent?** Its event IDs are disjoint from fitting, but it shares fibers, may share families, and its earlier results were known.

- **What is the decisive next dataset?** Longer recordings on unseen fibers with several independent noise days and analyst-checked arrivals.

## 9:38–10:00 · Three observations explain the result.

Close by returning to the turning-down-the-volume example. Ask what a model changed, what the chosen metric rewards, and how many independent units support the conclusion. These three questions also explain why larger gains on a new processing condition must be checked on held-out earthquakes. Pause between the three lines. End on the experiment you would run next: a held-out fiber with many independently recorded noise conditions. Remember: three takeaways.

**Backup answers**

- **What is the scientific question?** Does a station-trained denoiser recover a recorded fiber P arrival, and which comparisons support that claim?

- **Did you reproduce the whole paper?** No. The implementation check covers one published example, not the paper’s full population or endpoints.

- **Do smaller earthquakes benefit more?** The original head helped both extreme bins, but its increment was smaller in the lowest bin. Both extremes are Long Valley only.

- **What is the decisive next dataset?** Longer recordings on unseen fibers with several independent noise days and analyst-checked arrivals.

- **What result would change your conclusion?** Consistent held-out improvements in shape, amplitude and detection across unseen fibers and independent days would support broader deployment.

## Backup · allow 30–45 seconds · allow 30–45 seconds · Earthquake IDs define the split.

Point out that the test is disjoint by ID but shares fibers; the historical test results were already known. Remember: six excluded pilot events.

**Backup answers**

- **Why bootstrap by event?** Channels from one earthquake are correlated. Treating them as separate independent samples understates uncertainty.

- **Could the event interval still be optimistic?** Yes. Nearby events may belong to one sequence. Time-distance grouping tests sensitivity to dependence.

- **Is the test independent?** Its event IDs are disjoint from fitting, but it shares fibers, may share families, and its earlier results were known.

- **Why not split by channel?** Closely related channels would reach both training and testing.

- **What happens inside CV?** Fit on one subset, stop on a separate subset, and predict the held-out outer events.

## Backup · allow 30–45 seconds · allow 30–45 seconds · Each metric answers a different question.

A sign reversal preserves SI-SDR. A matched automatic pick is not a manual arrival or a field detection. Remember: one is the target peak ratio.

**Backup answers**

- **Why is the scalar better?** It can reduce squared residual energy through attenuation at low input SNR, even though waveform shape is identical.

- **How can SI-SDR miss a sign error?** The optimal projection coefficient can be negative. Signed correlation and first-excursion checks complement it.

- **Why report peak ratio?** Scale invariance discards amplitude information; the peak ratio retains a local amplitude diagnostic.

- **Can sign agreement validate a focal mechanism?** No. This automated threshold proxy lacks analyst polarity labels and is sensitive to noise and timing.

- **Why distinguish gain from picking?** Residual energy, shape and onset timing answer different scientific questions.

## Backup · allow 30–45 seconds · allow 30–45 seconds · Every development earthquake receives an out-of-fold prediction.

Original development OOF gains at zero dB. Within each outer fold separate fit, stopping and scoring events. Remember: three outer folds.

**Backup answers**

- **What does the head learn?** A shared linear mapping from frozen final features to the two mask logits.

- **Why did the decoder lose?** It underperformed under the particular training and stopping budgets. The comparison cannot isolate capacity from optimization.

- **Why bootstrap by event?** Channels from one earthquake are correlated. Treating them as separate independent samples understates uncertainty.

- **Why not split by channel?** Closely related channels would reach both training and testing.

- **What happens inside CV?** Fit on one subset, stop on a separate subset, and predict the held-out outer events.

## Backup · allow 30–45 seconds · allow 30–45 seconds · The external noise cuts are traceable.

Open the manifest for the exact samples, UTC times and checksums. Snippets from the same source hour are associated observations. Remember: 24 trace rows.

**Backup answers**

- **Is your reference clean?** No. It is a recorded P window with residual background. We measure recovery of that window.

- **Why mix with one scale per gather?** It sets aperture energy SNR while retaining relative channel strengths.

- **Does the pre-P window guarantee noise independence?** No. It shares both the acquisition and the event time; the external-noise tests probe this weakness.

- **Can the external-noise result generalize?** Only to the sampled origins and target set. Multiple snippets from one recording share its conditions.

- **What is the decisive next dataset?** Longer recordings on unseen fibers with several independent noise days and analyst-checked arrivals.

## Backup · allow 30–45 seconds · allow 30–45 seconds · One command rebuilds the study.

The build regenerates the study and checks traceable claims against CSV-backed sources. Read the README for system dependencies. Remember: one reproduction command.

**Backup answers**

- **Which part meets the reproduction requirement?** The released station example is run through the local adapter and compared against the upstream predictor.

- **Did you reproduce the whole paper?** No. The implementation check covers one published example, not the paper’s full population or endpoints.

- **Could the event interval still be optimistic?** Yes. Nearby events may belong to one sequence. Time-distance grouping tests sensitivity to dependence.

- **Is the test independent?** Its event IDs are disjoint from fitting, but it shares fibers, may share families, and its earlier results were known.

- **What happens inside CV?** Fit on one subset, stop on a separate subset, and predict the held-out outer events.

## Backup · allow 30–45 seconds · allow 30–45 seconds · The study builds on published open data.

The full report bibliography gives the source DOIs, data record, STEAD credit and the journal publication dates. Remember: one Caltech P-phase dataset.

**Backup answers**

- **Which part meets the reproduction requirement?** The released station example is run through the local adapter and compared against the upstream predictor.

- **Did you reproduce the whole paper?** No. The implementation check covers one published example, not the paper’s full population or endpoints.

- **Why would a station model transfer at all?** Both sensors see seismic wave energy with time-frequency structure. Sensor response and noise can still differ.

- **What physical quantity does DAS sense?** Axial deformation along the fiber, with sensitivity dependent on direction and coupling.

- **Why not train from scratch?** We trained a compact residual U-Net on two apertures per event and scored held-out events in five folds. Its different training population limits comparison with the fixed-test model.
