# Model card

The released GR seisDAE model estimates signal and noise masks from normalized complex STFT features. We use the upstream GPL-3.0 checkpoint and preserve its original preprocessing for reproduction. The original adapted head and decoder weights are included. New stretch results apply a deterministic input transformation selected on development folds; there is no separate stretch checkpoint.

The matched-budget experiment compares masks and waveform error on identical development traces. The population head and scratch U-Net are out-of-fold research models, retrained per fold by the scripts. A deployed all-event checkpoint is outside this evaluation. Outputs retain arbitrary field amplitude units and depend on normalization, window length, coupling and noise. Supported use is a reproducible comparison of waveform recovery on the included data. Field alerting and analyst polarity inference have not been validated.
