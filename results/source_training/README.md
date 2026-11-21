# Conventional seismometer training runs

Both runs below used the pinned original seisDAE trainer, the published model architecture and CPU execution. The DAS comparison in the report starts from the authors' released GR checkpoint. These small source-training runs demonstrate the training procedure separately.

| Run | Input | Epochs | Final training loss | Final validation loss |
|---|---|---:|---:|---:|
| Upstream example | 24 sampled example signal files and upstream noise | 1 | 0.6707 | 0.6875 |
| STEAD + FDSN pilot | 24 distinct STEAD earthquakes and 11 station-noise candidates | 3 | 0.6564 | 0.6769 |

For the STEAD pilot we took the first 24 eligible earthquakes from the EQTransformer 100-sample STEAD example. Each vertical component exceeded or equaled 20 dB source SNR. Signals were extended by one terminal zero to satisfy the 6,001-sample model. Noise came from IU.ANMO.00.BHZ on 10 January 2023, 00:00–00:12 UTC, at 40 Hz, detrended and resampled to 100 Hz. We inspected all 11 nonoverlapping windows in the 1–12 Hz band for obvious transients. No independent catalog-based certification of noise purity was performed. The original trainer randomly draws background from a common pool for training and validation. Its validation loss is a training diagnostic. The separate DAS benchmark partitions both signal and background by earthquake.

The checkpoint and configuration are in `models/stead_fdsn_pilot.*`. The two CSVs here identify all source traces. The epoch-by-epoch losses, source file identities and figures are retained so that a larger run can replace the pilot without changing the published-checkpoint study.

## Rerun both source experiments

Use a dedicated checkout. The trainer writes a named model, checkpoint and configuration into it. Each invocation reuses `local_conventional`, so copy the outputs you wish to keep before starting another run.

```bash
git clone https://github.com/JanisHe/seisDAE.git vendor/seisDAE
git -C vendor/seisDAE checkout 26c9ba0fd70c532f0fb1a612d816cd32475b5036
python scripts/upstream_train.py vendor/seisDAE --count 24 --batch 2 --epochs 1

mkdir -p data/conventional
curl -fL https://raw.githubusercontent.com/smousavi05/EQTransformer/a589da6504b11df8341fe1dc859a025f49955409/ModelsAndSampleData/100samples.hdf5 -o data/conventional/100samples.hdf5
python scripts/source_data.py stead data/conventional/100samples.hdf5 data/conventional/signal --count 24
python scripts/source_data.py fdsn data/conventional/noise --minutes 12
python scripts/upstream_train.py vendor/seisDAE --signals data/conventional/signal --noise data/conventional/noise --count 24 --batch 2 --epochs 3
```

The STEAD sample SHA-256 is `1ac7024c16f0cd62d4e05c41e9920a8bd5359d10a8c35cd504a3dab419aeec02`. For a larger run, use a local-earthquake HDF5 chunk from the [STEAD release](https://github.com/smousavi05/STEAD), raise `--count`, and use more independently screened station-noise windows. The exporter requires a new or empty output directory. Seeding controls random choices; upstream file enumeration, framework kernels and service changes may affect repeat runs.
