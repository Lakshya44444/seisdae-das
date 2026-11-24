# Data: the Ridgecrest DAS archive and how this project reads it

All data are public. Nothing has to be downloaded by hand: `src/das.py` reads exactly the bytes it needs.

## Where the data are

| What | Location |
|---|---|
| DAS recordings | `s3://scedc-pds/Ridgecrest_DAS/SEG-Y/hourly/YYYYMMDDHH.segy` (HTTPS: `https://scedc-pds.s3.us-west-2.amazonaws.com/Ridgecrest_DAS/...`) |
| Channel positions | `Ridgecrest_DAS/das_info.csv` (cached in `data/ridgecrest/das_info.csv`) |
| Earthquakes | SCEDC earthquake catalogue, read by `src/catalog.py` (cached in `data/ridgecrest/catalog_*.csv`) |

Provider: Southern California Earthquake Data Center (SCEDC), AWS Open Data. Days used: 23 June – 29 July 2020 (`src/config.json`, section `archive`).

One file holds one hour, all channels. Example: `SEG-Y/hourly/2020070114.segy` is 1 July 2020, 14:00–15:00 UTC.

## Layout of one hourly file

```
3600-byte file header
trace 0:    240-byte trace header | 900 000 samples
trace 1:    240-byte trace header | 900 000 samples
...
trace 1249: 240-byte trace header | 900 000 samples
```

- One trace is one channel; a file holds 1250 channels.
- One trace holds one hour at 250 Hz: 250 × 3600 = 900 000 samples.
- Each sample is a **little-endian** 4-byte float (`<f4`). Standard SEG-Y is big-endian, so a generic SEG-Y reader may decode these files wrongly.
- One trace takes 240 + 4 × 900 000 = 3 600 240 bytes; a full file is about 4.5 GB.

The physical unit of the samples was not checked in this project. All results use relative quantities (SNR, SI-SDR, ROC AUC), so the unit does not affect them.

## Reading only what is needed

Because the layout is fixed, the position of any sample is known:

```
byte = 3600 + row × 3 600 240 + 240 + 4 × sample
```

`src/das.py` (`byte_offset`, `read_window`) sends HTTP range requests (`Range: bytes=a-b`) for the 96 channels and 60 s of each window, so a window costs a few MB instead of a 4.5 GB file. A window that crosses the hour is read from two files.

Two checks guard the layout assumption:

1. Every file's size must equal 3600 + k × 3 600 240 bytes; otherwise the read fails with `MissingData`.
2. `python -m src.das --inspect` prints the trace count, the time stamps in the trace headers and the columns of `das_info.csv`, so the layout can be checked on real files before anything else runs.

## Channel positions (`das_info.csv`)

```
index,latitude,longitude,elevation_m
25,35.6452,-117.667931,0.0
26,35.645272,-117.667929,0.0
...
```

- 1150 rows: only channels with a known position.
- **`index` is the trace (row) number in the SEG-Y file.** The line number in the CSV is not the file row, so rows are always mapped through `index`.
- This project uses file rows 918–1013: 96 neighbouring channels, 7.95 m apart, about 760 m of fiber (`src/config.json`, section `channels`).

## From raw files to the windows used here

1. **Earthquake windows.** SCEDC catalogue events with M ≥ 1.0 within 0.6° of the array. The window is 60 s long, and the predicted P arrival (6 km/s straight ray) sits at 10 s.
2. **Noise windows.** Random start times with no catalogued earthquake nearby.
3. **Preprocessing** (`src/das.py`, `preprocess`): remove the linear trend, subtract the common mode (median over the 96 channels at each sample), 0.5 Hz high-pass (4th-order Butterworth, zero phase), resample 250 → 100 Hz. Each window becomes 96 channels × 6001 samples.
4. **Labelling.** A window counts as a clean earthquake if at least 30% of channels reach 10 dB SNR after an f-k filter that keeps apparent velocities above 1000 m/s (this removes traffic). The data given to the methods stay unfiltered.
5. **Splits by day** (day number modulo 5: 0, 1, 2 train; 3 validation; 4 test).

| Set | Train | Validation | Test |
|---|---|---|---|
| Clean earthquakes (training, synthetic test) | 35 | 6 | 19 |
| Catalogue earthquake windows with no other event overlapping (detection) | 129 | 36 | 73 |
| Noise windows | 132 | 42 | 42 + 168 extra = 210 |

6. **Training traces.** The 35 clean training earthquakes give 1894 single-channel signal traces (`python -m src.build --export`), in the folder layout the original seisDAE trainer expects (`data/ridgecrest/upstream/signal` and `.../noise`).

`data/ridgecrest/manifest.csv` lists every window with its time, kind, split and file.
