"""Read 60 s windows from the public Ridgecrest DAS archive (SCEDC, AWS Open Data).

File layout assumed here (it reproduces the byte offsets stored in
data/noise_v2/continuous.json, which were downloaded and checksummed in V3):

    3600-byte SEG-Y file header
    for every channel: 240-byte trace header + 900000 little-endian float32 samples
                       (one hour at 250 Hz)

Run `python -m src.v4.das --inspect` once before anything else. It prints the
trace count, the time stamps written in the trace headers and the columns of
das_info.csv, so the assumptions above can be checked on real files.
"""

import argparse
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((Path(__file__).with_name("config.json")).read_text())
DATA = ROOT / "data" / "v4"

FILE_HEADER = 3600
TRACE_HEADER = 240
RAW_HZ = CFG["archive"]["raw_hz"]
PER_HOUR = CFG["archive"]["samples_per_hour"]
TRACE_BYTES = TRACE_HEADER + 4 * PER_HOUR


class MissingData(IOError):
    """Hour file absent or too short for the requested rows."""


def hour_url(hour):
    return f"{CFG['archive']['base_url']}/SEG-Y/hourly/{hour:%Y%m%d%H}.segy"


def byte_offset(row, sample):
    """First byte of `sample` (0-based) on trace `row` (0-based)."""
    return FILE_HEADER + row * TRACE_BYTES + TRACE_HEADER + 4 * sample


def split_by_hour(start, n_raw):
    """Split [start, start + n_raw samples) into (hour, first_sample, count) pieces."""
    start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
    hour = start.floor("h")
    first = int(round((start - hour).total_seconds() * RAW_HZ))
    pieces = []
    while n_raw > 0:
        count = min(n_raw, PER_HOUR - first)
        pieces.append((hour, first, count))
        n_raw -= count
        hour += pd.Timedelta(hours=1)
        first = 0
    return pieces


_SESSION = requests.Session()
_SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=64))
_SIZES = {}


def _get(url, first, last, retries=4):
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, headers={"Range": f"bytes={first}-{last}"}, timeout=60)
            if r.status_code in (403, 404):
                raise MissingData(f"{url} returned {r.status_code}")
            if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {first}-{last}/"):
                raise IOError(f"Byte range not honoured for {url}")
            if len(r.content) != last - first + 1:
                raise IOError("Short read")
            return r.content, r.headers["Content-Range"]
        except MissingData:
            raise
        except (IOError, requests.RequestException):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def n_traces(hour):
    """Number of channels in an hour file, from its total size."""
    url = hour_url(hour)
    if url not in _SIZES:
        _, content_range = _get(url, 0, 0)
        total = int(content_range.rsplit("/", 1)[1])
        if (total - FILE_HEADER) % TRACE_BYTES:
            raise MissingData(f"{url}: size {total} is not 3600 + k*{TRACE_BYTES}")
        _SIZES[url] = (total - FILE_HEADER) // TRACE_BYTES
    return _SIZES[url]


def read_raw(start, seconds, rows, workers=None):
    """Raw 250 Hz samples, shape (len(rows), seconds*250), may span two hour files."""
    n_raw = int(round(seconds * RAW_HZ))
    workers = workers or CFG["archive"].get("download_workers", 8)
    parts = []
    for hour, first, count in split_by_hour(start, n_raw):
        if n_traces(hour) < max(rows) + 1:
            raise MissingData(f"{hour_url(hour)} has only {n_traces(hour)} channels")
        url = hour_url(hour)

        def one(row):
            a = byte_offset(row, first)
            content, _ = _get(url, a, a + 4 * count - 1)
            return np.frombuffer(content, dtype="<f4")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts.append(np.stack(list(pool.map(one, rows))))
    return np.concatenate(parts, axis=1).astype(np.float64)


def preprocess(raw):
    """Detrend, remove the common mode, high-pass, resample 250 -> 100 Hz.

    No band-limiting beyond the 0.5 Hz high-pass and the anti-alias filter of
    resample_poly: noise suppression is left to the methods being compared.
    """
    w = CFG["window"]
    x = detrend(raw, axis=-1)
    x = x - np.median(x, axis=0, keepdims=True)
    sos = butter(4, w["highpass_hz"], fs=RAW_HZ, btype="highpass", output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    return resample_poly(x, 2, 5, axis=-1)


def read_window(start, rows):
    """Preprocessed 100 Hz gather of 6001 samples beginning at `start`."""
    w = CFG["window"]
    pad = w["pad_s"]
    seconds = (w["samples"] - 1) / w["fs"] + 2 * pad + 0.04
    x = preprocess(read_raw(pd.Timestamp(start) - pd.Timedelta(seconds=pad), seconds, rows))
    a = int(round(pad * w["fs"]))
    y = x[:, a:a + w["samples"]]
    if y.shape[1] != w["samples"] or not np.isfinite(y).all():
        raise MissingData("Incomplete or non-finite window")
    return y.astype(np.float32)


def block_rows():
    c = CFG["channels"]
    return list(range(c["first_row"], c["first_row"] + c["count"]))


def das_info():
    path = DATA / "das_info.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = _SESSION.get(f"{CFG['archive']['base_url']}/das_info.csv", timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    return pd.read_csv(path)


def _coordinate_columns(info):
    c = CFG["channels"]
    lat = c["info_lat_column"] or next((k for k in info.columns if "lat" in k.lower()), None)
    lon = c["info_lon_column"] or next((k for k in info.columns if "lon" in k.lower()), None)
    if lat is None or lon is None:
        raise KeyError(f"Set info_lat_column/info_lon_column in config.json. Columns: {list(info.columns)}")
    return lat, lon


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def located_runs(index_values, min_length=1):
    """Runs of consecutive file rows that have coordinates: (first_row, last_row, length)."""
    rows = np.unique(np.asarray(index_values, dtype=int))
    breaks = np.flatnonzero(np.diff(rows) != 1)
    starts, ends = np.r_[rows[0], rows[breaks + 1]], np.r_[rows[breaks], rows[-1]]
    return [(int(a), int(b), int(b - a + 1)) for a, b in zip(starts, ends) if b - a + 1 >= min_length]


def geometry():
    """Block centre (lat, lon) and channel spacing in metres.

    das_info.csv lists only located channels; its index column gives the
    SEG-Y trace row of each one. Every row of the block must be located.
    """
    info = das_info()
    lat, lon = _coordinate_columns(info)
    rows = block_rows()
    b = info.set_index(CFG["channels"]["info_index_column"]).reindex(rows)
    missing = [r for r, ok in zip(rows, b[lat].notna()) if not ok]
    if missing:
        raise KeyError(f"File rows without coordinates in das_info.csv: {missing[:10]}"
                       f"{' ...' if len(missing) > 10 else ''}. Choose channels.first_row inside a located run.")
    step = haversine_km(b[lat].values[:-1], b[lon].values[:-1], b[lat].values[1:], b[lon].values[1:]) * 1000
    spacing = CFG["channels"]["spacing_m"] or float(np.median(step))
    return float(b[lat].mean()), float(b[lon].mean()), spacing


def _trace_time(header):
    """SEG-Y rev1 trace header bytes 157-166: year, day, hour, minute, second."""
    out = {}
    for name, fmt in (("little", "<5h"), ("big", ">5h")):
        out[name] = struct.unpack(fmt, header[156:166])
    return out


def inspect(day):
    hour = pd.Timestamp(day, tz="UTC")
    url = hour_url(hour)
    print("file:", url)
    print("channels from file size:", n_traces(hour))
    for row in (0, CFG["channels"]["first_row"]):
        a = FILE_HEADER + row * TRACE_BYTES
        header, _ = _get(url, a, a + TRACE_HEADER - 1)
        print(f"row {row} header time (year, day, hour, min, sec):", _trace_time(header))
    info = das_info()
    column = CFG["channels"]["info_index_column"]
    print("das_info.csv rows:", len(info), "columns:", list(info.columns))
    print(f"located file rows ({column}): {info[column].min()} to {info[column].max()}")
    runs = located_runs(info[column], CFG["channels"]["count"])
    print(f"runs of >= {CFG['channels']['count']} consecutive located rows (first, last, length):", runs)
    try:
        print("block rows", block_rows()[0], "-", block_rows()[-1], "centre and spacing:", geometry())
    except KeyError as error:
        print(error)
    x = read_window(hour + pd.Timedelta(minutes=10), block_rows())
    print("window shape:", x.shape, "rms:", float(np.sqrt(np.mean(x ** 2))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--hour", default="2020-06-24T17:00")
    a = p.parse_args()
    if a.inspect:
        inspect(a.hour)
