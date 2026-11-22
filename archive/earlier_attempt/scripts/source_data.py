"""Prepare conventional seismometer files for the optional source-training run."""

import argparse
import re
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly


def conventional_signals(hdf, folder, count):
    folder.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    with h5py.File(hdf) as file:
        g = file["data"]
        for name in g:
            d = g[name]
            if d.shape != (6000, 3) or not name.endswith("_EV"):
                continue
            snr = d.attrs.get("snr_db", [])
            if isinstance(snr, (str, bytes)):
                snr = re.findall(r"-?\d+(?:\.\d+)?", str(snr))
            snr = np.asarray(snr, dtype=float).ravel()
            if len(snr) != 3 or not np.isfinite(snr[2]) or snr[2] < 20:
                continue
            # Keep one vertical trace per source earthquake.
            eid = d.attrs.get("source_id", d.attrs.get("source_origin_time"))
            if eid is None or str(eid) in seen:
                continue
            z = d[:, 2].astype("float32")
            if not np.isfinite(z).all() or z.std() == 0:
                continue
            seen.add(str(eid))
            filename = f"stead_{len(rows):06d}.npz"
            np.savez(
                folder / filename,
                data=np.pad(z, (0, 1)),
                itp=int(d.attrs["p_arrival_sample"]),
                its=int(d.attrs["s_arrival_sample"]),
                starttime=str(d.attrs.get("trace_start_time", "")),
            )
            rows.append((filename, name, str(eid), float(snr[2])))
            if len(rows) == count:
                break
    if not rows:
        raise ValueError(
            "No eligible STEAD earthquakes with vertical-component SNR >=20 dB."
        )
    pd.DataFrame(
        rows, columns=["file", "trace_name", "source_id", "vertical_snr_db"]
    ).to_csv(folder / "manifest.csv", index=False)
    print(f"Exported {len(rows)} earthquakes.")


def station_noise(folder, station_id, start, minutes, service):
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client

    folder.mkdir(parents=True, exist_ok=True)
    t = UTCDateTime(start)
    net, sta, loc, cha = station_id.split(".")
    stream = Client(service, timeout=90).get_waveforms(
        net, sta, loc, cha, t, t + minutes * 60
    )
    rows = []
    for tr in stream.split():  # Process each continuous segment separately.
        tr.detrend("linear")
        ratio = Fraction(100 / tr.stats.sampling_rate).limit_denominator(10000)
        a = resample_poly(tr.data, ratio.numerator, ratio.denominator)
        for j in range(0, len(a) - 6000, 6001):
            x = a[j : j + 6001]
            if np.isfinite(x).all() and x.std() > 0:
                fn = f"{tr.id}_{int(tr.stats.starttime.timestamp * 1000)}_{j}.npz"
                np.savez(folder / fn, data=x.astype("float32"))
                rows.append(
                    (
                        fn,
                        tr.id,
                        str(tr.stats.starttime + j / 100),
                        float(tr.stats.sampling_rate),
                    )
                )
    if not rows:
        raise ValueError("No gap-free windows of 6001 samples.")
    pd.DataFrame(rows, columns=["file", "seed_id", "utc_start", "original_hz"]).to_csv(
        folder / "manifest.csv", index=False
    )
    print(
        f"Exported {len(rows)} candidate noise windows. Review for transient arrivals before training."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stead")
    s.add_argument("hdf5", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--count", type=int, default=20000)
    n = sub.add_parser("fdsn")
    n.add_argument("out", type=Path)
    n.add_argument("--station", default="IU.ANMO.00.BHZ")
    n.add_argument("--start", default="2023-01-10T00:00:00Z")
    n.add_argument("--minutes", type=int, default=60)
    n.add_argument("--service", default="https://service.earthscope.org")
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Choose an empty output folder to avoid mixing runs.")
    if a.cmd == "stead":
        conventional_signals(a.hdf5, a.out, a.count)
    else:
        station_noise(a.out, a.station, a.start, a.minutes, a.service)
