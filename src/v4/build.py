"""Build the 60 s Ridgecrest DAS dataset: earthquake gathers, noise gathers, splits.

Steps (each is resumable; finished windows are cached under data/v4/gathers):
  python -m src.v4.build --download      fetch candidate events and noise windows
  python -m src.v4.build --export        write single-trace NPZ files for the
                                         original seisDAE trainer (train days only)

Every window has 6001 samples at 100 Hz. Earthquake windows place the
predicted P arrival at sample 1000 (10 s). Days are assigned to train,
validation or test by day number modulo 5, so the three sets never share a day.
"""

import argparse

import numpy as np
import pandas as pd

from .das import CFG, DATA, MissingData, block_rows, geometry, haversine_km, read_window

GATHERS = DATA / "gathers"
MANIFEST = DATA / "manifest.csv"
UPSTREAM = DATA / "upstream"


def split_of(time):
    first = pd.Timestamp(CFG["archive"]["first_day"], tz="UTC")
    index = (pd.Timestamp(time).floor("D") - first).days
    for name in ("train", "validation", "test"):
        if index % 5 in CFG["split"][name]:
            return name
    raise ValueError(index)


def travel_times(events, lat, lon):
    """Straight-ray P and S times from the block centre (window placement only)."""
    w = CFG["window"]
    epi = haversine_km(lat, lon, events.latitude.values, events.longitude.values)
    r = np.hypot(epi, events.depth_km.values)
    return r, r / w["vp_km_s"], r / w["vs_km_s"]


def channel_snr_db(x, itp, its):
    """Per-channel power ratio: event window over the pre-P window, in dB."""
    fs = CFG["window"]["fs"]
    noise = x[:, fs: itp - fs]
    signal = x[:, itp - fs // 2: min(x.shape[1], its + 10 * fs)]
    pn = np.mean(noise.astype(np.float64) ** 2, axis=1)
    ps = np.mean(signal.astype(np.float64) ** 2, axis=1)
    return 10 * np.log10(np.maximum(ps, 1e-30) / np.maximum(pn, 1e-30))


def other_events(screen, t0, t1, exclude=None, min_mag=None):
    hit = (screen.time >= t0) & (screen.time <= t1)
    if exclude is not None:
        hit &= screen.event_id.astype(str) != str(exclude)
    if min_mag is not None:
        hit &= screen.magnitude >= min_mag
    return int(hit.sum())


def load_catalogs():
    read = lambda name: pd.read_csv(DATA / name, parse_dates=["time"])
    events, screen, regional = (read(n) for n in ("catalog_events.csv", "catalog_screen.csv", "catalog_regional.csv"))
    for d in (events, screen, regional):
        d["time"] = pd.to_datetime(d.time, utc=True)
    return events, screen, regional


def candidates(events, lat, lon):
    s, rng = CFG["selection"], np.random.default_rng(CFG["seed"])
    e = events.copy()
    e["dist_km"], e["tp"], e["ts"] = travel_times(e, lat, lon)
    e["split"] = [split_of(t) for t in e.time]
    # Ranking proxy only: larger and closer events first. Measured SNR decides later.
    e["proxy"] = e.magnitude - 1.5 * np.log10(np.maximum(e.dist_km, 1.0))
    strong = e.sort_values("proxy", ascending=False).head(s["strong_candidates"])
    lo, hi = s["weak_magnitude"]
    pool = e[(e.split == "test") & e.magnitude.between(lo, hi) & ~e.event_id.isin(strong.event_id)]
    weak = pool.iloc[rng.permutation(len(pool))[: s["weak_candidates"]]]
    return pd.concat([strong.assign(candidate="strong"), weak.assign(candidate="weak")], ignore_index=True)


def fetch(name, start, rows):
    path = GATHERS / f"{name}.npz"
    if path.exists():
        with np.load(path) as z:
            return z["data"]
    x = read_window(start, rows)
    np.savez_compressed(path, data=x)
    return x


def download(limit=None):
    GATHERS.mkdir(parents=True, exist_ok=True)
    s, w, c = CFG["selection"], CFG["window"], CFG["catalog"]
    fs, itp = w["fs"], w["p_index"]
    lat, lon, spacing = geometry()
    events, screen, regional = load_catalogs()
    rows = block_rows()
    cand = candidates(events, lat, lon)
    if limit:
        cand = cand.groupby("candidate").head(limit)
    records = []
    for k, e in enumerate(cand.itertuples()):
        start = e.time + pd.Timedelta(seconds=e.tp - itp / fs)
        end = start + pd.Timedelta(seconds=(w["samples"] - 1) / fs)
        its = int(round(itp + (e.ts - e.tp) * fs))
        overlap = other_events(screen, start - pd.Timedelta(seconds=10), end, e.event_id, e.magnitude - 1.0)
        name = f"eq_{e.event_id}"
        try:
            x = fetch(name, start, rows)
        except MissingData as error:
            print(f"[{k + 1}/{len(cand)}] {name}: missing ({error})")
            continue
        snr = channel_snr_db(x, itp, its)
        median, frac = float(np.median(snr)), float(np.mean(snr >= s["strong_snr_db"]))
        if overlap:
            kind = "overlap"
        elif frac >= s["strong_min_channel_fraction"]:
            kind = "strong"
        elif e.split == "test" and s["weak_snr_db"][0] <= median < s["weak_snr_db"][1]:
            kind = "weak"
        else:
            kind = "unused"
        np.save(GATHERS / f"{name}_snr.npy", snr)
        records.append(dict(id=name, kind=kind, split=e.split, start=start, event_id=e.event_id,
                            magnitude=e.magnitude, dist_km=e.dist_km, itp=itp, its=min(its, w["samples"] - 1),
                            median_snr_db=median, strong_fraction=frac, overlap=overlap, candidate=e.candidate))
        print(f"[{k + 1}/{len(cand)}] {name} M{e.magnitude:.1f} {e.dist_km:.0f} km {e.split}: "
              f"median SNR {median:.1f} dB -> {kind}")
    rng = np.random.default_rng(CFG["seed"] + 1)
    days = pd.date_range(CFG["archive"]["first_day"], CFG["archive"]["last_day"], freq="D", tz="UTC")
    if limit:
        days = days[:5]
    for day in days:
        found = 0
        for _ in range(s["noise_attempts_per_day"]):
            if found >= (2 if limit else s["noise_windows_per_day"]):
                break
            start = day + pd.Timedelta(seconds=int(rng.integers(0, 86400 - 120)))
            end = start + pd.Timedelta(seconds=60)
            if other_events(screen, start - pd.Timedelta(seconds=60), end) or \
               other_events(regional, start - pd.Timedelta(seconds=300), end):
                continue
            name = f"noise_{start:%Y%m%d%H%M%S}"
            try:
                fetch(name, start, rows)
            except MissingData:
                continue
            found += 1
            records.append(dict(id=name, kind="noise", split=split_of(start), start=start))
        print(f"{day:%Y-%m-%d} ({split_of(day)}): {found} noise windows")
    m = pd.DataFrame(records)
    m.to_csv(MANIFEST, index=False)
    print(m.groupby(["kind", "split"]).size().to_string())
    print(f"channel spacing used for f-k: {spacing:.2f} m")


def manifest():
    m = pd.read_csv(MANIFEST, parse_dates=["start"])
    m["start"] = pd.to_datetime(m.start, utc=True)
    return m


def load_gather(name):
    with np.load(GATHERS / f"{name}.npz") as z:
        return z["data"].astype(np.float64)


def export():
    """Single-trace NPZ files in the format the original seisDAE trainer reads."""
    m, e, s = manifest(), CFG["export"], CFG["selection"]
    sig_dir, noise_dir = UPSTREAM / "signal", UPSTREAM / "noise"
    for d in (sig_dir, noise_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.npz"):
            old.unlink()
    n_sig = n_noise = 0
    for r in m[(m.kind == "strong") & (m.split == "train")].itertuples():
        x, snr = load_gather(r.id), np.load(GATHERS / f"{r.id}_snr.npy")
        for ch in range(0, len(x), e["channel_stride"]):
            if snr[ch] < s["strong_snr_db"]:
                continue
            name = f"{r.id}_c{ch:03d}"
            np.savez(sig_dir / f"{name}.npz", data=x[ch], its=int(r.its), itp=int(r.itp),
                     starttime=str(r.start))
            n_sig += 1
    for r in m[(m.kind == "noise") & (m.split == "train")].itertuples():
        x = load_gather(r.id)
        for ch in range(0, len(x), e["noise_channel_stride"]):
            name = f"{r.id}_c{ch:03d}"
            np.savez(noise_dir / f"{name}.npz", data=x[ch])
            n_noise += 1
    print(f"signal traces: {n_sig}  noise traces: {n_noise}")
    print(f"signal folder: {sig_dir}\nnoise folder:  {noise_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true")
    p.add_argument("--export", action="store_true")
    p.add_argument("--limit", type=int, help="Smoke test: this many candidates per kind and 5 days of noise")
    a = p.parse_args()
    if a.download:
        download(a.limit)
    if a.export:
        export()
