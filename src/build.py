"""Build the 60 s Ridgecrest DAS dataset: earthquake gathers, noise gathers, splits.

Steps (each is resumable; finished windows are cached under data/ridgecrest/gathers):
  python -m src.build --download      fetch candidate events and noise windows
  python -m src.build --export        write single-trace NPZ files for the
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


def fast_snr(x, itp, its, dx):
    """Channel SNR after an f-k fan filter keeps apparent velocities above fast_vmin_m_s.

    Traffic crosses the block slowly (tens of m/s); a local earthquake crosses
    it at km/s. Used only to decide which windows hold an earthquake; the
    stored gathers and every method's input stay unfiltered.
    """
    from .methods import fk

    return channel_snr_db(fk(x, dx, CFG["selection"]["fast_vmin_m_s"]), itp, its)


def selection_snr(name):
    """The per-channel SNR used for selection, export and scoring (config rule)."""
    fast = CFG["selection"].get("snr_on_fast_component")
    return np.load(GATHERS / f"{name}_{'fastsnr' if fast else 'snr'}.npy")


def classify(split, snr, overlap):
    """Event class from per-channel SNR (rules in config.json "selection")."""
    s = CFG["selection"]
    median = float(np.median(snr))
    if overlap and not s.get("allow_overlap", False):
        return "overlap"
    if np.mean(snr >= s["strong_snr_db"]) >= s["strong_min_channel_fraction"]:
        return "strong"
    if split == "test" and not overlap and s["weak_snr_db"][0] <= median < s["weak_snr_db"][1]:
        return "weak"
    return "unused"


def earthquake_snr(m):
    """Per-channel selection SNR for every downloaded earthquake window."""
    eq = m[m.kind != "noise"].copy()
    if not CFG["selection"].get("snr_on_fast_component"):
        return eq, {r.id: np.load(GATHERS / f"{r.id}_snr.npy") for r in eq.itertuples()}
    _, _, dx = geometry()
    out = {}
    for r in eq.itertuples():
        path = GATHERS / f"{r.id}_fastsnr.npy"
        if not path.exists():
            np.save(path, fast_snr(load_gather(r.id), int(r.itp), int(r.its), dx))
        out[r.id] = np.load(path)
    return eq, out


def stats():
    """How many earthquakes each strong rule would keep (no waveform scores involved)."""
    m = manifest()
    eq, snr = earthquake_snr(m)
    print(f"downloaded earthquakes: {len(eq)}  (overlap flag set: {int((eq.overlap > 0).sum())})")
    print("channel_db  fraction | train/val/test no-overlap | train/val/test incl. overlap | train traces")
    for db in (8, 10, 12, 15):
        for frac in (0.2, 0.3, 0.5):
            keep = np.array([np.mean(snr[i] >= db) >= frac for i in eq.id])
            alone = keep & (eq.overlap.to_numpy() == 0)
            count = lambda mask: "/".join(str(int((mask & (eq.split == k).to_numpy()).sum()))
                                          for k in ("train", "validation", "test"))
            traces = sum(int(np.sum(snr[i] >= db)) for i, ok, sp in zip(eq.id, keep, eq.split) if ok and sp == "train")
            print(f"{db:>10}  {frac:>8} | {count(alone):>25} | {count(keep):>28} | {traces:>12}")


def reclassify():
    """Re-apply the config rules to downloaded earthquakes without downloading again."""
    m = manifest()
    eq, snr = earthquake_snr(m)
    for i in eq.index:
        r = m.loc[i]
        m.loc[i, "kind"] = classify(r.split, snr[r.id], r.overlap > 0)
        m.loc[i, "strong_fraction"] = float(np.mean(snr[r.id] >= CFG["selection"]["strong_snr_db"]))
    m.to_csv(MANIFEST, index=False)
    print(m.groupby(["kind", "split"]).size().to_string())


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
        kind = classify(e.split, snr, overlap)
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


def extra_noise():
    """Add more noise windows on test days only (detection test). Resumable.

    New rows carry extra=True; the synthetic test keeps using the original
    noise windows, so its pairs do not change.
    """
    s = CFG["selection"]
    per_day, attempts = s["extra_test_noise_per_day"], s["extra_test_noise_attempts_per_day"]
    m = manifest()
    if "extra" not in m:
        m["extra"] = False
    _, screen, regional = load_catalogs()
    rows, rng = block_rows(), np.random.default_rng(CFG["seed"] + 2)
    days = pd.date_range(CFG["archive"]["first_day"], CFG["archive"]["last_day"], freq="D", tz="UTC")
    added = []
    for day in [d for d in days if split_of(d) == "test"]:
        have = int(((m.kind == "noise") & (m.extra == True) & (m.start.dt.floor("D") == day)).sum())
        found = 0
        for _ in range(attempts):
            if have + found >= per_day:
                break
            start = day + pd.Timedelta(seconds=int(rng.integers(0, 86400 - 120)))
            end = start + pd.Timedelta(seconds=60)
            name = f"noise_{start:%Y%m%d%H%M%S}"
            if name in set(m.id) or other_events(screen, start - pd.Timedelta(seconds=60), end) or \
               other_events(regional, start - pd.Timedelta(seconds=300), end):
                continue
            try:
                fetch(name, start, rows)
            except MissingData:
                continue
            found += 1
            added.append(dict(id=name, kind="noise", split="test", start=start, extra=True))
            m = pd.concat([m, pd.DataFrame(added[-1:])], ignore_index=True)
            m.to_csv(MANIFEST, index=False)
        print(f"{day:%Y-%m-%d}: {have + found} extra noise windows")
    print(m[m.kind == "noise"].groupby(["split", "extra"]).size().to_string())


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
        x, snr = load_gather(r.id), selection_snr(r.id)
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
    p.add_argument("--stats", action="store_true", help="Event counts under several strong-SNR rules")
    p.add_argument("--reclassify", action="store_true", help="Re-apply config rules to downloaded events")
    p.add_argument("--extra-noise", action="store_true", help="More noise windows on test days (detection test)")
    p.add_argument("--limit", type=int, help="Smoke test: this many candidates per kind and 5 days of noise")
    a = p.parse_args()
    if a.stats:
        stats()
    if a.download:
        download(a.limit)
    if a.reclassify:
        reclassify()
    if a.extra_noise:
        extra_noise()
    if a.export:
        export()
