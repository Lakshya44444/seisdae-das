"""Check the released seisDAE model on public broadband seismometer data.

  python -m src.public_check download land      regional earthquakes at a land station (IU.ANMO)
  python -m src.public_check download obs       regional earthquakes at ocean-bottom stations (7D)
  python -m src.public_check evaluate

seisDAE was built for seismometer data, so before blaming the model for its
behaviour on DAS we check it where it is meant to work. Data come from the
public IRIS/EarthScope archive; events from the USGS catalogue.

Each record is one vertical trace around a regional earthquake (1 to 8 degrees,
M >= 4), resampled to 100 Hz:
  signal window  60 s with the predicted P (iasp91) at 10 s, as in the DAS windows
  noise window   60 s ending 70 s before the predicted P (no earthquake)
Records are stored in data/public/<site>/ and listed in data/public/<site>.csv.

Two tests, as for DAS:
  1. real records: SNR (power 0-20 s after P over power 1-9 s) before and after
     each method
  2. synthetic: high-SNR records (raw SNR >= 15 dB) mixed with noise windows of
     other records at -10 to 10 dB; SI-SDR gain against the recorded event
The bandpass is fixed at 1-10 Hz (the DAS detection setting); nothing is tuned here.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.signal import detrend, resample_poly, butter, sosfiltfilt

from .evaluate import FIG, OUT, bootstrap
from .methods import FS, ROOT, SeisDAE, bandpass, mix, si_sdr

DATA = ROOT / "data/public"
SITES = {
    "land": dict(network="IU", station="ANMO", location="00", channel="BHZ",
                 start="2016-01-01", end="2020-12-31", per_event=1),
    "obs": dict(network="7D", station="*", location="*", channel="BHZ",
                start="2013-08-01", end="2014-06-30", per_event=2),
}
MIN_MAG, MIN_DEG, MAX_DEG, MAX_EVENTS = 4.0, 1.0, 8.0, 150
N, P_INDEX = 6001, 1000
BAND = (1.0, 10.0)
LEVELS = [-10, -5, 0, 5, 10]
SEED, DRAWS = 2026, 5000
METHODS = ("Raw", "Bandpass 1-10 Hz", "seisDAE released")


# ---------------------------------------------------------------- download

def first_p(model, depth_km, dist_deg):
    arrivals = model.get_travel_times(source_depth_in_km=max(depth_km, 0.0), distance_in_degree=dist_deg,
                                      phase_list=["P", "p", "Pn", "Pg"])
    return min(a.time for a in arrivals) if arrivals else None


def prepare(tr, t_start):
    """Detrend, 0.5 Hz high-pass, resample to 100 Hz, cut 60 s from t_start."""
    tr = tr.copy()
    tr.detrend("linear")
    tr.data = tr.data.astype("float64")
    sos = butter(4, 0.5, fs=tr.stats.sampling_rate, btype="highpass", output="sos")
    tr.data = sosfiltfilt(sos, tr.data)
    tr.resample(FS)
    i0 = int(round((t_start - tr.stats.starttime) * FS))
    x = tr.data[i0:i0 + N]
    return x if len(x) == N and np.isfinite(x).all() and np.abs(x).max() > 0 else None


def download(site):
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.geodetics import locations2degrees
    from obspy.taup import TauPyModel

    s = SITES[site]
    data_client, event_client, taup = Client("IRIS"), Client("USGS"), TauPyModel("iasp91")
    inv = data_client.get_stations(network=s["network"], station=s["station"], location=s["location"],
                                   channel=s["channel"], starttime=UTCDateTime(s["start"]),
                                   endtime=UTCDateTime(s["end"]), level="channel")
    chans = [dict(net=n.code, sta=st.code, loc=c.location_code, cha=c.code, lat=c.latitude, lon=c.longitude,
                  t0=c.start_date, t1=c.end_date or UTCDateTime(s["end"]))
             for n in inv for st in n for c in st]
    print(f"{len(chans)} channels found")
    lat, lon = np.mean([c["lat"] for c in chans]), np.mean([c["lon"] for c in chans])
    cat = event_client.get_events(starttime=UTCDateTime(s["start"]), endtime=UTCDateTime(s["end"]),
                                  latitude=lat, longitude=lon, maxradius=MAX_DEG + 2, minmagnitude=MIN_MAG)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(cat))
    print(f"{len(cat)} catalogue events, trying up to {MAX_EVENTS}")
    out = DATA / site
    out.mkdir(parents=True, exist_ok=True)
    rows, tried = [], 0
    for k in order:
        if tried >= MAX_EVENTS:
            break
        ev = cat[int(k)]
        o = ev.preferred_origin() or ev.origins[0]
        mag = (ev.preferred_magnitude() or ev.magnitudes[0]).mag
        depth = (o.depth or 0) / 1000
        near = []
        for c in chans:
            d = locations2degrees(o.latitude, o.longitude, c["lat"], c["lon"])
            if MIN_DEG <= d <= MAX_DEG and c["t0"] <= o.time <= c["t1"]:
                near.append((d, c))
        if not near:
            continue
        tried += 1
        for d, c in sorted(near, key=lambda t: t[0])[:s["per_event"]]:
            tp = first_p(taup, depth, d)
            if tp is None:
                continue
            t_p = o.time + tp
            rid = f"{site}_{c['net']}.{c['sta']}_{o.time.strftime('%Y%m%dT%H%M%S')}"
            try:
                st = data_client.get_waveforms(c["net"], c["sta"], c["loc"], c["cha"], t_p - 140, t_p + 60)
                st.merge(fill_value=None)
                if len(st) != 1 or np.ma.is_masked(st[0].data):
                    raise ValueError("gap")
                sig, noi = prepare(st[0], t_p - 10), prepare(st[0], t_p - 130)
                if sig is None or noi is None:
                    raise ValueError("short or empty")
            except Exception as e:
                print(f"  skip {rid}: {type(e).__name__}")
                continue
            np.savez_compressed(out / f"{rid}.npz", signal=sig.astype("float32"), noise=noi.astype("float32"))
            rows.append(dict(id=rid, site=site, station=f"{c['net']}.{c['sta']}", magnitude=mag, depth_km=depth,
                             dist_deg=d, origin=str(o.time)))
        print(f"events tried {tried}, records kept {len(rows)}", end="\r")
    print()
    pd.DataFrame(rows).to_csv(DATA / f"{site}.csv", index=False)
    print(f"{len(rows)} records written to {out}")


# ---------------------------------------------------------------- evaluate

def snr_db(x):
    pre = np.mean(x[..., 100:900] ** 2, axis=-1)
    post = np.mean(x[..., P_INDEX:P_INDEX + 20 * FS] ** 2, axis=-1)
    return 10 * np.log10(np.maximum(post, 1e-30) / np.maximum(pre, 1e-30))


def run(methods, x):
    return {"Raw": x, "Bandpass 1-10 Hz": bandpass(x, BAND), "seisDAE released": methods(x)}


def load(site):
    path = DATA / f"{site}.csv"
    if not path.exists():
        return None, None, None
    m = pd.read_csv(path)
    keep = [i for i in m.id if (DATA / site / f"{i}.npz").exists()]
    m = m[m.id.isin(keep)].reset_index(drop=True)
    sig = np.stack([np.load(DATA / site / f"{i}.npz")["signal"] for i in m.id]).astype("float64")
    noi = np.stack([np.load(DATA / site / f"{i}.npz")["noise"] for i in m.id]).astype("float64")
    return m, sig, noi


def evaluate():
    net = SeisDAE(ROOT / "models/gr_mixed_stft.h5", ROOT / "models/gr_mixed_stft.config")
    real_rows, syn_rows = [], []
    for site in SITES:
        m, sig, noi = load(site)
        if m is None or len(m) == 0:
            print(f"{site}: no records, run 'python -m src.public_check download {site}' first")
            continue
        out = run(net, sig)
        base = snr_db(sig)
        for name, y in out.items():
            gain = snr_db(y) - base
            for i, g in enumerate(gain):
                real_rows.append(dict(site=site, id=m.id[i], method=name, snr_raw_db=base[i], snr_gain_db=g))
        refs = np.where(base >= 15)[0]
        rng = np.random.default_rng(SEED)
        for r in refs:
            others = np.delete(np.arange(len(m)), r)
            for level in LEVELS:
                n = noi[rng.choice(others)]
                x = mix(sig[r], n, level)
                for name, y in run(net, x[None]).items():
                    w = slice(P_INDEX - 100, P_INDEX + 30 * FS)
                    gain = float(si_sdr(sig[r][None, w], y[:, w])[0] - si_sdr(sig[r][None, w], x[None, w])[0])
                    syn_rows.append(dict(site=site, id=m.id[r], level=level, method=name, si_sdr_gain_db=gain))
    real, syn = pd.DataFrame(real_rows), pd.DataFrame(syn_rows)
    real.to_csv(OUT / "public_check_real.csv", index=False)
    syn.to_csv(OUT / "public_check_synthetic.csv", index=False)
    summary(real, syn)
    figure()


def summary(real, syn):
    rows = []
    for (site, name), g in real.groupby(["site", "method"]):
        if name == "Raw":
            continue
        mean, lo, hi = bootstrap(g.snr_gain_db, DRAWS, SEED)
        rows.append(dict(test="real SNR gain", site=site, level="all", method=name, n=len(g), mean_db=mean, lo=lo,
                         hi=hi, improved=f"{(g.snr_gain_db > 0).sum()}/{len(g)}"))
    for (site, level, name), g in syn.groupby(["site", "level", "method"]):
        if name == "Raw":
            continue
        mean, lo, hi = bootstrap(g.si_sdr_gain_db, DRAWS, SEED)
        rows.append(dict(test="synthetic SI-SDR gain", site=site, level=level, method=name, n=len(g), mean_db=mean,
                         lo=lo, hi=hi, improved=f"{(g.si_sdr_gain_db > 0).sum()}/{len(g)}"))
    paired = []
    for (site, level), g in syn.groupby(["site", "level"]):
        w = g.pivot(index="id", columns="method", values="si_sdr_gain_db")
        d = w["seisDAE released"] - w["Bandpass 1-10 Hz"]
        mean, lo, hi = bootstrap(d, DRAWS, SEED)
        paired.append(dict(site=site, level=level, n=len(d), seisdae_minus_bandpass_db=mean, lo=lo, hi=hi))
    s, p = pd.DataFrame(rows), pd.DataFrame(paired)
    s.to_csv(OUT / "public_check_summary.csv", index=False)
    p.to_csv(OUT / "public_check_paired.csv", index=False)
    print("\nReleased seisDAE on public broadband seismometer data (gains in dB, 95% interval)")
    for site in s.site.unique():
        n_rec = real[(real.site == site) & (real.method == "Raw")].shape[0]
        n_ref = syn[(syn.site == site) & (syn.method == "Raw")].id.nunique()
        print(f"\n{site.upper()}: {n_rec} records, {n_ref} high-SNR references for the synthetic test")
        for _, r in s[s.site == site].iterrows():
            lev = "" if r.level == "all" else f" at {int(r.level):+d} dB"
            print(f"  {r.test + lev:34} {r.method:18} {r.mean_db:+6.2f} [{r.lo:+6.2f}, {r.hi:+6.2f}]"
                  f"   better than input in {r.improved}")
        for _, r in p[p.site == site].iterrows():
            sig = "yes" if r.lo > 0 or r.hi < 0 else "no"
            print(f"  seisDAE minus bandpass, SI-SDR at {int(r.level):+d} dB: {r.seisdae_minus_bandpass_db:+6.2f} "
                  f"[{r.lo:+6.2f}, {r.hi:+6.2f}] significant: {sig}")
    print("\nFull numbers: results/public_check_summary.csv, results/public_check_paired.csv")


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    net = SeisDAE(ROOT / "models/gr_mixed_stft.h5", ROOT / "models/gr_mixed_stft.config")
    t = np.arange(N) / FS
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    for row, site in enumerate(SITES):
        m, sig, _ = load(site)
        if m is None or len(m) == 0:
            continue
        i = int(np.argsort(snr_db(sig))[len(m) // 2])
        out = run(net, sig[i][None])
        for col, name in enumerate(METHODS):
            ax = axes[row, col]
            ax.plot(t, out[name][0], lw=0.6, color="k")
            ax.axvline(10, color="tab:red", lw=0.8, ls="--")
            ax.set_title(f"{site.upper()} {m.station[i]} M{m.magnitude[i]:.1f} {m.dist_deg[i]:.1f} deg | {name}",
                         fontsize=9)
            ax.set_yticks([])
    for ax in axes[-1]:
        ax.set_xlabel("Time (s), predicted P at 10 s")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "public_check_examples.png", dpi=150)
    print("figure: results/figures/public_check_examples.png (median-SNR record of each site)")


def control():
    """Checks requested in review (28 Nov), fixed before running.

    1. Null test: the same SNR measure on the noise windows (P-130 s to P-70 s, no
       earthquake expected). A method whose "SNR gain" on pure noise is as large as on
       earthquakes is not recovering signal.
    2. Separation: ROC AUC of earthquake records against noise windows using the
       output SNR of each method (as the DAS detection test).
    3. Real-record SNR gain with a bootstrap over events instead of records (on the
       ocean bottom one event often gives two records).
    """
    from .detection_stats import auc

    net = SeisDAE(ROOT / "models/gr_mixed_stft.h5", ROOT / "models/gr_mixed_stft.config")
    rows, summ = [], []
    rng = np.random.default_rng(SEED)
    for site in SITES:
        m, sig, noi = load(site)
        if m is None or len(m) == 0:
            continue
        out_s, out_n = run(net, sig), run(net, noi)
        base_s, base_n = snr_db(sig), snr_db(noi)
        events = m.id.str.rsplit("_", n=1).str[-1].to_numpy()
        for name in out_s:
            s_db, n_db = snr_db(out_s[name]), snr_db(out_n[name])
            g_s, g_n = s_db - base_s, n_db - base_n
            for i in range(len(m)):
                rows.append(dict(site=site, id=m.id[i], method=name, event_gain_db=g_s[i], noise_gain_db=g_n[i],
                                 event_snr_db=s_db[i], noise_snr_db=n_db[i]))
            ue = np.unique(events)
            boots = []
            for _ in range(2000):
                pick = rng.choice(ue, len(ue))
                idx = np.concatenate([np.where(events == e)[0] for e in pick])
                boots.append(g_s[idx].mean())
            a = auc(s_db, n_db)
            ab = [auc(s_db[rng.integers(0, len(s_db), len(s_db))], n_db[rng.integers(0, len(n_db), len(n_db))])
                  for _ in range(2000)]
            summ.append(dict(site=site, method=name, records=len(m), events=len(ue),
                             event_gain_db=g_s.mean(), event_gain_lo=np.percentile(boots, 2.5),
                             event_gain_hi=np.percentile(boots, 97.5), noise_gain_db=g_n.mean(),
                             net_gain_db=g_s.mean() - g_n.mean(), auc=a, auc_lo=np.percentile(ab, 2.5),
                             auc_hi=np.percentile(ab, 97.5)))
    pd.DataFrame(rows).to_csv(OUT / "public_check_control_rows.csv", index=False)
    d = pd.DataFrame(summ)
    d.to_csv(OUT / "public_check_control.csv", index=False)
    print("Controls for the real-record SNR gain (dB). Brackets: 95% interval, bootstrap over events.\n")
    print(f"{'site':5} {'method':18}{'gain on earthquakes':>28}{'gain on noise':>15}{'net':>8}{'AUC eq vs noise':>26}")
    for r in d.itertuples():
        print(f"{r.site:5} {r.method:18}{r.event_gain_db:+9.2f} [{r.event_gain_lo:+.2f},{r.event_gain_hi:+.2f}]"
              f"{r.noise_gain_db:+15.2f}{r.net_gain_db:+8.2f}{r.auc:10.3f} [{r.auc_lo:.3f},{r.auc_hi:.3f}]")
    print("\nFiles: results/public_check_control.csv, public_check_control_rows.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["download", "evaluate", "control"])
    p.add_argument("site", nargs="?", choices=list(SITES))
    a = p.parse_args()
    if a.step == "download":
        if not a.site:
            p.error("download needs a site: land or obs")
        download(a.site)
    elif a.step == "control":
        control()
    else:
        evaluate()


if __name__ == "__main__":
    main()
