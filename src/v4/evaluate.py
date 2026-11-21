"""V4 evaluation on held-out test days.

  python -m src.v4.evaluate --das-model models/das_seisdae.h5 [--upstream vendor/seisDAE]

1. Settings of bandpass, f-k, the cascade and the scalar control are chosen on
   validation days only (rule in config.json) and frozen.
2. Synthetic test: each test earthquake gather is mixed with real DAS noise
   gathers from test days at every level; all methods are scored against the
   recorded earthquake gather (which still holds its own field noise).
3. Field test: weak test earthquakes are denoised as recorded (no added noise);
   measured SNR and neighbouring-channel coherence are reported. There is no
   clean reference here, so these are descriptive measures.
Outputs go to results_v4/ and figures_v4/.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import build
from .build import load_gather, manifest, channel_snr_db
from .das import CFG, geometry
from .methods import ROOT, SeisDAE, bandpass, coherence, fk, mix, scores

OUT, FIG = ROOT / "results_v4", ROOT / "figures_v4"
E = CFG["evaluation"]


def event_window(r):
    fs = CFG["window"]["fs"]
    return slice(int(r.itp) - fs // 2, min(CFG["window"]["samples"], int(r.its) + 10 * fs))


def good_channels(r):
    """Channels whose recorded reference reaches the scoring SNR (fixed rule, config)."""
    snr = np.load(build.GATHERS / f"{r.id}_snr.npy")
    return snr >= E["score_channel_snr_db"]


def pairs(events, noise, draws, seed):
    """Deterministic noise assignment: (event row, noise id, draw)."""
    rng = np.random.default_rng(seed)
    ids = noise.id.to_numpy()
    out = []
    for r in events.itertuples():
        for d, j in enumerate(rng.choice(len(ids), size=min(draws, len(ids)), replace=False)):
            out.append((r, ids[j], d))
    return out


class Runner:
    def __init__(self, das_model, dx):
        self.dx = dx
        self.gr = SeisDAE(ROOT / "models/gr_mixed_stft.h5", ROOT / "models/gr_mixed_stft.config")
        self.das = SeisDAE(das_model, das_model.with_suffix(".config"))
        self.ms = {}

    def timed(self, name, fn, x):
        t = time.perf_counter()
        y = fn(x)
        self.ms.setdefault(name, []).append(1000 * (time.perf_counter() - t) / len(x))
        return y

    def outputs(self, x, settings):
        """All method outputs for one gather; settings maps method -> frozen value."""
        y = {"Noisy": x}
        y["Scalar"] = settings["Scalar"] * x
        y["Bandpass"] = self.timed("Bandpass", lambda v: bandpass(v, settings["Bandpass"]), x)
        y["f-k"] = self.timed("f-k", lambda v: fk(v, self.dx, settings["f-k"]), x)
        y["seisDAE (released)"] = self.timed("seisDAE (released)", self.gr, x)
        y["seisDAE (DAS-trained)"] = self.timed("seisDAE (DAS-trained)", self.das, x)
        y["seisDAE (DAS) + f-k"] = fk(y["seisDAE (DAS-trained)"], self.dx, settings["seisDAE (DAS) + f-k"])
        return y


def select(runner, m):
    """Grid search on validation days. Returns frozen settings and the grid table."""
    ev = m[(m.kind == "strong") & (m.split == "validation")]
    nz = m[(m.kind == "noise") & (m.split == "validation")]
    if ev.empty or nz.empty:
        raise ValueError("Need validation earthquakes and validation noise")
    rows = []
    for r, nid, _ in pairs(ev, nz, E["validation_noise_draws"], CFG["seed"] + 10):
        s, n, win, ok = load_gather(r.id), load_gather(nid), event_window(r), good_channels(r)
        for level in E["levels_db"]:
            x = mix(s, n, level)
            for a in E["scalar_grid"]:
                rows.append(dict(method="Scalar", setting=json.dumps(a), event=r.id, level=level,
                                 **scores(s[ok], x[ok], (a * x)[ok], win)))
            for band in E["bandpass_hz"]:
                rows.append(dict(method="Bandpass", setting=json.dumps(band), event=r.id, level=level,
                                 **scores(s[ok], x[ok], bandpass(x, band)[ok], win)))
            das = runner.das(x)
            for v in E["fk_velocity_m_s"]:
                rows.append(dict(method="f-k", setting=json.dumps(v), event=r.id, level=level,
                                 **scores(s[ok], x[ok], fk(x, runner.dx, v)[ok], win)))
                rows.append(dict(method="seisDAE (DAS) + f-k", setting=json.dumps(v), event=r.id, level=level,
                                 **scores(s[ok], x[ok], fk(das, runner.dx, v)[ok], win)))
    grid = pd.DataFrame(rows)
    table = grid.groupby(["method", "setting"], as_index=False)[["gain_db", "si_gain_db"]].mean()
    chosen = {}
    for method, g in table.groupby("method"):
        key = "gain_db" if method == "Scalar" else "si_gain_db"
        chosen[method] = json.loads(g.loc[g[key].idxmax(), "setting"])
    return chosen, table, len(ev)


def bootstrap(values, draws, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    boot = np.nanmean(values[idx], axis=1)
    return float(np.nanmean(values)), *np.nanquantile(boot, [0.025, 0.975])


def summarize(d, keys, measures, seed):
    rows = []
    for labels, g in d.groupby(keys):
        row = dict(zip(keys, labels if isinstance(labels, tuple) else (labels,)), events=len(g))
        for q in measures:
            row[q], row[q + "_lo"], row[q + "_hi"] = bootstrap(g[q], E["bootstrap_draws"], seed)
        rows.append(row)
    return pd.DataFrame(rows)


def synthetic(runner, m, settings):
    ev = m[(m.kind == "strong") & (m.split == "test")]
    nz = m[(m.kind == "noise") & (m.split == "test")]
    rows, keep = [], {}
    example = ev.iloc[np.argsort(np.abs(ev.median_snr_db - ev.median_snr_db.median()).to_numpy())[0]].id
    for k, (r, nid, draw) in enumerate(pairs(ev, nz, E["noise_draws"], CFG["seed"] + 20)):
        s, n, win, ok = load_gather(r.id), load_gather(nid), event_window(r), good_channels(r)
        for level in E["levels_db"]:
            x = mix(s, n, level)
            for method, y in runner.outputs(x, settings).items():
                rows.append(dict(event=r.id, noise=nid, draw=draw, level=level, method=method,
                                 scored_channels=int(ok.sum()), **scores(s[ok], x[ok], y[ok], win)))
                if r.id == example and draw == 0 and level == 0:
                    keep[method] = y.astype(np.float32)
            if r.id == example and draw == 0 and level == 0:
                keep["Reference"] = s.astype(np.float32)
        print(f"synthetic {k + 1}/{len(ev) * E['noise_draws']}", end="\r")
    print()
    d = pd.DataFrame(rows)
    np.savez_compressed(OUT / "example_synthetic.npz", event=example, **keep)
    return d, len(ev), len(nz)


def field(runner, m, settings):
    ev = m[(m.kind == "weak") & (m.split == "test")]
    rows, keep = [], {}
    example = ev.iloc[np.argsort(np.abs(ev.median_snr_db - ev.median_snr_db.median()).to_numpy())[0]].id \
        if len(ev) else None
    for r in ev.itertuples():
        x, win = load_gather(r.id), event_window(r)
        for method, y in runner.outputs(x, settings).items():
            if method == "Scalar":
                continue
            snr = channel_snr_db(y, int(r.itp), int(r.its))
            rows.append(dict(event=r.id, magnitude=r.magnitude, method=method,
                             median_snr_db=float(np.median(snr)), channels_10db=float(np.mean(snr >= 10)),
                             coherence=coherence(y, win)))
            if r.id == example:
                keep[method] = y.astype(np.float32)
    if keep:
        np.savez_compressed(OUT / "example_field.npz", event=example, **keep)
    return pd.DataFrame(rows), len(ev)


def run(das_model, upstream=None):
    OUT.mkdir(exist_ok=True)
    m = manifest()
    _, _, dx = geometry()
    runner = Runner(das_model, dx)
    record = {"channel_spacing_m": dx, "das_model": str(das_model)}
    if upstream:
        trace = load_gather(m[m.kind == "strong"].iloc[0].id)[0]
        record["upstream_check_released"] = runner.gr.check_upstream(upstream, trace)
        record["upstream_check_das"] = runner.das.check_upstream(upstream, trace)
        print("upstream relative peak errors:", record["upstream_check_released"], record["upstream_check_das"])
    settings, grid, n_val = select(runner, m)
    grid.to_csv(OUT / "selection_grid.csv", index=False)
    settings["Noisy"] = None
    record.update(settings=settings, validation_events=n_val)
    print("frozen settings:", settings)

    d, n_test, n_noise = synthetic(runner, m, settings)
    d.to_csv(OUT / "synthetic_rows.csv", index=False)
    per_event = d.groupby(["event", "level", "method"], as_index=False).mean(numeric_only=True)
    per_event.to_csv(OUT / "synthetic_events.csv", index=False)
    measures = ["gain_db", "si_gain_db", "cc", "gain_db_event", "si_gain_db_event", "cc_event"]
    summarize(per_event, ["level", "method"], measures, CFG["seed"]).to_csv(OUT / "synthetic_summary.csv", index=False)
    wide = per_event.pivot_table(index=["event", "level"], columns="method", values="si_gain_db")
    paired = []
    for a, b in (("seisDAE (DAS-trained)", "seisDAE (released)"), ("seisDAE (DAS-trained)", "f-k"),
                 ("seisDAE (DAS) + f-k", "f-k"), ("seisDAE (DAS) + f-k", "seisDAE (DAS-trained)")):
        for level, g in (wide[a] - wide[b]).groupby(level="level"):
            mean, lo, hi = bootstrap(g.to_numpy(), E["bootstrap_draws"], CFG["seed"])
            paired.append(dict(comparison=f"{a} minus {b}", level=level, si_gain_db=mean, lo=lo, hi=hi,
                               events=len(g), wins=int((g > 0).sum())))
    pd.DataFrame(paired).to_csv(OUT / "synthetic_paired.csv", index=False)

    f, n_weak = field(runner, m, settings)
    if len(f):
        f.to_csv(OUT / "field_events.csv", index=False)
        summarize(f, ["method"], ["median_snr_db", "channels_10db", "coherence"], CFG["seed"]).to_csv(
            OUT / "field_summary.csv", index=False)
    record.update(test_events=n_test, test_noise_windows=n_noise, weak_test_events=n_weak,
                  ms_per_trace={k: float(np.median(v)) for k, v in runner.ms.items()})
    (OUT / "run.json").write_text(json.dumps(record, indent=2, default=str))
    s = pd.read_csv(OUT / "synthetic_summary.csv")
    print(s[s.level == 0][["method", "events", "gain_db", "si_gain_db", "si_gain_db_lo", "si_gain_db_hi",
                           "si_gain_db_event", "cc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--das-model", type=Path, default=ROOT / "models/das_seisdae.h5")
    p.add_argument("--upstream", help="Original seisDAE checkout for an adapter cross-check")
    a = p.parse_args()
    run(a.das_model, a.upstream)
