"""Synthetic test (SI-SDR) for every seisDAE variant and seed.

  python -m src.synthetic_seeds

Uses exactly the mixtures of the synthetic test in evaluate.py (same test
earthquakes, same noise windows, same random draws, same scored channels).
Only the seisDAE models are run; bandpass and f-k are read from
results/synthetic_events.csv. The released model and from-scratch seed 2026
are run again as a check: they must reproduce the earlier numbers.
Writes results/synthetic_seeds_events.csv, synthetic_seeds_summary.csv and
synthetic_seeds_paired.csv.
"""

import numpy as np
import pandas as pd

from .build import load_gather, manifest
from .das import CFG
from .evaluate import E, OUT, bootstrap, event_window, good_channels, original, pairs
from .methods import SeisDAE, mix, scores
from .variants import family, models


def main():
    m = original(manifest())
    ev = m[(m.kind == "strong") & (m.split == "test")]
    nz = m[(m.kind == "noise") & (m.split == "test")]
    nets = {name: SeisDAE(h5, h5.with_suffix(".config")) for name, h5 in models().items()}
    jobs = pairs(ev, nz, E["noise_draws"], CFG["seed"] + 20)
    rows = []
    for k, (r, nid, draw) in enumerate(jobs):
        s, n, win, ok = load_gather(r.id), load_gather(nid), event_window(r), good_channels(r)
        for level in E["levels_db"]:
            x = mix(s, n, level)
            for name, net in nets.items():
                rows.append(dict(event=r.id, draw=draw, level=level, model=name,
                                 si_gain_db=scores(s[ok], x[ok], net(x)[ok], win)["si_gain_db"]))
        print(f"synthetic {k + 1}/{len(jobs)}", end="\r")
    print()
    d = pd.DataFrame(rows).groupby(["event", "level", "model"], as_index=False).si_gain_db.mean()
    d.to_csv(OUT / "synthetic_seeds_events.csv", index=False)

    wide = d.pivot_table(index=["event", "level"], columns="model", values="si_gain_db")
    old = pd.read_csv(OUT / "synthetic_events.csv").pivot_table(index=["event", "level"], columns="method",
                                                                values="si_gain_db")
    check = {"released": "seisDAE (released)", "scratch_s2026": "seisDAE (DAS-trained)"}
    for new, earlier in check.items():
        diff = np.nanmax(np.abs(wide[new] - old[earlier].reindex(wide.index)))
        print(f"check {new} against earlier '{earlier}': largest difference {diff:.2e} dB")
    for b in ("Bandpass", "f-k"):
        wide[b] = old[b].reindex(wide.index)

    summary, paired = [], []
    for name in nets:
        for level, g in wide[name].groupby(level="level"):
            mean, lo, hi = bootstrap(g.to_numpy(), E["bootstrap_draws"], CFG["seed"])
            summary.append(dict(model=name, family=family(name), level=level, si_gain_db=mean, lo=lo, hi=hi))
        for b in ("released", "Bandpass", "f-k"):
            if b == name:
                continue
            for level, g in (wide[name] - wide[b]).groupby(level="level"):
                mean, lo, hi = bootstrap(g.to_numpy(), E["bootstrap_draws"], CFG["seed"])
                paired.append(dict(model=name, family=family(name), minus=b, level=level, si_gain_db=mean,
                                   lo=lo, hi=hi, events=len(g), wins=int((g > 0).sum())))
    summary, paired = pd.DataFrame(summary), pd.DataFrame(paired)
    summary.to_csv(OUT / "synthetic_seeds_summary.csv", index=False)
    paired.to_csv(OUT / "synthetic_seeds_paired.csv", index=False)
    print(f"{len(ev)} test earthquakes, {E['noise_draws']} noise draws each")
    print("SI-SDR gain at 0 dB input SNR:")
    print(summary[summary.level == 0].round(2).to_string(index=False))
    print("Minus released seisDAE, range over input SNR levels:")
    r = paired[paired.minus == "released"].groupby("model").agg(
        lowest=("si_gain_db", "min"), highest=("si_gain_db", "max"), lowest_lo=("lo", "min"),
        fewest_wins=("wins", "min"), most_wins=("wins", "max"), events=("events", "first"))
    print(r.round(2).to_string())


if __name__ == "__main__":
    main()
