"""Check split hygiene, numerical invariants, and the full packaged V2 CSV series."""

from pathlib import Path
import numpy as np
import pandas as pd
from ..data import ROOT, LEVELS


def verify(actual, expected):
    reference = sorted(p.name for p in expected.glob("*.csv"))
    files = sorted(p.name for p in actual.glob("*.csv"))
    if files != reference or not {
        "decoder_summary.csv",
        "sensitivity.csv",
        "size.csv",
        "extremes.csv",
    }.issubset(files):
        raise ValueError("Incomplete V2 output table set")
    for name in files:
        a = pd.read_csv(actual / name)
        b = pd.read_csv(expected / name)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=2e-4, atol=2e-4)
    events = pd.read_csv(ROOT / "data/prepared/events.csv")
    test = set(events[events.split == "test"].event_id)
    val = set(events[events.split == "validation"].event_id)
    dev = val | set(events[events.split == "train"].event_id)
    for name, allowed, methods in [
        ("baseline", test, 7),
        ("decoder", test, 1),
        ("cv", dev, 3),
        ("validation", val, 3),
    ]:
        d = pd.read_csv(actual / f"{name}_events.csv")
        assert set(d.event_id) == allowed and set(d.level) == set(LEVELS)
        assert (
            len(d) == len(allowed) * len(LEVELS) * methods
            and not d.duplicated(["event_id", "method", "level"]).any()
        )
    base = pd.read_csv(actual / "baseline_events.csv")
    legacy = pd.read_csv(ROOT / "results/per_event.csv")
    joined = base.merge(
        legacy, on=["event_id", "level", "method"], validate="one_to_one"
    )
    for m in ("gain_db", "cc", "pick_abs_ms_sum"):
        np.testing.assert_allclose(
            joined[m + "_x"], joined[m + "_y"], atol=1e-5, rtol=1e-5
        )
    np.testing.assert_allclose(
        base.loc[base.method.isin(["Noisy", "Scalar"]), "si_gain_db"], 0, atol=1e-9
    )
    size = pd.read_csv(actual / "size.csv")
    assert set(size.seed) == set(range(2026, 2031)) and len(size) == 80
    sensitivity = pd.read_csv(actual / "sensitivity_events.csv")
    assert set(sensitivity.event_id) == val
    assert len(pd.read_csv(actual / "extremes.csv")) == 160
    print(
        f"Verified {len(files)} V2 tables, disjoint cohorts, original scores and scale invariance."
    )
