"""Compare independently regenerated V3 tables and numerical arrays with a release."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = set("""boundary classical_events classical_grid classical_selection classical_summary
cluster_intervals clusters crossfit_events crossfit_roles crossfit_summary external_events
external_summary mechanism metric_checks objective_events objective_fits objective_summary
scratch_events scratch_fits scratch_summary sensitivity_events sensitivity_site sensitivity_summary
spectra stead_spectrum stretch_cv_events stretch_events stretch_grid stretch_selection
stretch_strata stretch_summary""".split())
KEYS = ("event_id","array","condition","method","level","fold","case","factor","role","fraction",
        "recording","draw","bin","setting","resampling","metric","context","hz")


def compare(actual, expected):
    if actual.resolve()==expected.resolve():
        raise ValueError("Verification needs separate results and reference directories")
    for path in (actual,expected):
        names={f.stem for f in path.glob("*.csv")}
        if names!=TABLES:
            raise ValueError(f"Unexpected V3 inventory at {path}: {sorted(names^TABLES)}")
    for name in sorted(TABLES):
        a,b=[pd.read_csv(path / (name+".csv")) for path in (actual,expected)]
        ids=[key for key in KEYS if key in a.columns]
        if ids:
            a,b=[d.sort_values(ids).reset_index(drop=True) for d in (a,b)]
        pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=2e-4,atol=2e-4,obj=name)
    for name in ("mask_example","station"):
        with np.load(actual/(name+".npz")) as left, np.load(expected/(name+".npz")) as right:
            if set(left.files)!=set(right.files):
                raise ValueError(f"Different arrays in {name}")
            for key in left.files:
                np.testing.assert_allclose(left[key],right[key],rtol=2e-4,atol=2e-4,err_msg=f"{name}/{key}")
    print(f"Verified {len(TABLES)} V3 result tables and both waveform/mask archives.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual",type=Path)
    parser.add_argument("expected",type=Path)
    args=parser.parse_args()
    compare(args.actual,args.expected)
