"""Quick check that the repository works on a new computer (about 1-2 minutes).

  python -m src.quick_check

Needs only the repository and the packages in requirements.txt: no data
download, no internet. It runs:
  1. package versions
  2. the offline tests in tests/ (no TensorFlow, no data)
  3. the three neural models: released seisDAE, DAS-trained seisDAE, released jDAS
     (load, run on a random input, check output shape; jDAS is also compared
     with its original file)
  4. the main saved results, read from results/*.csv
  5. whether the Ridgecrest windows are on disk (needed only for the full run)
Each step prints PASS or FAIL. Details are in README.md, "How to run".
"""

import contextlib
import importlib
import io
import os
import sys
import unittest

import numpy as np
import pandas as pd

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from .methods import ROOT  # noqa: E402

PACKAGES = ["numpy", "scipy", "pandas", "matplotlib", "h5py", "obspy", "tensorflow", "tf_keras"]
status = []


def report(name, ok, detail=""):
    status.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  - ' + detail if detail else ''}")


def versions():
    print("1. Packages")
    print(f"  Python {sys.version.split()[0]}")
    for p in PACKAGES:
        try:
            v = getattr(importlib.import_module(p), "__version__", "?")
            report(p, True, v)
        except Exception as e:
            report(p, False, f"not importable ({type(e).__name__}); run: pip install -r requirements.txt")


def tests():
    print("2. Offline tests (tests/)")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    with contextlib.redirect_stdout(io.StringIO()):
        result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    bad = len(result.failures) + len(result.errors)
    report(f"{result.testsRun} tests", bad == 0, f"{bad} failed" if bad else "all passed")
    for case, trace in (result.failures + result.errors)[:3]:
        print(f"      {case.id()}: {trace.strip().splitlines()[-1]}")


def networks():
    print("3. Neural models")
    try:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
    except Exception as e:
        report("TensorFlow", False, str(e))
        return
    from . import jdas
    from .methods import SeisDAE

    x = np.random.default_rng(0).normal(size=(12, 6001))
    for name, stem in (("seisDAE released", "gr_mixed_stft"), ("seisDAE DAS-trained", "das_seisdae")):
        try:
            y = SeisDAE(ROOT / f"models/{stem}.h5", ROOT / f"models/{stem}.config")(x)
            report(name, y.shape == x.shape and np.isfinite(y).all(), f"models/{stem}.h5, output {y.shape}")
        except Exception as e:
            report(name, False, f"{type(e).__name__}: {e}")
    try:
        import tf_keras as keras

        rng = np.random.default_rng(0)
        s = rng.normal(size=(2, jdas.NSUB, jdas.NT, 1)).astype("float32")
        m = np.ones_like(s)
        m[[0, 1], [0, 5]] = 0
        mine = jdas.load(jdas.PRETRAINED).predict([s, m], verbose=0)
        ref = keras.models.load_model(str(jdas.PRETRAINED), compile=False).predict([s, m], verbose=0)
        diff = float(np.abs(mine - ref).max() / np.abs(ref).max())
        report("jDAS released", diff < 1e-4, f"rebuilt network vs original file: relative difference {diff:.1e}")
        y = jdas.JDAS(jdas.PRETRAINED)(x)
        report("jDAS on a 100 Hz window", y.shape == x.shape and np.isfinite(y).all(), f"output {y.shape}")
    except Exception as e:
        report("jDAS released", False, f"{type(e).__name__}: {e}")


def results():
    print("4. Saved results (results/*.csv)")
    try:
        t = pd.read_csv(ROOT / "results/variants_auc.csv").set_index("model")
        for mth in ("Noisy", "Bandpass 1-10 Hz", "f-k 1000 m/s", "released", "scratch_s2026"):
            if mth in t.index:
                r = t.loc[mth]
                print(f"      detection AUC  {mth:18} {r.auc:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
        report("results/variants_auc.csv", True)
    except Exception as e:
        report("results/variants_auc.csv", False, str(e))
    try:
        s = pd.read_csv(ROOT / "results/public_check_summary.csv")
        report("results/public_check_summary.csv", True, f"{len(s)} rows")
    except Exception as e:
        report("results/public_check_summary.csv", False, str(e))


def data():
    print("5. Ridgecrest data (only for the full run)")
    g = ROOT / "data/ridgecrest/gathers"
    n = len(list(g.glob("*.npz"))) if g.exists() else 0
    print(f"      {n} windows in data/ridgecrest/gathers"
          + ("" if n else "  (not downloaded: the full run starts with python -m src.build --download)"))


def main():
    print("Quick check of the seisdae-das repository\n")
    versions()
    tests()
    networks()
    results()
    data()
    print(f"\n{sum(status)} of {len(status)} checks passed."
          + ("" if all(status) else " See the FAIL lines above."))
    sys.exit(0 if all(status) else 1)


if __name__ == "__main__":
    main()
