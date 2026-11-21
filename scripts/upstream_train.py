"""Launch the pinned original trainer in a dedicated checkout using Keras 2."""

import argparse
import importlib
import os
import random
import subprocess
import sys
from pathlib import Path

REV = "26c9ba0fd70c532f0fb1a612d816cd32475b5036"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkout", type=Path)
    p.add_argument("--signals", type=Path)
    p.add_argument("--noise", type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--count", type=int, default=24)
    p.add_argument("--batch", type=int, default=2)
    a = p.parse_args()
    root = a.checkout.resolve()
    if (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        != REV
    ):
        p.error("Check out the revision listed in README.md.")
    if bool(a.signals) != bool(a.noise):
        p.error("Supply both --signals and --noise or neither.")
    if min(a.epochs, a.batch) < 1 or a.count < 5 * a.batch:
        p.error("Need at least five batches to populate the validation split.")
    changes = {
        "filename": "local_conventional",
        "num_signals": a.count,
        "epochs": a.epochs,
        "workers": 1,
        "batch_size": a.batch,
        "max_queue_size": 1,
    }
    if a.signals:
        for key, folder in (
            ("signal_pathname", a.signals),
            ("noise_pathname", a.noise),
        ):
            if not any(folder.glob("*.npz")):
                p.error(f"No NPZ files in {folder}")
            changes[key] = str(folder.resolve() / "*.npz")
    lines = []
    for line in (root / "model_parfile").read_text().splitlines():
        key = line.partition("=")[0].strip()
        lines.append(f"{key} = {changes[key]}" if key in changes else line)
    parfile = root / "local_conventional.par"
    parfile.write_text("\n".join(lines) + "\n")
    os.environ.update(
        TF_USE_LEGACY_KERAS="1",
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="1",
        MPLBACKEND="Agg",
    )
    # Original code imports both keras and tf.keras. Bind both to matching Keras 2.
    for name in ("", ".layers", ".utils", ".models"):
        sys.modules["keras" + name] = importlib.import_module("tf_keras" + name)
    import numpy as np
    import tensorflow as tf

    random.seed(2026)
    np.random.seed(2026)
    tf.random.set_seed(2026)
    sys.path.insert(0, str(root))
    os.chdir(root)
    from run_model_from_parfile import main as upstream_main

    upstream_main(str(parfile))


if __name__ == "__main__":
    main()
