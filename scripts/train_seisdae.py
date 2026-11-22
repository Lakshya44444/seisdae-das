"""Run the pinned original seisDAE trainer (Keras 2) from scratch or as fine-tuning.

From scratch (original run_model_from_parfile.py, model_parfile defaults):
  python scripts/train_seisdae.py vendor/seisDAE --signals S --noise N --count C --batch 16 --epochs 100 --name NAME

Fine-tuning a trained model (original retrain_from_parfile.py; its defaults of
50 epochs, early stopping with patience 5 and validation split 0.2 are kept):
  python scripts/train_seisdae.py vendor/seisDAE --signals S --noise N --count C --batch 16 --retrain-from models/gr_mixed_stft.h5 --name NAME

--seed fixes Python, NumPy and TensorFlow random states (default 2026).
"""

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
    p.add_argument("--epochs", type=int, help="Default: the value in the original parameter file")
    p.add_argument("--count", type=int, default=24)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--name", default="local_conventional", help="Model name written by the trainer")
    p.add_argument("--threads", type=int, default=2, help="TensorFlow intra-op CPU threads")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--retrain-from", type=Path, help="Fine-tune this .h5 model (its .config must sit beside it)")
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
    if (a.epochs is not None and a.epochs < 1) or a.batch < 1 or a.count < 5 * a.batch:
        p.error("Need at least five batches to populate the validation split.")
    changes = {
        "filename": a.name,
        "num_signals": a.count,
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
    if a.epochs is not None:
        changes["epochs"] = a.epochs
    if a.retrain_from:
        model = a.retrain_from.resolve()
        config = model.with_suffix(".config")
        if not model.exists() or not config.exists():
            p.error(f"Need {model} and {config}")
        changes.update(model=str(model), config=str(config))
    template = "retrain_parfile" if a.retrain_from else "model_parfile"
    lines = []
    for line in (root / template).read_text().splitlines():
        key = line.partition("=")[0].strip()
        lines.append(f"{key} = {changes[key]}" if key in changes else line)
    parfile = root / f"{a.name}.par"
    parfile.write_text("\n".join(lines) + "\n")
    os.environ.update(
        TF_USE_LEGACY_KERAS="1",
        TF_NUM_INTRAOP_THREADS=str(a.threads),
        TF_NUM_INTEROP_THREADS="1",
        MPLBACKEND="Agg",
    )
    # Original code imports both keras and tf.keras. Bind both to matching Keras 2.
    for name in ("", ".layers", ".utils", ".models"):
        sys.modules["keras" + name] = importlib.import_module("tf_keras" + name)
    import numpy as np
    import tensorflow as tf

    random.seed(a.seed)
    np.random.seed(a.seed)
    tf.random.set_seed(a.seed)
    sys.path.insert(0, str(root))
    os.chdir(root)
    if a.retrain_from:
        from retrain_from_parfile import main_retraining as upstream_main
    else:
        from run_model_from_parfile import main as upstream_main

    upstream_main(str(parfile))


if __name__ == "__main__":
    main()
