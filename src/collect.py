"""Copy a model written by the original seisDAE trainer into models/.

  python -m src.collect vendor/seisDAE --name das_ridgecrest_v42

Finds <name>.h5 and <name>.config anywhere inside the checkout, copies them to
models/das_seisdae.h5 and models/das_seisdae.config, and prints the config so
its STFT settings can be compared with the released GR model.
"""

import argparse
import shutil
from pathlib import Path

from .methods import ROOT, read_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkout", type=Path)
    p.add_argument("--name", default="das_ridgecrest")
    a = p.parse_args()
    for suffix in (".h5", ".config"):
        found = sorted(a.checkout.rglob(f"{a.name}{suffix}"), key=lambda f: f.stat().st_mtime)
        if not found:
            raise SystemExit(f"No {a.name}{suffix} under {a.checkout}")
        target = ROOT / "models" / f"das_seisdae{suffix}"
        shutil.copy2(found[-1], target)
        print(f"{found[-1]} -> {target}")
    new = read_config(ROOT / "models/das_seisdae.config")
    old = read_config(ROOT / "models/gr_mixed_stft.config")
    for key in sorted(set(new) | set(old)):
        flag = "" if new.get(key) == old.get(key) else "   <- differs from released GR model"
        print(f"{key}: {new.get(key)}{flag}")


if __name__ == "__main__":
    main()
