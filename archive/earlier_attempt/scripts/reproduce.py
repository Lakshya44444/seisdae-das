"""Run the complete scientific and publication pipeline in a fresh, isolated working copy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]


def run(jobs):
    if jobs < 1:
        raise ValueError("--jobs must be positive")
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    work = Path(mkdtemp(prefix="reproduction-", dir=runs))
    print(f"Independent rerun: {work}", flush=True)
    ignore = shutil.ignore_patterns(".git", "runs", "__pycache__", "*.pyc", "raw", "conventional", ".venv",
                                    ".ipynb_checkpoints", "*.zip", "SHA256SUMS.txt")
    shutil.copytree(ROOT, work, ignore=ignore, dirs_exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="2",
               TF_NUM_INTEROP_THREADS="1", TF_CPP_MIN_LOG_LEVEL="3", PYTHONHASHSEED="2026", MPLBACKEND="Agg",
               SEISDAE_V2_OUT=str(work / "runs/reproduced_v2"))
    Path(env["SEISDAE_V2_OUT"]).mkdir(parents=True)
    def call(*args):
        print("Running", *args, flush=True)
        subprocess.run([sys.executable,*map(str,args)],cwd=work,env=env,check=True)
    def module(name):
        call("-m",name)

    call("-m","unittest","discover","-s","tests","-v")
    books=sorted((work/"notebooks").glob("*.ipynb"))
    if len(books)!=12:
        raise ValueError("Expected twelve notebooks")
    call("scripts/execute_local.py",*books[:7])
    call("-m","src.verify","runs/notebook_test","--expected",ROOT/"results")
    for name in ["stats","baseline","training","sensitivity"]:
        module("src.v2."+name)
    call("-m","src.v2.training","--sizes")
    module("src.v2.final")
    module("src.v2.failures")
    call("-m","src.verify","runs/reproduced_v2","--v2","--expected",ROOT/"results_v2")
    # Every online-independent derivative must be regenerated. Keep only the pinned
    # STEAD spectrum and its source fingerprint as inputs for the default offline run.
    out=work/"results_v3"
    for path in out.iterdir():
        if path.name not in {"stead_spectrum.csv","stead_source.json"}:
            path.unlink()
    for folder in ("figures_v3","reports_v3"):
        shutil.rmtree(work/folder)
        (work/folder).mkdir()
    studies=["experiments","boundary","mechanism","stretch","noise","sensitivity","objective","crossfit","scratch_model"]
    # V2 completed its immutable development-feature cache before either V3 reader starts.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(module,["src.v3."+s for s in studies]))
    call("-m","src.verify",out,"--v3","--expected",ROOT/"results_v3")
    call("scripts/execute_local.py",*books[7:])
    for name in ["visuals","extra_plots","report","release","deck"]:
        module("src.v3."+name)
    call("scripts/export_pdfs.py")
    call("scripts/check_consistency.py")
    print(f"Full reproduction passed. Reports and result tables: {work}",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs",type=int,default=int(os.environ.get("SEISDAE_JOBS","1")),
                   help="Number of independent V3 experiments to run concurrently")
    run(p.parse_args().jobs)
