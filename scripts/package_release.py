"""Create a Windows-compatible ZIP and require byte-identical recovery of every shipped file."""
import argparse
import hashlib
import re
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "SHA256SUMS.txt"
REQUIRED = set("""__init__.py boundary.py crossfit.py deck.py experiments.py extra_plots.py mechanism.py
multi_data.py noise.py objective.py protocol.json release.py report.py scene.py scratch_model.py
sensitivity.py stead.py stretch.py visuals.py""".split())
IGNORE = {".git","__pycache__",".pytest_cache",".ipynb_checkpoints",".venv","runs","vendor","raw","conventional","gathers","upstream","gathers_v41","upstream_v41"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def included(path,root):
    rel=path.relative_to(root)
    return (not set(rel.parts)&IGNORE and rel.name!=MANIFEST and
            path.suffix.lower() not in {".zip",".tmp",".pyc",".part"} and not rel.name.startswith(".~lock."))


def inventory(root):
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and included(p,root)}


def check(root):
    actual=inventory(root)
    expected={}
    for line in (root/MANIFEST).read_text(encoding="ascii").splitlines():
        value,name=line.split("  ",1)
        if name in expected or not re.fullmatch(r"[a-f0-9]{64}",value):
            raise ValueError(f"Invalid manifest entry: {name}")
        expected[name]=value
    if actual!=expected:
        changed=[k for k in sorted(set(actual)|set(expected)) if actual.get(k)!=expected.get(k)]
        raise ValueError(f"Missing, extra or changed files: {changed}")
    if not REQUIRED.issubset({p.name for p in (root/"src/v3").iterdir()}):
        raise ValueError("Incomplete V3 source")
    print(f"{len(actual)} files match SHA-256; all required V3 modules and configuration present.")


def build(dest):
    data=inventory(ROOT)
    lines="".join(f"{value}  {name}\n" for name,value in data.items())
    with ZipFile(dest,"w",compression=ZIP_DEFLATED,allowZip64=False) as z:
        seen=set()
        for name in [*data,MANIFEST]:
            relative="seisdae-das/"+name
            win=PureWindowsPath(relative)
            if (win.is_absolute() or not relative.isascii() or relative.casefold() in seen or
                len(relative)>180 or any(p.is_reserved() or str(p).endswith((" ",".")) or
                re.search(r'[<>:"|?*\x00-\x1f]',str(p)) for p in map(PureWindowsPath,win.parts))):
                raise ValueError(f"Incompatible archive path: {relative}")
            seen.add(relative.casefold())
            info=ZipInfo(relative,(2026,11,10,0,0,0))
            info.create_system=0
            info.external_attr=0x20
            value=lines.encode("ascii") if name==MANIFEST else (ROOT/name).read_bytes()
            z.writestr(info,value,compress_type=ZIP_DEFLATED,compresslevel=8)
    with TemporaryDirectory() as temporary, ZipFile(dest) as archive:
        if archive.testzip():
            raise ValueError("ZIP CRC check failed")
        archive.extractall(temporary)
        check(Path(temporary)/"seisdae-das")
    print(f"{dest.name}  {digest(dest)}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check",action="store_true",help="Verify the extracted release before running experiments")
    group.add_argument("--output",type=Path,help="Write a verified distribution archive")
    args=parser.parse_args()
    check(ROOT) if args.check else build(args.output)
