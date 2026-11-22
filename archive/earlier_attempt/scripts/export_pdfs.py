"""Render each Office artifact to a fresh directory, validate, then atomically replace its PDF."""
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil,subprocess
root=Path(__file__).resolve().parents[1]/"reports_v3"
for name in ["report.docx","Presentation.pptx"]:
    with TemporaryDirectory() as temp:
        work=Path(temp)
        subprocess.run(["soffice","-env:UserInstallation="+(work/"profile").as_uri(),"--headless","--convert-to","pdf","--outdir",str(work),str(root/name)],check=True)
        result=work/(Path(name).stem+".pdf")
        subprocess.run(["pdfinfo",str(result)],stdout=subprocess.DEVNULL,check=True)
        dest=root/result.name
        part=dest.with_suffix(".pdf.part")
        shutil.copyfile(result,part);part.replace(dest)
