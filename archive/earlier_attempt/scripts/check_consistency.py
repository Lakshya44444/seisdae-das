"""Reject mismatches in result-backed prose and charts, including the actual delivered PPTX."""
from pathlib import Path
import sys,zipfile
from hashlib import sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.v3.release import viva,readme
from src.v3.report import content,markdown
from src.v3.deck import scene
from src.v3.visuals import ROOT,F
from pptx import Presentation
from docx import Document
from docx.oxml.ns import qn
def equal(a,b,where):
    if a!=b:raise ValueError("Consistency mismatch in "+where)
def check_summary_rows():
    """Recompute point estimates and event-bootstrap intervals from the event rows."""
    import pandas as pd
    from src.v2.stats import aggregate
    from src.v2.baseline import MEASURES
    def data(stem):
        return pd.read_csv(ROOT/"results_v3"/(stem+".csv"))
    rules = [
        ("classical", ["method","level"], MEASURES),
        ("stretch", ["method","factor","level"], MEASURES),
        ("objective", ["method","level"], ["gain_db","si_gain_db","cc","pick_mae_ms"]),
        ("crossfit", ["fraction","level"], ["gain_db","si_gain_db","cc"]),
        ("scratch", ["method","level"], ["gain_db","si_gain_db","cc"]),
        ("sensitivity", ["condition","method","level"], MEASURES),
    ]
    for stem, groups, measures in rules:
        rebuilt=aggregate(data(stem+"_events"),groups,measures)
        pd.testing.assert_frame_equal(rebuilt,data(stem+"_summary"),check_exact=False,rtol=1e-8,atol=1e-8,obj=stem)
    cols=["gain_db","si_gain_db","cc"]
    ext=data("external_events").groupby(["recording","level","method"],as_index=False)[cols].mean()
    pd.testing.assert_frame_equal(ext,data("external_summary"),check_exact=False,rtol=1e-8,atol=1e-8)
    print("Recomputed six bootstrap summary tables and the matched external-noise means from event rows.")

def check():
    check_summary_rows()
    sections=content();pages,talk=scene();D=ROOT/"reports_v3"
    for path,source in [(ROOT/"README.md",readme()),(ROOT/"VIVA_NOTES.md",viva()),(D/"report.md",markdown(sections)),(D/"TALK_SCRIPT.md",talk)]:
        equal(path.read_text(),source,path.name)
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    prs=Presentation(D/"Presentation.pptx");equal(len(prs.slides),len(pages),"slide count")
    for j,(sl,p) in enumerate(zip(prs.slides,pages),1):
        where=f"slide {j}"
        target=[t["v"] for t in p["items"] if t["type"]=="text"]
        equal([sh.text for sh in sl.shapes if sh.has_text_frame and sh.text.strip()],target,where)
        equal(sl.notes_slide.notes_text_frame.text,p["notes"],where+" speaker notes")
        expected=[sha256((F/t["name"]).read_bytes()).digest() for t in p["items"] if t["type"]=="image"]
        equal([sha256(sh.image.blob).digest() for sh in sl.shapes if sh.shape_type==MSO_SHAPE_TYPE.PICTURE],expected,where+" figure")
    doc=Document(D/"report.docx");values=[]
    for page in sections:
        for k,*a in page:
            if k in ["text","heading","title"]:values.append(a[0])
            elif k=="image":values.append(a[1])
            elif k=="box":values.extend(a[0])
            elif k=="table":values+=[a[0],*a[1],*[str(cell) for r in a[2] for cell in r]]
    text=" ".join(el.text or "" for el in doc.element.body.iter(qn("w:t")))
    import re
    nums=lambda t:re.findall(r"[-−+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?",t)
    equal(nums(text),nums(" ".join(values)),"Word body numbers")
    with zipfile.ZipFile(D/"report.docx") as z:
        actual={sha256(z.read(k)).digest() for k in z.namelist() if k.startswith("word/media/")}
    equal(actual,{sha256((F/a[0]).read_bytes()).digest() for pg in sections for kind,*a in pg if kind=="image"},"Word figures")
    print(f"Matched README, viva, Markdown and Word, {len(pages)} slides, speaker notes, embedded figures and timed script.")
if __name__=="__main__":check()
