"""Build the study report and result-backed text from released CSVs."""
import json
import re
from pathlib import Path
from string import Template

import pandas as pd
from .visuals import ROOT, F, read

TEMPLATES = Path(__file__).with_name("templates")
D = ROOT / "reports_v3"


def row(d, **where):
    for key, val in where.items():
        d = d[d[key] == val]
    if len(d) != 1:
        raise ValueError(f"Expected one row for {where}, found {len(d)}")
    return d.iloc[0]


def pair(r):
    return f"{r.gain_db:.3f} / {r.si_gain_db:.3f}"


def bounds(r, metric):
    return f"[{r[metric+'_lo']:.3f}, {r[metric+'_hi']:.3f}]"


def interval(r, metric):
    return f"{r[metric]:.3f} {bounds(r, metric)}"


def table(headers, rows):
    entries = [headers, ["---"] * len(headers), *rows]
    return "\n".join("| " + " | ".join(map(str, cells)) + " |" for cells in entries)


def evidence():
    old = read("baseline_summary", 2)
    klass = read("classical_summary")
    stretch = read("stretch_summary")
    combined = pd.concat([klass, stretch], ignore_index=True)
    head_old = row(old, level=0, method="Fine-tuned")
    head = row(stretch, level=0, method="Fine-tuned")
    fk = row(klass, level=0, method="f-k")
    scalar = row(old, level=0, method="Scalar")
    scratch = row(read("scratch_summary"),level=0)
    mag = read("magnitude_contrast",2).iloc[0]
    mech = read("mechanism").set_index("factor")
    original_rc = row(read("site_summary",2),array="ridgecrest",method="Fine-tuned",level=0)
    external = read("external_summary").query("method=='Fine-tuned' and level==0")
    boundary = read("boundary").query("level==0").set_index("context")
    vals = dict(head_original=pair(head_old), head_stretch=pair(head), fk_new=pair(fk), scalar=pair(scalar),
                upstream=f'{json.loads((ROOT/"results/reproduction.json").read_text())["upstream_relative_peak_error"]:.2e}',
                short=pair(boundary.loc["short"]), padded=pair(boundary.loc["zero padded"]),
                fk_ci=interval(fk,"si_gain_db"), magnitude_ci=f"{mag.small_minus_large_db:.3f} dB [{mag.lo:.3f}, {mag.hi:.3f}]",
                head_factor=int(head.factor), released_factor=int(row(stretch,level=0,method="Zero-shot").factor),
                external_si=f"{external.si_gain_db.mean():.3f}", original_rc_si=f"{original_rc.si_gain_db:.3f}",
                population_head=pair(row(read("crossfit_summary"),fraction=1,level=0)), scratch=pair(scratch),
                scratch_ci=interval(scratch,"si_gain_db"), mask_original=f"{mech.loc[1,'weighted_mask']:.3f}",
                mask_stretch=f"{mech.loc[4,'weighted_mask']:.3f}", change_original=f"{mech.loc[1,'relative_change']:.3f}",
                change_stretch=f"{mech.loc[4,'relative_change']:.3f}")
    methods = []
    for name in ["Scalar","Bandpass","Wavelet","f-k","Zero-shot","Fine-tuned","Median stack","SVD"]:
        prior = pair(row(old,level=0,method=name)) if name in set(old.method) else "—"
        new = pair(row(combined,level=0,method=name)) if name != "Scalar" else "Same control"
        methods.append([name,prior,new])
    vals["methods_table"] = table(["Method","Original gains","Revised gains"],methods)
    vals["comparison"] = table(["Method","Original gains (dB)","Revised gains (dB)"],
                               [cells for cells in methods if cells[0] in {"Bandpass","f-k","Zero-shot","Fine-tuned"}])
    vals["controls_table"] = table(["Condition","SI-SDR (dB)","Peak ratio","Sign","CC"],
        [[r.case, f"{r.si_sdr_db:.3f}", f"{r.peak_ratio:.3f}", int(r.sign), f"{r.cc:.3f}"] for r in read("metric_checks").itertuples()])
    clusters=read("cluster_intervals").query("level==0")
    grouping=[]
    for unit,g in clusters.groupby("resampling"):
        a,b = row(g,metric="gain_db"), row(g,metric="si_gain_db")
        grouping.append([unit,int(a.clusters if unit=="cluster" else a.events),f"[{a.lo:.3f}, {a.hi:.3f}]",f"[{b.lo:.3f}, {b.hi:.3f}]"])
    vals["cluster_table"]=table(["Resampling","Units","Gain 95% CI (dB)","SI 95% CI (dB)"],grouping)
    vals["stretch_table"]=table(["Model","Factor","Ordinary gain (dB)","SI gain (dB)"],
        [[r.method,int(r.factor),interval(r,"gain_db"),interval(r,"si_gain_db")] for _,r in stretch.query("level==0").iterrows()])
    noises=[["Original early window",f"{original_rc.gain_db:.3f}",f"{original_rc.si_gain_db:.3f}"]]
    noises += [[r.recording,f"{r.gain_db:.3f}",f"{r.si_gain_db:.3f}"] for r in external.itertuples()]
    vals["noise_table"]=table(["Background","Ordinary gain (dB)","SI gain (dB)"],noises)
    vals["sensitivity_table"]=table(["Condition","Events","Ordinary gain (dB)","SI gain (dB)"],
        [[r.condition,r.events,f"{r.gain_db:.3f}",f"{r.si_gain_db:.3f}"] for r in read("sensitivity_summary").query("level==0 and method=='Fine-tuned'").itertuples()])
    fits=read("objective_fits")[["method","params","steps"]].drop_duplicates()
    joined=read("objective_summary").query("level==0").merge(fits,on="method",validate="one_to_one")
    vals["objective_table"]=table(["Model and loss","Weights","Steps","Gain (dB)","SI (dB)"],
        [[r.method,r.params,r.steps,f"{r.gain_db:.3f}",f"{r.si_gain_db:.3f}"] for r in joined.itertuples()])
    vals["levels_table"]=table(["Input (dB)","Revised f–k","Released + stretch","Head + stretch"],
        [[level,*[pair(row(combined,level=level,method=m)) for m in ["f-k","Zero-shot","Fine-tuned"]]] for level in sorted(old.level.unique())])
    return vals


def content():
    """Represent each report page once for Markdown, Word and the consistency checker."""
    md = Template((TEMPLATES / "report.md.in").read_text()).substitute(evidence())
    pages=[]
    for page in md.split("<!-- page -->"):
        items=[]
        for block in page.strip().split("\n\n"):
            if block.startswith("| "):
                cells=[[t.strip() for t in line.strip("|").split("|")] for line in block.splitlines()]
                _,caption=items.pop()
                items.append(("table",caption,cells[0],cells[2:]))
            elif block.startswith("!["):
                m=re.fullmatch(r"!\[(.*)\]\(\.\./figures_v3/([^)]*)\)",block)
                items.append(("image",m[2],m[1]))
            elif block.startswith("> "):items.append(("box",[s[2:].strip() for s in block.splitlines()]))
            elif block.startswith("## "):items.append(("heading",block[3:]))
            elif block.startswith("# "):items.append(("title",block[2:]))
            else:items.append(("text",block))
        pages.append(items)
    return pages


def markdown(pages):
    blocks=[]
    for page in pages:
        for kind,*a in page:
            if kind=="table":blocks.extend([a[0],table(a[1],a[2])])
            elif kind=="image":blocks.append(f"![{a[1]}](../figures_v3/{a[0]})")
            elif kind=="heading":blocks.append("## "+a[0])
            elif kind=="title":blocks.append("# "+a[0])
            elif kind=="box":blocks.append("  \n".join("> "+s for s in a[0]))
            else:blocks.append(a[0])
    return "\n\n".join(blocks)+"\n"


def build():
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    D.mkdir(exist_ok=True)
    pages=content()
    (D/"report.md").write_text(markdown(pages))
    doc=Document()
    section=doc.sections[0]
    section.page_height=Inches(11.7);section.page_width=Inches(8.3)
    section.left_margin=section.right_margin=Inches(.75)
    section.top_margin=section.bottom_margin=Inches(.65)
    section.footer.paragraphs[0].text="SeisDAE on optical fiber   ·   IIT Roorkee"
    normal=doc.styles["Normal"]
    normal.font.name="Calibri";normal.font.size=Pt(10)
    normal.paragraph_format.space_after=Pt(4)
    normal.paragraph_format.line_spacing=1.04
    for j,page in enumerate(pages):
        if j:doc.add_page_break()
        for kind,*a in page:
            if kind in ("title","heading"):
                doc.add_heading(a[0],level=0 if kind=="title" else 2)
            elif kind=="text":doc.add_paragraph(a[0])
            elif kind=="box":
                doc.add_paragraph("\n".join(a[0]),style="Quote")
            elif kind=="image":
                doc.add_picture(str(F/a[0]),width=Inches(6.35))
                doc.add_paragraph(a[1],style="Caption")
            else:
                doc.add_paragraph(a[0],style="Caption")
                tbl=doc.add_table(rows=0,cols=len(a[1]));tbl.style="Light Shading Accent 1"
                for idx,values in enumerate([a[1],*a[2]]):
                    cells=tbl.add_row().cells
                    for cell,value in zip(cells,values):
                        para=cell.paragraphs[0];para.paragraph_format.space_after=Pt(2)
                        run=para.add_run(str(value));run.bold=idx==0;run.font.size=Pt(9)
                    if idx==0:tbl.rows[-1]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    doc.save(D/"report.docx")


if __name__=="__main__":
    build()
