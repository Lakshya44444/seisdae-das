"""Load the editable slide geometry and bind displayed outcomes directly to result tables."""
import json
from string import Template
from .report import D, TEMPLATES, row, evidence
from .visuals import read


def make():
    pages = json.loads((TEMPLATES / "slides.json").read_text())
    vals = evidence()
    base = read("baseline_summary", 2)
    for key,method in [("zero","Zero-shot"),("head","Fine-tuned")]:
        vals[key] = f"{row(base,method=method,level=0).gain_db:.3f}"
    scalar=row(base,method="Scalar",level=0)
    vals.update(scalar_gain=f"{scalar.gain_db:.2f}",scalar_si=f"{scalar.si_gain_db:.2f}",
                magnitude=f'{read("magnitude_contrast",2).iloc[0].small_minus_large_db:.3f}',
                decoder=f'{row(read("decoder_summary",2),method="Decoder",level=0).gain_db:.3f}',
                stretch_si=f'{row(read("stretch_summary"),method="Fine-tuned",level=0).si_gain_db:+.3f}')
    for method,key in [("Zero-shot","cv_zero"),("Fine-tuned","cv_head"),("Decoder","cv_decoder")]:
        vals[key]=f'{row(read("cv_summary",2),method=method,level=0).gain_db:.3f}'
    for page in pages:
        for item in page["items"]:
            if item["type"]=="text":
                item["v"]=Template(item["v"]).substitute(vals)
        page["notes"]=Template(page["notes"]).substitute(vals)
    return pages
