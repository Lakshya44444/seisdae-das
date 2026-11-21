"""Editable PowerPoint implementation of the shared slide scene and notes."""
import json,re
from PIL import Image
from pptx import Presentation
from pptx.util import Inches,Pt,Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from .scene import make,D
from .visuals import F,ROOT,INK,PAPER
DUR=[34,40,40,44,52,57,53,40,49,43,52,43,31,22]
MORE=[
"Point to the coherent arrival on the right. Return to this image when explaining why neighboring channels are associated observations.",
"Trace the optical path on screen. A bend in the cable changes its directional sensitivity, so similar ground motion can produce different fiber amplitudes.",
"Ask the audience to distinguish two possible successes: reproducing a station prediction, and recovering a previously unseen fiber earthquake. One can succeed while the other fails.",
"Use the dashed pick to orient the audience. The error measures agreement between two implementations. It does not measure how closely either output matches an unknown clean signal.",
"Pause at the crop. The model input is long enough for the network architecture; the actual evidence remains the short recorded center. The pilot set is excluded.",
"Point to a shared peak, then to a shared zero crossing. Neither has moved. A metric that rewards this example needs a second measure before anyone claims better waveform recovery.",
"Follow the f-k row from circle to diamond. It illustrates a genuine physical trade-off. Later, the revised controls change this comparison by changing boundary treatment and the velocity setting.",
"Trace one line from left to right. A gain is relative to a different noisy input at each level. A negative value means the filter increased the particular measured distortion.",
"Read the two extremes first, then the middle group. Whiskers are uncertainties of group means. The direct difference uses its own resampling calculation; visual overlap is not that test.",
"Say what a fairer budget means: equal numbers of gradient updates on the same traces, plus separate validation decisions. The follow-up experiment is described in the report.",
"Keep the denominators separate. The stretch panel scores controlled same-event mixtures. The external panel combines target earthquakes with separately recorded backgrounds. These are distinct questions.",
"Use the color bar: larger values mean the input coefficient passes. Low-energy bins can have striking colors and still contribute very little to the final time-domain signal.",
"If asked about deployment, name the missing evaluation: long untouched noise intervals, a false-alarm metric and an independent array or acquisition campaign.",
"Pause between the three lines. End on the experiment you would run next: a held-out fiber with many independently recorded noise conditions."]
QMAP=[[1,2,4,5,40],[5,6,7,23,26],[1,3,4,32,40],[2,3,13,14,15],[6,7,8,9,33],[19,20,21,27,28],[19,20,23,24,25],[19,20,23,27,38],[7,29,31,35,36],[16,17,18,32,34],[10,11,12,37,39],[13,14,15,16,17],[8,22,28,32,39],[1,3,36,39,40],[29,30,32,33,34],[19,20,21,22,27],[16,17,29,33,34],[8,9,10,37,39],[2,3,30,32,34],[2,3,4,5,18]]
def script(pages):
    qa=[m.groups() for m in re.finditer(r"^\d+\. \*\*(.*?)\*\* (.+)$",(ROOT/"VIVA_NOTES.md").read_text(),re.M)]
    lines=["# Ten-minute talk and examiner prompts","Main talk: ten minutes including pointing and pauses. Backup slides are outside the clock."]
    t=0
    for j,p in enumerate(pages):
        if j<len(DUR):
            end=t+DUR[j];when=f"{t//60}:{t%60:02d}–{end//60}:{end%60:02d}";t=end;p["notes"]+=" "+MORE[j]
        else:when="Backup · allow 30–45 seconds"
        # Attach a reminder tied to the visible chart/slide, and a concrete question bank.
        cues=["145 test events","24 adjacent channels per selected aperture","one reproduced station example","one upstream predictor comparison","70 recorded samples scored","zero SI gain for a scalar","0 dB input","four input levels","two extreme Long Valley bins","16 head weights","three external recording hours","one example aperture","two fibers","three takeaways","six excluded pilot events","one is the target peak ratio","three outer folds","24 trace rows","one reproduction command","one Caltech P-phase dataset"]
        p["notes"]+=" Remember: "+cues[j]+"."
        lines+=["## "+when+" · "+p["title"],p["notes"],"**Backup answers**"]
        lines += ["- **"+qa[k-1][0]+"** "+qa[k-1][1] for k in QMAP[j]]
    assert t==600 and len(qa)==40
    return "\n\n".join(lines)+"\n"
def scene():
    pages=make();talk=script(pages);return pages,talk
def build():
    pages,talk=scene();D.mkdir(exist_ok=True)
    (D/"TALK_SCRIPT.md").write_text(talk)
    prs=Presentation();unit=914400/96
    px=lambda v:Emu(round(v*unit))
    prs.slide_width=px(1280);prs.slide_height=px(720)
    rgb=lambda v:RGBColor.from_string(v.lstrip("#"))
    for p in pages:
        sl=prs.slides.add_slide(prs.slide_layouts[6]);fill=sl.background.fill;fill.solid();fill.fore_color.rgb=rgb(INK if p["dark"] else PAPER)
        for item in p["items"]:
            x,y,w,h=[item[a] for a in ["x","y","w","h"]]
            if item["type"]=="image":
                path=F/item["name"]
                with Image.open(path) as im:iw,ih=im.size
                scale=min(w/iw,h/ih)
                sl.shapes.add_picture(str(path),px(x+(w-iw*scale)/2),px(y+(h-ih*scale)/2),width=px(iw*scale));continue
            if item["type"]=="rect":
                sh=sl.shapes.add_shape(MSO_SHAPE.RECTANGLE,px(x),px(y),px(w),px(h));sh.fill.solid();sh.fill.fore_color.rgb=rgb(item["color"]);sh.line.fill.background();continue
            sh=sl.shapes.add_textbox(px(x),px(y),px(w),px(h));tf=sh.text_frame;tf.word_wrap=True;tf.vertical_anchor=MSO_ANCHOR.MIDDLE
            tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
            for i,line in enumerate(item["v"].splitlines()):
                para=tf.paragraphs[0] if i==0 else tf.add_paragraph()
                para.text=line;para.font.name="Georgia" if item["serif"] else "Calibri";para.font.size=Pt(item["size"]*.75);para.font.bold=item["bold"];para.font.color.rgb=rgb(item["color"]);para.space_after=Pt(0)
        sl.notes_slide.notes_text_frame.text=p["notes"]
    prs.save(D/"Presentation.pptx")
if __name__=="__main__":build()
