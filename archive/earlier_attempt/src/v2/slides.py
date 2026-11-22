"""Build the course presentation with editable PowerPoint text and tables."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.dml import MSO_THEME_COLOR
from .report_additions import at_zero, read, interval
from .stats import FIG
from ..data import ROOT
import pandas as pd


def build_slides():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    ink = "142E3B"
    teal = "087D77"
    muted = "516A75"
    paper = "FAF9F4"

    def color(s):
        return RGBColor.from_string(s)

    def label(sl, x, y, w, h, text, size=20, fg=ink, bold=False):
        box = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = box.text_frame
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = Inches(0)
        for j, line in enumerate(str(text).split("\n")):
            p = frame.paragraphs[0] if j == 0 else frame.add_paragraph()
            p.text = line
            p.font.name = "Calibri"
            p.font.size = Pt(size)
            p.font.bold = bold
            p.font.color.rgb = color(fg)
            p.space_after = Pt(4)
        return box

    def slide(title, subtitle):
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        bg = sl.background.fill
        bg.solid()
        bg.fore_color.rgb = color(paper)
        label(sl, 0.68, 0.38, 12, 0.67, title, 34, bold=True)
        label(sl, 0.70, 1.18, 11.9, 0.65, subtitle, 18, muted)
        bar = sl.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(6.84), Inches(11.9), Inches(0.018)
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = color(teal)
        bar.line.fill.background()
        label(
            sl, 0.7, 6.94, 11, 0.33, "seisDAE  /  DAS earthquake denoising", 12, muted
        )
        label(sl, 12.0, 6.94, 0.5, 0.33, str(len(prs.slides)), 12, muted)
        return sl

    def note(sl, text):
        label(sl, 0.72, 5.87, 11.8, 0.90, text, 18, teal, True)

    def pic(sl, path, x=0.7, y=1.9, w=8.6, h=3.8):
        from PIL import Image

        with Image.open(path) as im:
            iw, ih = im.size
        sc = min(w / iw, h / ih)
        sl.shapes.add_picture(
            str(path),
            Inches(x + (w - iw * sc) / 2),
            Inches(y + (h - ih * sc) / 2),
            width=Inches(iw * sc),
        )

    def grid(sl, frame, x=0.74, y=1.95, w=11.8, h=3.55):
        obj = sl.shapes.add_table(
            len(frame) + 1,
            len(frame.columns),
            Inches(x),
            Inches(y),
            Inches(w),
            Inches(h),
        )
        t = obj.table
        for i, row in enumerate(
            [frame.columns, *frame.itertuples(index=False, name=None)]
        ):
            for j, v in enumerate(row):
                c = t.cell(i, j)
                c.fill.solid()
                c.fill.fore_color.rgb = color(
                    teal if i == 0 else ("EDF2EE" if i % 2 else "FFFFFF")
                )
                c.text = v if isinstance(v, str) else f"{v:.3f}"
                para = c.text_frame.paragraphs[0]
                para.font.name = "Calibri"
                para.font.size = Pt(16)
                para.font.bold = i == 0
                para.font.color.rgb = color("FFFFFF" if i == 0 else ink)

    b = at_zero("baseline_summary").set_index("method")
    dec = at_zero("decoder_summary").iloc[0]
    cv = at_zero("cv_summary").set_index("method")
    ss = at_zero("sensitivity")
    nr = at_zero("noise_ridgecrest")
    n = int(dec.events)
    mag = read("magnitude_contrast").iloc[0]
    sl = slide(
        "Can seisDAE recover a DAS P arrival?",
        "Transfer from conventional seismometers to earthquake records on an optical fiber",
    )
    pic(sl, ROOT / "figures/02_raw_gather.png", 0.78, 1.72, 7.1, 4.0)
    label(sl, 8.05, 2.2, 4.45, 0.70, f"{n} test earthquakes", 27, teal, True)
    label(
        sl,
        8.05,
        3.15,
        4.30,
        2.05,
        "A field arrival serves as the reference. Controlled noise lets us measure what changes in amplitude, shape and timing.",
        21,
    )
    note(
        sl,
        "The output head improves waveform shape modestly. A larger decoder fit does less under the tested training budgets.",
    )
    sl = slide(
        "How a benchmark example is made",
        "The target and the original background come from disjoint windows of one field recording.",
    )
    grid(
        sl,
        pd.DataFrame(
            [
                ["Reference", "70 samples around P on 24 neighboring channels"],
                [
                    "Background",
                    "70 earlier samples; one scale sets each gather’s input SNR",
                ],
                [
                    "Network input",
                    "Centered in 6,001 samples; only the observed center is scored",
                ],
                ["Split", "24 train · 6 validation · 6 previous pilot · 145 test"],
            ],
            columns=["Stage", "Protocol"],
        ),
        h=3.4,
    )
    note(
        sl,
        "Selecting the highest-SNR eligible block favors visible arrivals; the reference contains residual field noise.",
    )
    sl = slide(
        "A lower residual can hide lost waveform shape",
        "Held-out test, 0 dB input. Gains are in dB. Pick MAE is measured in milliseconds.",
    )
    a = b.loc[
        ["Noisy", "Scalar", "f-k", "Zero-shot", "Fine-tuned"],
        ["gain_db", "si_gain_db", "cc", "pick_mae_ms"],
    ].reset_index()
    a.columns = ["Method", "SNR gain", "SI-SDR gain", "CC", "Pick MAE"]
    grid(sl, a)
    note(
        sl,
        "Scalar attenuation improves squared error. Its shape and timing are identical to the noisy input.",
    )
    sl = slide(
        "How the ranking changes with input SNR",
        f"Means and 95% event-bootstrap intervals across {n} held-out earthquakes.",
    )
    pic(sl, FIG / "shape.png", 0.8, 1.64, 9.0, 4.15)
    label(
        sl,
        9.9,
        2.75,
        2.5,
        2.3,
        "SI-SDR removes a global amplitude factor before measuring residual distortion.",
        20,
    )
    note(
        sl, "The f–k filter can improve ordinary gain and still distort waveform shape."
    )
    sl = slide(
        "Smaller events did not benefit more",
        "Paired change in ordinary gain from the released network to the fixed head.",
    )
    pic(sl, FIG / "magnitude.png", 0.75, 1.7, 8.8, 4.15)
    label(
        sl,
        9.55,
        2,
        3.05,
        3.4,
        f"Small minus large at 0 dB\n{mag.small_minus_large_db:+.3f} dB\n95% CI [{mag.lo:.3f}, {mag.hi:.3f}]\n\nBoth extreme bins are Long Valley only.",
        18,
    )
    note(
        sl,
        "Magnitude and source SNR covary. The comparison describes these events and fibers.",
    )
    sl = slide(
        "Updating the last decoder block",
        "Out-of-fold development predictions and one frozen held-out evaluation at 0 dB.",
    )
    rows = []
    for name, r in [
        ("Zero-shot", b.loc["Zero-shot"]),
        ("Fine-tuned", b.loc["Fine-tuned"]),
        ("Decoder", dec),
    ]:
        rows.append(
            [name, cv.loc[name, "gain_db"], r.gain_db, r.si_gain_db, r.pick_mae_ms]
        )
    grid(
        sl,
        pd.DataFrame(
            rows, columns=["Model", "OOF gain", "Test gain", "Test SI gain", "Test MAE"]
        ),
        h=2.65,
    )
    note(
        sl,
        "The decoder beats zero-shot and trails the 16-weight head. Capacity and optimization budget both differ.",
    )
    sl = slide(
        "Padding changes both gain and shape",
        "Fixed checkpoints, identical six validation events at 0 dB input.",
    )
    g = ss[ss.study == "padding"].pivot(
        index="setting", columns="method", values=["gain_db", "si_gain_db"]
    )
    g.columns = ["Head gain", "Released gain", "Head SI", "Released SI"]
    grid(sl, g.reset_index().rename(columns={"setting": "Padding"}))
    note(
        sl,
        "Mirroring and tiling raised ordinary gain while lowering SI-SDR. Stretching changes apparent frequency.",
    )
    sl = slide(
        "The background recording matters",
        "Matched Ridgecrest comparison: the same three validation events and the same fixed head.",
    )
    g = nr[(nr.method == "Fine-tuned")][
        ["setting", "gain_db", "si_gain_db", "cc", "pick_mae_ms"]
    ].copy()
    g.columns = ["Noise source", "Gain", "SI gain", "CC", "Pick MAE"]
    grid(sl, g, h=3.1)
    note(
        sl,
        "All three continuous snippets came from one 30-second recording. Additional independent noise recordings are needed.",
    )
    sl = slide(
        "Training size, aperture and filter order",
        "Development experiments use the same six validation earthquakes.",
    )
    pic(sl, FIG / "size.png", 0.7, 1.75, 8.6, 4.0)
    label(
        sl,
        9.45,
        2.0,
        3.1,
        3.5,
        "Shading spans five training seeds. Wider apertures also change the channels being scored. Both f–k cascades lost SI-SDR at 0 dB.",
        19,
    )
    note(
        sl,
        "More training events also yield more optimizer updates at the same epoch limit.",
    )
    sl = slide(
        "What the experiment establishes",
        "A reproducible comparison on short, processed windows, with event-level uncertainty.",
    )
    label(
        sl,
        0.8,
        2.0,
        5.65,
        3.6,
        "Supported\n\nA small improvement over zero-shot in ordinary and scale-invariant gain. On this split, the original head outperformed the tested decoder fit.",
        22,
    )
    label(
        sl,
        7,
        2.0,
        5.55,
        3.6,
        "Still open\n\nDetection on continuous streams, analyst first motions, transfer to new fibers, and performance across independent noise recordings.",
        22,
    )
    note(
        sl,
        "The repository includes event tables, split membership, both checkpoints, the external noise manifest and the complete rerun command.",
    )
    prs.save(ROOT / "reports_v2/Presentation.pptx")
