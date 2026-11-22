"""Update the results blocks in the README and project log from verified tables."""

from pathlib import Path
import pandas as pd
from ..data import ROOT
from .report_additions import read, at_zero, interval


def replace_section(path, body):
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    s = path.read_text()
    if start not in s:
        s += "\n" + start + "\n" + end + "\n"
    before, tail = s.split(start, 1)
    _, after = tail.split(end, 1)
    path.write_text(before + start + "\n\n" + body.strip() + "\n\n" + end + after)


def update(table):
    base = at_zero("baseline_summary")
    decoder = at_zero("decoder_summary")
    results = pd.concat([base, decoder], ignore_index=True)
    formatted = results[["method", "gain_db", "si_gain_db", "cc", "pick_mae_ms"]].copy()
    formatted.columns = [
        "Method",
        "SNR gain (dB)",
        "SI-SDR gain (dB)",
        "CC",
        "Pick MAE (ms)",
    ]
    c = read("magnitude_contrast").iloc[0]
    head = results.set_index("method").loc["Fine-tuned"]
    d = at_zero("decoder_paired").set_index("comparator").loc["Fine-tuned"]
    s = f"""At 0 dB input on {int(head.events)} test earthquakes, the original fine-tuned head gave {interval(head,'gain_db')} dB ordinary gain and {interval(head,'si_gain_db')} dB scale-invariant gain (95% event-bootstrap intervals). The larger decoder fit trailed the head by {abs(d.gain_db):.3f} dB ordinary gain.

{table(formatted)}

Fine-tuning helped the smallest events, but the head-minus-zero-shot gain at 0 dB was {abs(c.small_minus_large_db):.3f} dB lower in the smallest bin than in the largest. Both extreme bins contain only Long Valley events."""
    replace_section(ROOT / "README_v2.md", s)
    folds = at_zero("cv_summary").set_index("method")
    sizes = at_zero("size").groupby("count").gain_db.mean()
    cohorts = read("strata_paired")
    cohorts = cohorts[(cohorts.level == 0) & (cohorts.stratifier == "magnitude")]
    sens = at_zero("sensitivity")
    pad = sens[(sens.study == "padding") & (sens.method == "Fine-tuned")].set_index(
        "setting"
    )
    noise = at_zero("noise_ridgecrest")
    noise = noise[noise.method == "Fine-tuned"].set_index("setting")
    hy = sens[sens.study == "hybrid"].set_index("method")
    ranks = read("extremes")
    nbs = []
    import json

    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        cells = [
            c for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"
        ]
        nbs.append((len(cells), sum(len(c["outputs"]) for c in cells)))
    log = f"""## Measured outputs

| Task | Measured result |
|---|---|
| T1 | {len(nbs)} notebooks, {sum(v[0] for v in nbs)} executed code cells, {sum(v[1] for v in nbs)} saved output blocks. |
| T2 | Magnitude counts {', '.join(map(str,cohorts.events))}; small-minus-large {c.small_minus_large_db:.3f} dB, 95% CI [{c.lo:.3f}, {c.hi:.3f}]. |
| T3 | Head SI gain {head.si_gain_db:.3f} dB at 0 dB on {int(head.events)} test events; scalar gain is zero on this scale. |
| T4 | Markdown, Word, PDF and an editable ten-slide presentation. |
| T5 | OOF head/decoder ordinary gains {folds.loc['Fine-tuned','gain_db']:.3f}/{folds.loc['Decoder','gain_db']:.3f} dB. Test decoder minus head {interval(d,'gain_db')} dB. |
| T6 | Validation head zero vs reflect ordinary gain {pad.loc['zero','gain_db']:.3f}/{pad.loc['reflect','gain_db']:.3f} dB and SI gain {pad.loc['zero','si_gain_db']:.3f}/{pad.loc['reflect','si_gain_db']:.3f} dB. |
| T7 | Matched Ridgecrest original/external head SI gain {noise.loc['original','si_gain_db']:.3f}/{noise.loc['continuous','si_gain_db']:.3f} dB. |
| T8 | Five-seed mean head gains from {sizes.iloc[0]:.3f} to {sizes.iloc[-1]:.3f} dB across training sizes. |
| T9 | f-k → head gain/SI gain {hy.loc['f-k then head','gain_db']:.3f}/{hy.loc['f-k then head','si_gain_db']:.3f} dB on validation. |
| T10 | {len(ranks)} ranked rows, {ranks.method.nunique()} methods. |
| T11 | Optional larger from-scratch run not performed. |

All gains quoted in this table refer to 0 dB input. Run `bash run_all.sh` for the full sequence. `python -m src.verify runs/reproduced_v2 --v2` compares the independent rerun's scientific values to the packaged tables with absolute/relative tolerance 0.0002. Legacy outputs use 0.00001; elapsed CPU times are excluded."""
    replace_section(ROOT / "CHANGELOG.md", log)
