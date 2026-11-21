"""Evaluate every method on identical held-out event gathers and SNR levels."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .data import LEVELS, ROOT, load, mix
from .metrics import bandpass, fk, scores, wavelet
from .summary import paired_intervals, summarise


def fit_scalar(events, pairs):
    """Minimize mean reference-energy-normalized error over training mixtures."""
    cross, power = 0.0, 0.0
    for i in events.index[events.split == "train"]:
        s, noise = pairs["reference"][i], pairs["noise"][i]
        energy = np.sum(s * s)
        for level in LEVELS:
            x = mix(s, noise, level)
            cross += np.sum(x * s) / energy
            power += np.sum(x * x) / energy
    return cross / power


def evaluate(folder, out, head, classical=False, save_predictions=False):
    events, pairs = load(folder)
    scalar = fit_scalar(events, pairs)
    methods = {
        "Scalar": lambda x: scalar * x,
        "Noisy": lambda x: x,
        "Bandpass": bandpass,
        "Wavelet": wavelet,
        "f-k": fk,
    }
    if not classical:
        from .neural import denoise, model

        base, tuned = model(), model(head)
        methods.update(
            {
                "Zero-shot": lambda x: denoise(base, x),
                "Fine-tuned": lambda x: denoise(tuned, x),
            }
        )
        for net in (base, tuned):
            denoise(
                net, pairs["reference"][0, :1]
            )  # exclude graph initialization from timing
    rows, show = [], {}
    if save_predictions:
        (out / "predictions").mkdir(parents=True, exist_ok=True)
    test = events[events.split == "test"]
    for i, event in test.iterrows():
        s, n, p = (pairs[key][i] for key in ("reference", "noise", "nominal_p"))
        for level in LEVELS:
            x = mix(s, n, level)
            predictions = {}
            for name, fun in methods.items():
                t = perf_counter()
                y = fun(x)
                elapsed = perf_counter() - t
                if save_predictions:
                    predictions[name] = y
                rows.append(
                    dict(
                        event_id=event.event_id,
                        array=event["array"],
                        level=level,
                        method=name,
                        ms_per_trace=elapsed * 1000 / len(s),
                        **scores(s, x, y, p),
                    )
                )
                if i == test.index[0] and level == 0:
                    show[name] = y
                    show.update(
                        reference=s,
                        raw=pairs["raw"][i],
                        event_id=event.event_id,
                        nominal_p=p,
                    )
            if save_predictions:
                np.savez(
                    out / "predictions" / f"{event.event_id}_{level}.npz", **predictions
                )
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_event.csv", index=False)
    summary = summarise(df, ["level", "method"])
    summary.to_csv(out / "summary.csv")
    summarise(df, ["array", "level", "method"]).to_csv(out / "by_array.csv")
    paired_intervals(df).to_csv(out / "paired_gain.csv", index=False)
    pd.DataFrame({"training_scalar": [scalar]}).to_csv(out / "scalar.csv", index=False)
    np.savez_compressed(out / "example.npz", **show)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/prepared")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--head", type=Path, default=ROOT / "models/das_head.npz")
    p.add_argument("--classical", action="store_true")
    a = p.parse_args()
    destination = a.out or ROOT / "runs" / (
        "classical" if a.classical else "evaluation"
    )
    evaluate(a.data, destination, a.head, a.classical)
