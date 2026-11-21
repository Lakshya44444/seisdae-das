"""Denoise an observed 100 Hz Caltech P-phase gather, without adding noise."""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from .data import FS, ROOT, load
from .neural import N, denoise, model


def read_event(path, start, count):
    if start < 0 or count < 1:
        raise ValueError(
            "Specify a nonnegative first row and a positive channel count."
        )
    with h5py.File(path) as f:
        d = f["data/P/data"]
        if not np.isclose(d.attrs["dt"], 1 / FS):
            raise ValueError("This checkpoint expects input at 100 Hz.")
        if start + count > d.shape[0]:
            raise ValueError(
                f"The row range exceeds this file's {d.shape[0]} channels."
            )
        return d[start : start + count, :], str(f["data"].attrs["event_id"])


def run(raw, event_id, out, first_row, head):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not 1 <= raw.shape[1] <= N or not np.isfinite(raw).all():
        raise ValueError("Expected finite channel × time data of length 1 to 6001.")
    out.mkdir(parents=True, exist_ok=True)
    base = denoise(model(), raw)
    tuned = denoise(model(head), raw)
    np.savez_compressed(
        out / "observed.npz",
        observed=raw,
        zero_shot=base,
        fine_tuned=tuned,
        sampling_hz=FS,
        event_id=event_id,
        first_h5_row=first_row,
    )
    fig, axes = plt.subplots(
        1, 3, sharex=True, sharey=True, figsize=(10, 3), layout="constrained"
    )
    raw_centered = raw - raw.mean(axis=1, keepdims=True)
    limit = max(np.quantile(abs(raw_centered), 0.99), np.finfo(float).tiny)
    for ax, signal, label in zip(
        axes,
        (raw_centered, base, tuned),
        ("Observed, demeaned", "Zero-shot", "Fine-tuned"),
    ):
        img = ax.imshow(
            signal,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(0, raw.shape[1] / FS, len(raw) - 0.5, -0.5),
        )
        ax.set_title(label)
        ax.set_xlabel("Time in clip (s)")
    axes[0].set_ylabel("Channel offset")
    fig.colorbar(img, ax=axes, label="Archive amplitude")
    fig.suptitle(f"Observed P-phase clip · event {event_id}")
    fig.savefig(out / "observed.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--event", type=int, help="Event ID in the bundled manifest")
    source.add_argument("--h5", type=Path, help="Caltech phase_P/<event>.h5 file")
    p.add_argument("--first-row", type=int, default=0, help="First HDF5 row for --h5")
    p.add_argument("--count", type=int, default=24, help="Number of HDF5 rows for --h5")
    p.add_argument("--out", type=Path, default=ROOT / "runs/field")
    p.add_argument("--head", type=Path, default=ROOT / "models/das_head.npz")
    a = p.parse_args()
    if a.h5:
        raw, eid = read_event(a.h5, a.first_row, a.count)
        row = a.first_row
    else:
        events, pairs = load(ROOT / "data/prepared")
        selection = (
            events[events.split == "test"]
            if a.event is None
            else events[events.event_id == a.event]
        )
        if selection.empty:
            p.error("Event ID absent from data/prepared/events.csv.")
        i = selection.index[0]
        raw, eid, row = (
            pairs["raw"][i],
            int(events.at[i, "event_id"]),
            int(events.at[i, "first_h5_row"]),
        )
    run(raw, eid, a.out, row, a.head)
    print(f"Wrote observed.npz and observed.png to {a.out}")
