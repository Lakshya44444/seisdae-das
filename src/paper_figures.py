"""Figures for the paper.

  python -m src.paper_figures            figures 1-10 from results/*.csv and data/ridgecrest/*.csv (seconds)
  python -m src.paper_figures --data     also figures 11-15 from the Ridgecrest windows and the trained models
                                         (needs the full run; a few minutes on a laptop CPU)

All figures go to results/figures/paper/ as PNG (300 dpi) and PDF.
No figure chooses its example by how good it looks: the example event is the
test event whose U-Net SI-SDR gain at 0 dB is the median of the 19 test events.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .das import CFG, DATA  # noqa: E402
from .detection_stats import auc  # noqa: E402
from .evaluate import E, OUT, bootstrap  # noqa: E402

PAPER = OUT / "figures/paper"
LEVELS = [-10, -5, 0, 5, 10]

# Colour-blind-safe categorical colours (Okabe-Ito based, checked with a CVD
# validator) plus a marker per method, so identity never rests on colour alone.
STYLE = {
    "Bandpass": dict(color="#555555", marker="s", ls="-"),
    "f-k": dict(color="#009E73", marker="^", ls="-"),
    "seisDAE released": dict(color="#B07A00", marker="v", ls="--"),
    "seisDAE fine-tuned (3 seeds)": dict(color="#0072B2", marker="o", ls="-"),
    "seisDAE from scratch (3 seeds)": dict(color="#0072B2", marker="o", ls=":"),
    "jDAS fine-tuned": dict(color="#CC79A7", marker="D", ls="-"),
    "2-D U-Net v1 (3 seeds)": dict(color="#D55E00", marker="*", ls="-"),
}
LABEL = {
    "Bandpass": "Bandpass 0.5–45 Hz",
    "f-k": "f-k 250 m/s",
    "seisDAE released": "seisDAE, released",
    "seisDAE fine-tuned (3 seeds)": "seisDAE, fine-tuned (3 seeds)",
    "seisDAE from scratch (3 seeds)": "seisDAE, from scratch (3 seeds)",
    "jDAS fine-tuned": "jDAS, fine-tuned",
    "2-D U-Net v1 (3 seeds)": "2-D U-Net, ours (3 seeds)",
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E3E6EA", "grid.linewidth": 0.6, "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
})


def save(fig, name):
    PAPER.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER / f"{name}.png", bbox_inches="tight")
    fig.savefig(PAPER / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def boot(x):
    return bootstrap(np.asarray(x, dtype=float), E["bootstrap_draws"], CFG["seed"])


def panel_letter(ax, k):
    """Put '(a)' in front of the panel title, left-aligned, so letters never collide with titles."""
    t = ax.get_title()
    ax.set_title("")
    ax.set_title(f"({'abcdefghij'[k]}) {t}".strip(), loc="left", fontsize=8.5, fontweight="bold")


# ---------------------------------------------------------------- figures from tables
def fig_map():
    info = pd.read_csv(DATA / "das_info.csv")
    m = pd.read_csv(DATA / "manifest.csv")
    ev = pd.read_csv(DATA / "catalog_events.csv")
    first, n = CFG["channels"]["first_row"], CFG["channels"]["count"]
    block = info[(info["index"] >= first) & (info["index"] < first + n)]
    used = ev[ev.event_id.isin(m.event_id.dropna().astype(int))]
    used = used.merge(m[["event_id", "split"]].drop_duplicates("event_id"), on="event_id")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    colors = {"train": "#0072B2", "validation": "#B07A00", "test": "#D55E00"}
    for split, g in used.groupby("split"):
        ax.scatter(g.longitude, g.latitude, s=4 + 6 * g.magnitude.clip(0) ** 2, c=colors[split], alpha=0.6,
                   lw=0, label=f"{split} ({len(g)})")
    ax.plot(info.longitude, info.latitude, color="#333333", lw=1.2, label="fiber (all channels)")
    ax.plot(block.longitude, block.latitude, color="#FFD400", lw=3.5, label=f"{n} channels used")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect(1 / np.cos(np.radians(info.latitude.mean())))
    ax.legend(loc="lower left", frameon=True, fontsize=7, markerscale=1.5)
    panel_letter(ax, 0)
    ax = axes[1]
    w = m[m.kind != "noise"].dropna(subset=["magnitude"])
    for split, g in w.groupby("split"):
        ax.scatter(g.dist_km, g.magnitude, s=10, c=colors[split], alpha=0.75, lw=0, label=split)
    ax.set_xlabel("Epicentral distance (km)")
    ax.set_ylabel("Magnitude")
    ax.legend(fontsize=7)
    panel_letter(ax, 1)
    fig.tight_layout()
    save(fig, "fig01_map_events")


def per_event_families():
    from .final_table import per_event
    return per_event()


def fig_synthetic(fam):
    fs = pd.read_csv(OUT / "final_synthetic.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for name, st in STYLE.items():
        g = fs[fs.method == name].sort_values("level")
        if g.empty:
            continue
        if name in ("Bandpass", "2-D U-Net v1 (3 seeds)"):
            ax.fill_between(g.level, g.lo, g.hi, color=st["color"], alpha=0.12, lw=0)
        ax.plot(g.level, g.si_gain_db, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.8,
                ms=8 if st["marker"] == "*" else 5, label=LABEL[name])
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(LEVELS)
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("SI-SDR gain (dB)")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, "fig03_synthetic_sisdr")

    # jDAS released is far below; shown separately so the scale stays readable
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    for name in ("jDAS released", "jDAS fine-tuned", "jDAS from scratch"):
        g = fs[fs.method == name].sort_values("level")
        ax.plot(g.level, g.si_gain_db, marker="D", lw=1.5, label=name,
                ls={"jDAS released": "--", "jDAS fine-tuned": "-", "jDAS from scratch": ":"}[name], color="#CC79A7")
    g = fs[fs.method == "Bandpass"].sort_values("level")
    ax.plot(g.level, g.si_gain_db, color="#555555", marker="s", label="Bandpass 0.5–45 Hz")
    ax.set_xticks(LEVELS)
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("SI-SDR gain (dB)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, "figS1_synthetic_jdas")


def fig_paired(fam):
    others = ["Bandpass", "f-k", "seisDAE fine-tuned (3 seeds)", "jDAS fine-tuned"]
    fig, axes = plt.subplots(1, len(others), figsize=(7.6, 2.6), sharey=True)
    rng = np.random.default_rng(0)
    for k, (ax, b) in enumerate(zip(axes, others)):
        for i, lv in enumerate(LEVELS):
            d = (fam["2-D U-Net v1 (3 seeds)"] - fam[b]).xs(lv, level="level").dropna().to_numpy()
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=7, color="#D55E00", alpha=0.45, lw=0)
            mean, lo, hi = boot(d)
            sig = lo > 0 or hi < 0
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color="black", ms=4,
                        mfc="black" if sig else "white", capsize=2, lw=1)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xticks(range(len(LEVELS)), [str(v) for v in LEVELS])
        ax.set_xlabel("Input SNR (dB)")
        ax.set_title({"Bandpass": "minus bandpass", "f-k": "minus f-k", "seisDAE fine-tuned (3 seeds)":
                      "minus seisDAE (FT)", "jDAS fine-tuned": "minus jDAS (FT)"}[b])
        panel_letter(ax, k)
    axes[0].set_ylabel("Difference in SI-SDR gain (dB)")
    fig.tight_layout()
    save(fig, "fig04_paired_unet")


def roc_curve(e, n):
    thr = np.unique(np.concatenate([e, n]))[::-1]
    tpr = [np.mean(e >= t) for t in thr]
    fpr = [np.mean(n >= t) for t in thr]
    return np.r_[0, fpr, 1], np.r_[0, tpr, 1]


def fig_detection():
    rows = pd.read_csv(OUT / "variants_rows.csv")
    show = {"Raw data": ("Noisy", dict(color="#999999", ls=":")),
            "Bandpass 1–10 Hz": ("Bandpass 1-10 Hz", dict(color="#555555", ls="-")),
            "f-k 1000 m/s": ("f-k 1000 m/s", dict(color="#009E73", ls="-")),
            "seisDAE, released": ("released", dict(color="#B07A00", ls="--")),
            "seisDAE, fine-tuned s2026": ("finetune_s2026", dict(color="#0072B2", ls="-")),
            "jDAS, fine-tuned": ("jdas_finetune_s2026", dict(color="#CC79A7", ls="-")),
            "2-D U-Net, s2026": ("unet2d_s2026", dict(color="#D55E00", ls="-"))}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    for name, (m, st) in show.items():
        g = rows[rows.method == m]
        e, n = g[g.label == "earthquake"].stat_db.to_numpy(), g[g.label == "noise"].stat_db.to_numpy()
        x, y = roc_curve(e, n)
        ax.plot(x, y, lw=1.5, label=f"{name} ({auc(e, n):.2f})", **st)
    ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8)
    ax.set_xlabel("False-alarm rate (noise windows)")
    ax.set_ylabel("Detection rate (earthquake windows)")
    ax.legend(fontsize=6.3, loc="lower right")
    ax.set_aspect("equal")
    panel_letter(ax, 0)

    ax = axes[1]
    a = pd.read_csv(OUT / "variants_auc.csv")
    order = ["Noisy", "Bandpass 1-10 Hz", "f-k 1000 m/s", "released", "scratch_s2026", "scratch_s2027",
             "scratch_s2028", "finetune_s2026", "finetune_s2027", "finetune_s2028", "jdas_pretrained",
             "jdas_finetune_s2026", "jdas_scratch_s2026", "unet2d_s2026", "unet2d_s2027", "unet2d_s2028",
             "unet2d_v2_s2026", "unet2d_v3_s2026"]
    names = {"Noisy": "Raw data", "Bandpass 1-10 Hz": "Bandpass 1–10 Hz", "f-k 1000 m/s": "f-k 1000 m/s",
             "released": "seisDAE released", "jdas_pretrained": "jDAS released",
             "jdas_finetune_s2026": "jDAS fine-tuned", "jdas_scratch_s2026": "jDAS from scratch",
             "unet2d_v2_s2026": "U-Net v2", "unet2d_v3_s2026": "U-Net v3"}
    col = lambda m: ("#D55E00" if m.startswith("unet2d") else "#CC79A7" if m.startswith("jdas") else
                     "#0072B2" if m.startswith(("scratch", "finetune")) else "#B07A00" if m == "released" else
                     "#555555")
    for i, m in enumerate(order):
        r = a[a.model == m].iloc[0]
        ax.errorbar(r.auc, -i, xerr=[[r.auc - r.lo], [r.hi - r.auc]], fmt="o", color=col(m), ms=4, capsize=2)
    lab = [names.get(m, m.replace("scratch_s", "seisDAE scratch s").replace("finetune_s", "seisDAE fine-tuned s")
                     .replace("unet2d_s", "U-Net v1 s")) for m in order]
    ax.set_yticks([-i for i in range(len(order))], lab, fontsize=7)
    bp = a[a.model == "Bandpass 1-10 Hz"].auc.iloc[0]
    ax.axvline(bp, color="#555555", lw=0.8, ls="--")
    ax.axvline(0.5, color="#BBBBBB", lw=0.8)
    ax.set_xlabel("ROC AUC [95% interval]")
    panel_letter(ax, 1)
    fig.tight_layout()
    save(fig, "fig05_detection")


def fig_magnitude():
    rows = pd.read_csv(OUT / "variants_rows.csv")
    m = pd.read_csv(DATA / "manifest.csv")[["id", "magnitude"]].rename(columns={"id": "window"})
    rows = rows.merge(m, on="window", how="left")
    methods = {"Raw data": "Noisy", "Bandpass 1–10 Hz": "Bandpass 1-10 Hz", "f-k 1000 m/s": "f-k 1000 m/s",
               "seisDAE released": "released", "seisDAE fine-tuned s2026": "finetune_s2026",
               "jDAS fine-tuned": "jdas_finetune_s2026", "U-Net s2026": "unet2d_s2026"}
    bins = [(-9, 1.5, "M < 1.5"), (1.5, 9, "M ≥ 1.5")]
    out = []
    rng = np.random.default_rng(CFG["seed"])
    for name, key in methods.items():
        g = rows[rows.method == key]
        n = g[g.label == "noise"].stat_db.to_numpy()
        for lo, hi, lab in bins:
            e = g[(g.label == "earthquake") & (g.magnitude >= lo) & (g.magnitude < hi)].stat_db.to_numpy()
            vals = [auc(rng.choice(e, len(e)), rng.choice(n, len(n))) for _ in range(1000)]
            out.append(dict(method=name, bin=lab, events=len(e), auc=auc(e, n),
                            lo=np.percentile(vals, 2.5), hi=np.percentile(vals, 97.5)))
    d = pd.DataFrame(out)
    d.to_csv(OUT / "paper_detection_by_magnitude.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    cols = ["#999999", "#555555", "#009E73", "#B07A00", "#0072B2", "#CC79A7", "#D55E00"]
    width = 0.11
    for k, (name, c) in enumerate(zip(methods, cols)):
        g = d[d.method == name]
        x = np.arange(len(bins)) + (k - 3) * width
        ax.errorbar(x, g.auc, yerr=[g.auc - g.lo, g.hi - g.auc], fmt="o", color=c, ms=4, capsize=1.5,
                    label=name, lw=1)
    ax.axhline(0.5, color="#BBBBBB", lw=0.8)
    ax.set_xticks(range(len(bins)), [f"{b[2]}\n({int(d[d.bin == b[2]].events.iloc[0])} events)" for b in bins])
    ax.set_ylabel("ROC AUC vs 210 noise windows")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left")
    fig.tight_layout()
    save(fig, "fig06_detection_magnitude")


def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    for name, ls in (("s2026", "-"), ("s2027", "--"), ("s2028", ":")):
        h = pd.read_csv(OUT / f"unet2d_train_{name}.csv")
        ax.plot(h.epoch, h.val_loss, color="#D55E00", ls=ls, lw=1.4, label=f"v1 {name} validation")
    for name, c in (("v2_s2026", "#0072B2"), ("v3_s2026", "#009E73")):
        h = pd.read_csv(OUT / f"unet2d_train_{name}.csv")
        ax.plot(h.epoch, h.val_loss, color=c, lw=1.2, label=f"{name.split('_')[0]} (own validation mixtures)")
    ax.set_ylim(0.2, 0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (validation mixtures)")
    ax.set_title("2-D U-Net")
    ax.legend(fontsize=6.5)
    panel_letter(ax, 0)
    ax = axes[1]
    for name, ls in (("finetune_s2026", "-"), ("scratch_s2026", "--")):
        h = pd.read_csv(OUT / f"jdas_train_{name}.csv")
        ax.plot(h.epoch, h.val_loss, color="#CC79A7", ls=ls, lw=1.4, label=name.replace("_s2026", ""))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE on the hidden channel")
    ax.set_title("jDAS (self-supervised)")
    ax.legend(fontsize=7)
    panel_letter(ax, 1)
    fig.tight_layout()
    save(fig, "figS2_training")


def fig_public():
    real = pd.read_csv(OUT / "public_check_real.csv")
    syn = pd.read_csv(OUT / "public_check_synthetic.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    bins = np.arange(-10, 45, 2.5)
    for k, (site, title) in enumerate((("land", "Land, IU.ANMO"), ("obs", "Ocean bottom, 7D"))):
        ax = axes[k]
        for meth, c in (("Bandpass 1-10 Hz", "#555555"), ("seisDAE released", "#B07A00")):
            v = real[(real.site == site) & (real.method == meth)].snr_gain_db
            ax.hist(v, bins=bins, color=c, alpha=0.55, label=f"{meth} (median {v.median():+.1f})")
        ax.axvline(0, color="#999999", lw=0.8)
        ax.set_xlabel("Real-record SNR gain (dB)")
        ax.set_title(f"{title} ({real[(real.site == site) & (real.method == 'Raw')].shape[0]})")
        ax.legend(fontsize=6.3)
        panel_letter(ax, k)
    axes[0].set_ylabel("Records")
    ax = axes[2]
    for site, ls in (("land", "-"), ("obs", "--")):
        for meth, c in (("Bandpass 1-10 Hz", "#555555"), ("seisDAE released", "#B07A00")):
            g = syn[(syn.site == site) & (syn.method == meth)].groupby("level").si_sdr_gain_db.mean()
            ax.plot(g.index, g.values, color=c, ls=ls, marker="o", ms=3.5,
                    label=f"{site.upper()} {meth.split(' ')[0]}")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(LEVELS)
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("SI-SDR gain (dB)")
    ax.set_title("Synthetic test")
    ax.legend(fontsize=6.3)
    panel_letter(ax, 2)
    fig.tight_layout()
    save(fig, "fig02_public_check")


def fig_jdas_diag():
    d = pd.read_csv(OUT / "jdas_diagnose.csv")
    c = pd.read_csv(OUT / "jdas_diagnose_cmr.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax = axes[0]
    meths = [("Bandpass 1-10 Hz", "#555555"), ("f-k 1000 m/s", "#009E73"), ("seisDAE DAS-trained", "#0072B2"),
             ("jDAS fine-tuned", "#CC79A7")]
    rng = np.random.default_rng(1)
    for i, (m, col) in enumerate(meths):
        v = d[m].to_numpy()
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=col, alpha=0.6, lw=0)
        mean, lo, hi = boot(v)
        ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color="black", ms=4, capsize=2)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(range(len(meths)), ["Bandpass\n1–10 Hz", "f-k\n1000 m/s", "seisDAE\nDAS-trained", "jDAS\nfine-tuned"])
    ax.set_ylabel("SI-SDR of output vs clean input (dB)")
    ax.set_title("Clean test earthquakes, no noise added (19)")
    panel_letter(ax, 0)
    ax = axes[1]
    x = np.arange(len(c))
    ax.bar(x - 0.2, c.coherence_without_cmr, 0.38, color="#0072B2", label="without common-mode removal")
    ax.bar(x + 0.2, c.coherence_with_cmr, 0.38, color="#D55E00", label="with (our preprocessing)")
    ax.set_xticks(x, [f"{100 * v:.1f}%" for v in c.energy_removed_by_cmr], fontsize=7)
    ax.set_xlabel("Event-window energy removed by common-mode removal")
    ax.set_ylabel("Neighbour coherence")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("6 test events downloaded again")
    panel_letter(ax, 1)
    fig.tight_layout()
    save(fig, "fig08_jdas_diagnosis")


# ---------------------------------------------------------------- figures from data and models
def example_event(fam):
    g = fam["2-D U-Net v1 (3 seeds)"].xs(0, level="level").sort_values()
    return g.index[len(g) // 2]


def load_methods(dx):
    from . import jdas, unet2d
    from .methods import ROOT, SeisDAE, bandpass, fk

    ft = ROOT / "models/variants/finetune_s2026"
    net = {
        "Bandpass 0.5–45 Hz": lambda x: bandpass(x, (0.5, 45)),
        "f-k 250 m/s": lambda x: fk(x, dx, 250),
        "seisDAE released": SeisDAE(ROOT / "models/gr_mixed_stft.h5", ROOT / "models/gr_mixed_stft.config"),
        "seisDAE fine-tuned": SeisDAE(ft.with_suffix(".h5"), ft.with_suffix(".config")),
        "jDAS fine-tuned": jdas.JDAS(jdas.RUNS / "finetune_s2026.h5"),
        "2-D U-Net (ours)": unet2d.UNet2D(unet2d.RUNS / "s2026.h5"),
    }
    return net


def gather_panel(ax, g, title, vmax, t0=5, t1=35):
    t = np.arange(g.shape[1]) / 100
    ax.imshow(g, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[t[0], t[-1], g.shape[0], 0],
              interpolation="none")
    ax.set_xlim(t0, t1)
    ax.set_title(title, fontsize=8)
    ax.grid(False)


def fig_data(fam):
    from .build import load_gather, manifest
    from .das import geometry
    from .evaluate import pairs, original
    from .methods import bandpass, fk, mix

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    dx = geometry()[2]
    nets = load_methods(dx)
    m = original(manifest())
    ev_id = example_event(fam)
    r = m[m.id == ev_id].iloc[0]
    nz = m[(m.kind == "noise") & (m.split == "test")]
    ev = m[(m.kind == "strong") & (m.split == "test")]
    job = [j for j in pairs(ev, nz, E["noise_draws"], CFG["seed"] + 20) if j[0].id == ev_id][0]
    s, n = load_gather(ev_id), load_gather(job[1])
    x = mix(s, n, 0)
    outs = {name: f(x) for name, f in nets.items()}
    v = np.percentile(np.abs(s), 99)

    # fig09: gathers
    panels = [("Reference (recorded)", s), ("Input: + noise at 0 dB", x)] + list(outs.items())
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.6), sharex=True, sharey=True)
    for ax, (title, g) in zip(axes.flat, panels):
        gather_panel(ax, g, title, v)
    for ax in axes[:, 0]:
        ax.set_ylabel("Channel")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"Test earthquake {ev_id} (M{r.magnitude:.1f}, {r.dist_km:.0f} km), "
                 f"median U-Net event; same colour scale", fontsize=9)
    fig.tight_layout()
    save(fig, "fig09_gathers_synthetic")

    # fig10: one channel
    ch = s.shape[0] // 2
    t = np.arange(s.shape[1]) / 100
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.2), sharex=True)
    for ax, (name, col) in zip(axes, (("Bandpass 0.5–45 Hz", "#555555"), ("seisDAE fine-tuned", "#0072B2"),
                                      ("2-D U-Net (ours)", "#D55E00"))):
        ax.plot(t, x[ch], color="#CCCCCC", lw=0.6, label="input (0 dB)")
        ax.plot(t, s[ch], color="black", lw=0.8, label="reference")
        ax.plot(t, outs[name][ch], color=col, lw=0.8, label=name)
        ax.set_xlim(8, 30)
        ax.legend(fontsize=6.5, loc="upper right", ncol=3)
        ax.set_ylabel("Amplitude (a.u.)")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Middle channel ({ch}) of {ev_id}", fontsize=9)
    fig.tight_layout()
    save(fig, "fig10_trace")

    # fig11: spectra over all 19 test references at 0 dB (first noise draw)
    from scipy.signal import welch

    spec = {k: [] for k in ["Reference", "Input (0 dB)"] + list(nets)}
    for k, (rr, nid, draw) in enumerate(pairs(ev, nz, E["noise_draws"], CFG["seed"] + 20)):
        if draw != 0:
            continue
        s2, n2 = load_gather(rr.id), load_gather(nid)
        x2 = mix(s2, n2, 0)
        sl = slice(900, 3000)
        for name, g in [("Reference", s2), ("Input (0 dB)", x2)] + [(nm, f(x2)) for nm, f in nets.items()]:
            f_, p = welch(g[:, sl], fs=100, nperseg=256, axis=-1)
            spec[name].append(p.mean(axis=0) / np.sum(s2[:, sl] ** 2))
        print(f"  spectra {k + 1}", end="\r")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    cols = {"Reference": ("black", "-"), "Input (0 dB)": ("#BBBBBB", "-"), "Bandpass 0.5–45 Hz": ("#555555", "--"),
            "f-k 250 m/s": ("#009E73", "-"), "seisDAE released": ("#B07A00", "--"),
            "seisDAE fine-tuned": ("#0072B2", "-"), "jDAS fine-tuned": ("#CC79A7", "-"),
            "2-D U-Net (ours)": ("#D55E00", "-")}
    for name, ps in spec.items():
        c, ls = cols[name]
        ax.loglog(f_[1:], np.mean(ps, axis=0)[1:], color=c, ls=ls, lw=1.4 if name != "Reference" else 1.8, label=name)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean power (normalised)")
    ax.set_xlim(0.4, 50)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig11_spectra")

    # fig12: f-k spectra
    def fk_spec(g):
        F = np.fft.fftshift(np.abs(np.fft.fft2(g[:, 900:3000])), axes=0)
        return 20 * np.log10(F[:, : F.shape[1] // 2] + 1e-12)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
    nk = s.shape[0]
    kax = np.fft.fftshift(np.fft.fftfreq(nk, dx))
    fax = np.fft.fftfreq(2100, 0.01)[:1050]
    ref = fk_spec(s)
    top = ref.max()
    for ax, (title, g) in zip(axes, [("Reference", s), ("Input (0 dB)", x), ("Bandpass 0.5–45 Hz", outs["Bandpass 0.5–45 Hz"]),
                                     ("2-D U-Net (ours)", outs["2-D U-Net (ours)"])]):
        ax.imshow(fk_spec(g).T, origin="lower", aspect="auto", cmap="magma", vmin=top - 60, vmax=top,
                  extent=[kax[0], kax[-1], fax[0], fax[-1]])
        for vel in (250, 1000):
            kk = np.linspace(kax[0], kax[-1], 50)
            ax.plot(kk, np.abs(kk) * vel, color="white", lw=0.6, ls="--")
        ax.set_ylim(0, 25)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Wavenumber (1/m)")
        ax.grid(False)
    axes[0].set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    save(fig, "fig12_fk_spectra")

    # fig13: recorded test windows without mixing: an earthquake and a noise window
    det_eq = m[(m.split == "test") & (m.kind == "strong")].sort_values("magnitude").iloc[len(ev) // 2]
    noise_id = nz.id.iloc[0]
    rows = [(f"Earthquake {det_eq.id} (M{det_eq.magnitude:.1f})", load_gather(det_eq.id)),
            (f"Noise window {noise_id}", load_gather(noise_id))]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for i, (title, g) in enumerate(rows):
        vv = np.percentile(np.abs(g), 99)
        for j, (name, out) in enumerate([("Recorded", g), ("Bandpass 1–10 Hz", bandpass(g, (1, 10))),
                                         ("seisDAE fine-tuned", nets["seisDAE fine-tuned"](g)),
                                         ("2-D U-Net (ours)", nets["2-D U-Net (ours)"](g))]):
            gather_panel(axes[i, j], out, name if i == 0 else "", vv, t0=0, t1=60)
        axes[i, 0].set_ylabel(f"{title}\nChannel", fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, "fig13_field_windows")


def fig_timefreq():
    """Figures in the style of Steinberg et al. (2025): waveform + spectrogram, and the continuous SNR."""
    from scipy.signal import spectrogram

    from . import build, unet2d
    from .build import load_gather, manifest
    from .das import geometry
    from .evaluate import original
    from .methods import ROOT, SeisDAE, bandpass
    from .onset_snr import continuous_snr

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    m = original(manifest())
    ev = m[(m.kind == "strong") & (m.split == "test")].sort_values("magnitude")
    ft = ROOT / "models/variants/finetune_s2026"
    nets = {"Recorded": lambda x: x, "Bandpass 1–10 Hz": lambda x: bandpass(x, (1, 10)),
            "seisDAE fine-tuned": SeisDAE(ft.with_suffix(".h5"), ft.with_suffix(".config")),
            "2-D U-Net v1": unet2d.UNet2D(unet2d.RUNS / "s2026.h5")}
    if (unet2d.RUNS / "v4_s2026.h5").exists():
        nets["2-D U-Net v4 (traffic-free targets)"] = unet2d.UNet2D(unet2d.RUNS / "v4_s2026.h5")

    # fig14: one event, the channel with the highest recorded SNR
    r = ev.iloc[len(ev) // 2]
    g = load_gather(r.id)
    ch = int(np.argmax(build.selection_snr(r.id)))
    outs = {k: f(g)[ch] for k, f in nets.items()}
    t = np.arange(g.shape[1]) / 100
    fig, axes = plt.subplots(len(outs), 2, figsize=(7.2, 1.45 * len(outs) + 0.6), sharex=True,
                             gridspec_kw=dict(width_ratios=[1, 1.15]))
    _, _, ref = spectrogram(outs["Recorded"], fs=100, nperseg=128, noverlap=120)
    top = 10 * np.log10(ref.max())
    for i, (name, y) in enumerate(outs.items()):
        ax = axes[i, 0]
        ax.plot(t, y / np.abs(y).max(), color="black", lw=0.5)
        ax.axvline(r.itp / 100, color="#D55E00", lw=0.8, ls="--")
        ax.axvline(r.its / 100, color="#0072B2", lw=0.8, ls="--")
        ax.set_ylabel(name, fontsize=7)
        ax.set_yticks([])
        ax.set_xlim(5, 40)
        ff, tt, S = spectrogram(y, fs=100, nperseg=128, noverlap=120)
        im = axes[i, 1].pcolormesh(tt, ff, 10 * np.log10(S + 1e-30), vmin=top - 60, vmax=top, cmap="magma",
                                   shading="auto")
        axes[i, 1].set_ylim(0, 40)
        axes[i, 1].set_ylabel("Hz", fontsize=7)
        axes[i, 1].grid(False)
    fig.suptitle(f"{r.id} (M{r.magnitude:.1f}, {r.dist_km:.0f} km), channel {ch}. Left: waveform, each normalised; "
                 "dashed: predicted P (orange) and S (blue). Right: spectrogram, same dB scale", fontsize=7.5)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.colorbar(im, ax=axes[:, 1], shrink=0.6, label="Power (dB)")
    save(fig, "fig14_waveform_spectrogram")

    # fig15: the six largest clean test earthquakes, continuous SNR (as Steinberg et al. 2025, Fig. 7)
    big = ev.sort_values("magnitude", ascending=False).head(6)
    show = {"Recorded": "#D55E00", "Bandpass 1–10 Hz": "#8E44AD", "2-D U-Net v1": "black"}
    if "2-D U-Net v4 (traffic-free targets)" in nets:
        show["2-D U-Net v4 (traffic-free targets)"] = "#0072B2"
    fig, axes = plt.subplots(len(big), 1, figsize=(7.2, 1.25 * len(big) + 0.5), sharex=True)
    for ax, r in zip(axes, big.itertuples()):
        g = load_gather(r.id)
        ch = int(np.argmax(build.selection_snr(r.id)))
        for k, (name, col) in enumerate(show.items()):
            y = nets[name](g)[ch]
            sl = slice(int(r.itp) - 300, int(r.itp) + 600)
            tt = t[sl] - r.itp / 100
            ax.plot(tt, y[sl] / np.abs(y[sl]).max() * 0.45 + 0, color=col, lw=0.6, alpha=0.9,
                    label=name if r.Index == big.index[0] else None)
            if name != "Recorded":
                snr = continuous_snr(y[None])[0, sl]
                ax.plot(tt, snr / np.nanmax(snr) * 0.45 - 1.0, color=col, lw=1.0, ls="-")
        ax.text(-2.95, 0.55, f"{r.id}  M{r.magnitude:.1f}  {r.dist_km:.0f} km  ch {ch}", fontsize=6.5)
        ax.set_yticks([0, -0.78], ["waveform", "SNR(t)"], fontsize=6.5)
        ax.axvline(0, color="#999999", lw=0.6)
        ax.set_ylim(-1.1, 0.7)
        ax.grid(False)
    axes[-1].set_xlabel("Time after predicted P (s)")
    fig.legend(loc="upper center", ncol=len(show), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig15_continuous_snr")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", action="store_true", help="also the figures that need data and models")
    p.add_argument("--timefreq", action="store_true", help="only the waveform/spectrogram and continuous-SNR figures")
    a = p.parse_args()
    print("Figures in results/figures/paper/:")
    if a.timefreq:
        fig_timefreq()
        return
    fam = per_event_families()
    fig_map()
    fig_public()
    fig_synthetic(fam)
    fig_paired(fam)
    fig_detection()
    fig_magnitude()
    fig_jdas_diag()
    fig_training()
    if a.data:
        fig_data(fam)
        fig_timefreq()


if __name__ == "__main__":
    main()
