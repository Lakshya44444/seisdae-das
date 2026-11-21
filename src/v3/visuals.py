"""Draw figures from event records, summary tables and measured waveforms."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..data import ROOT, load, mix

F = ROOT / "figures_v3"
INK, PAPER, SLATE, TEAL, AMBER = "#0E1A2B", "#F6F3EE", "#5B6B7C", "#087886", "#A95800"


def read(name, version=3):
    return pd.read_csv(ROOT / f"results_v{version}" / f"{name}.csv")


def panels(n=1, size=(8.6,3.5)):
    with plt.rc_context({"axes.spines.top":False,"axes.spines.right":False}):
        fig, axes = plt.subplots(1,n,figsize=size,squeeze=False,layout="constrained")
    fig.patch.set_facecolor(PAPER)
    for ax in axes.flat:
        ax.set_facecolor(PAPER)
        ax.tick_params(colors=SLATE)
    return fig, axes[0]


def save(fig, name):
    F.mkdir(exist_ok=True)
    fig.savefig(F / (name+".png"),dpi=300,facecolor=fig.get_facecolor())
    plt.close(fig)


def render():
    e,p = load(ROOT/"data/prepared")
    i = e.index[e.split=="test"][0]
    s,n = p["reference"][i],p["noise"][i]
    x = mix(s,n,0)
    raw = p["raw"][i]
    fig=plt.figure(figsize=(12.8,7.2))
    ax=fig.add_axes([0,0,1,1]); ax.axis("off");lim=np.quantile(abs(raw),.985)
    ax.imshow(raw,aspect="auto",cmap="PuBuGn",vmin=-lim,vmax=lim)
    ax.axvspan(-.5,raw.shape[1]-.5,color=INK,alpha=.65)
    fig.savefig(F/"hero.jpg",dpi=150);plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(8.8,2.8),sharex=True,layout="constrained")
    with np.load(ROOT/"results_v3/station.npz") as st:
        for ax,key,label in zip(axes,["raw","estimate"],["Recorded station waveform","Released model output"]):
            ax.plot(np.arange(len(st[key]))/100,st[key]/np.max(abs(st["raw"])),color=INK,lw=.65)
            ax.axvline(30,ls="--",color=AMBER);ax.set(ylabel=label.split()[0],ylim=(-1.05,1.05))
        axes[-1].set_xlabel("Time (s)")
    save(fig,"reproduction")

    scalar=pd.read_csv(ROOT/"results/scalar.csv").iloc[0,0]
    fig,axes=plt.subplots(2,1,figsize=(8.8,3.3),sharex=True,sharey=True,layout="constrained")
    for ax,y,title,c in zip(axes,[x[0],x[0]*scalar],["Noisy input","Scaled copy"],[INK,TEAL]):
        ax.plot(np.arange(70)*10,y,color=c,lw=2);ax.set_ylabel(title)
    axes[-1].set_xlabel("Time in scored clip (ms)");save(fig,"scalar")

    fig,(ax,)=panels(size=(8.8,3.0))
    mag=read("strata_paired",2).query("stratifier=='magnitude' and level==0")
    ax.bar(range(len(mag)),mag.delta_db,color=[SLATE,SLATE,TEAL])
    ax.errorbar(range(len(mag)),mag.delta_db,yerr=[mag.delta_db-mag.delta_db_lo,mag.delta_db_hi-mag.delta_db],fmt="none",ecolor=INK,capsize=4)
    ax.set(xticks=list(range(len(mag))),xticklabels=[f"{r.bin}\n(n={r.events})" for r in mag.itertuples()],ylabel="Head minus zero-shot gain (dB)")
    save(fig,"magnitude")

    base=read("baseline_summary",2)
    fig,(ax,)=panels(size=(8.6,4.2))
    ranking=base.query("level==0").copy()
    new=read("classical_summary").query("level==0 and method=='f-k'").copy();new["method"]="Tuned f-k"
    for j,r in enumerate(pd.concat([ranking,new]).itertuples()):
        c=TEAL if r.method=="Tuned f-k" else SLATE
        ax.plot([r.gain_db,r.si_gain_db],[j,j],color=c,alpha=.8)
        ax.scatter(r.gain_db,j,c=c,marker="o");ax.scatter(r.si_gain_db,j,c=c,marker="D")
    ax.axvline(0,color=SLATE,alpha=.3)
    ax.set(yticks=np.arange(len(ranking)+1),yticklabels=[*ranking.method,"Tuned f-k"],xlabel="Gain (dB)   • ordinary    ◆ SI-SDR")
    save(fig,"ranking")
    fig,(ax,)=panels()
    for method,c in [("Fine-tuned",TEAL),("f-k",SLATE),("Bandpass",AMBER)]:
        d=base[base.method==method].sort_values("level")
        ax.plot(d.level,d.si_gain_db,"o-",c=c);ax.text(10.4,d.si_gain_db.iloc[-1],method,color=c,va="center")
    ax.set(xlim=(-6,15),xticks=[-5,0,5,10],xlabel="Input SNR (dB)",ylabel="SI-SDR gain (dB)");save(fig,"levels")
    fig,axes=panels(2,(8.8,2.9))
    with np.load(ROOT/"results_v3/mask_example.npz") as data:
        for f,ax in zip([1,4],axes):
            frames=data[f"frame{f}"]
            im=ax.imshow(data[f"mask{f}"],origin="lower",aspect="auto",vmin=0,vmax=1,
                         extent=[frames.min()-.5,frames.max()+.5,0,50],cmap="viridis")
            ax.set(title=f"{f}× time scale",xlabel="STFT frame",ylabel="Frequency (Hz)")
    fig.colorbar(im,ax=list(axes),label="Signal mask");save(fig,"mask")


if __name__=="__main__":
    render()
