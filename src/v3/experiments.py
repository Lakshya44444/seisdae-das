"""Frozen classical controls, synthetic metric checks, and family sensitivity."""
import json
import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter,sosfiltfilt
from scipy.sparse.csgraph import connected_components
from ..data import ROOT,SEED,LEVELS,load,mix
from ..metrics import fk
from ..v2.metrics import all_scores,arrival,si_sdr
from ..v2.stats import aggregate
from ..v2.baseline import MEASURES
OUT=ROOT/"results_v3"
CFG=json.loads((ROOT/"src/v3/protocol.json").read_text())
def context(x):
    a=(6001-x.shape[-1])//2
    return np.pad(x,((0,0),(a,6001-x.shape[-1]-a))),slice(a,a+x.shape[-1])
def classical(x,method,setting):
    full,mid=context(x)
    v=json.loads(setting)
    if method=="Bandpass":
        y=sosfiltfilt(butter(3,v,fs=100,btype="bandpass",output="sos"),full)
    elif method=="f-k":y=fk(full,vmin=v)
    elif method=="Wavelet":
        co=pywt.wavedec(full,"db8",level=2,mode="symmetric")
        # Estimate noise from the actual center; zero background would yield a vacuous MAD.
        hi=pywt.wavedec(x,"db8",level=2)[-1]
        mad=np.median(abs(hi-np.median(hi,axis=1,keepdims=True)),axis=1,keepdims=True)/.67448975
        thr=v*mad*np.sqrt(2*np.log(x.shape[-1]))
        y=pywt.waverec([co[0],*[np.sign(a)*np.maximum(abs(a)-thr,0) for a in co[1:]]],"db8")
    elif method=="SVD":
        # Adding zero columns only adds zero right-singular-vector support.
        u,d,w=np.linalg.svd(x,full_matrices=False)
        return (u[:,:v]*d[:v])@w[:v]
    else:return np.broadcast_to(np.median(x,axis=0),x.shape)
    return y[:,mid]
def run_classical():
    ev,p=load(ROOT/"data/prepared");grid=CFG["classical"]
    choices={"Bandpass":grid["bandpass_hz"],"f-k":grid["f_k_velocity"],"Wavelet":grid["wavelet_threshold_multiplier"],"SVD":grid["svd_rank"],"Median stack":[None]}
    def evaluate(sub,settings):
        rows=[]
        for i,e in sub.iterrows():
            s,n,q=(p[k][i] for k in ("reference","noise","nominal_p"))
            for level in LEVELS:
                x=mix(s,n,level)
                for method,values in settings.items():
                    for setting in values:
                        token=json.dumps(setting)
                        rows.append(dict(event_id=e.event_id,array=e["array"],level=level,method=method,setting=token,**all_scores(s,x,classical(x,method,token),q)))
        return pd.DataFrame(rows)
    dev=evaluate(ev[ev.split.isin(["train","validation"])],choices)
    scores=dev.groupby(["method","setting"],sort=False,as_index=False)[["gain_db","si_gain_db"]].mean()
    scores.to_csv(OUT/"classical_grid.csv",index=False)
    selected=scores.loc[scores.groupby("method").gain_db.idxmax()]
    selected.to_csv(OUT/"classical_selection.csv",index=False)
    d=evaluate(ev[ev.split=="test"],{r.method:[json.loads(r.setting)] for r in selected.itertuples()})
    d.to_csv(OUT/"classical_events.csv",index=False)
    aggregate(d,["method","level"],MEASURES).to_csv(OUT/"classical_summary.csv",index=False)
def clusters(meta,days=2,km=10):
    times=pd.to_datetime(meta.event_time,format="mixed",utc=True).astype("int64").to_numpy()/86400e9
    lat,lon=np.deg2rad(meta.latitude.to_numpy()),np.deg2rad(meta.longitude.to_numpy())
    h=np.sin((lat[:,None]-lat)/2)**2+np.cos(lat[:,None])*np.cos(lat)*np.sin((lon[:,None]-lon)/2)**2
    dist=12742*np.arcsin(np.sqrt(np.clip(h,0,1)))
    site=meta["array"].to_numpy()
    edge=(abs(times[:,None]-times)<=days)&(dist<=km)&(site[:,None]==site)
    return connected_components(edge,directed=False)[1]
def bootstrap(d,by_cluster):
    rng=np.random.default_rng(SEED);means=d.groupby("array").size()/len(d)
    boot=np.zeros(5000)
    for site,g in d.groupby("array"):
        groups=g.groupby("cluster" if by_cluster else "event_id").delta.agg(["sum","count"])
        idx=rng.integers(len(groups),size=(5000,len(groups)))
        boot+=means[site]*groups["sum"].to_numpy()[idx].sum(1)/groups["count"].to_numpy()[idx].sum(1)
    return np.quantile(boot,[.025,.975])
def validity():
    t=np.arange(70);s=(np.exp(-((t-21)/4)**2)-.65*np.exp(-((t-27)/4)**2))[None,:]
    rows=[]
    for name,y in [("perfect",s),("sign flip",-s),("scale 0.5",.5*s),("shift 5 samples",np.roll(s,5,axis=-1))]:
        a=arrival(s,y,[14]);rows.append(dict(case=name,si_sdr_db=si_sdr(s,y)[0],peak_ratio=a["peak_ratio"],sign=a["polarity_hits"]/a["polarity_total"],cc=np.corrcoef(s[0],y[0])[0,1]))
    pd.DataFrame(rows).to_csv(OUT/"metric_checks.csv",index=False)
    ev=pd.read_csv(ROOT/"data/prepared/events.csv");ev=ev[ev.split=="test"].copy()
    ev["cluster"]=clusters(ev);ev[["event_id","array","cluster"]].to_csv(OUT/"clusters.csv",index=False)
    old=pd.read_csv(ROOT/"results_v2/baseline_events.csv");rows=[]
    for lv,g in old.groupby("level"):
        for metric in ["gain_db","si_gain_db"]:
            wide=g.pivot(index="event_id",columns="method",values=metric)
            d=ev.merge((wide["Fine-tuned"]-wide["Zero-shot"]).rename("delta"),on="event_id")
            for name in ("event","cluster"):
                lo,hi=bootstrap(d,name=="cluster")
                rows.append(dict(method="Head minus released",level=lv,metric=metric,resampling=name,events=len(d),clusters=d.cluster.nunique(),mean=d.delta.mean(),lo=lo,hi=hi))
    pd.DataFrame(rows).to_csv(OUT/"cluster_intervals.csv",index=False)
if __name__=="__main__":
    validity();run_classical()
