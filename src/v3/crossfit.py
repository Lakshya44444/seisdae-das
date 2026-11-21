"""Separate five-fold population diagnostic; never used for fixed-test model selection."""
import numpy as np,pandas as pd
from scipy.special import expit
from ..data import ROOT,LEVELS,SEED,load,mix
from .. import neural
from ..v2.metrics import all_scores
from ..v2.stats import aggregate
from .experiments import OUT,CFG
from .multi_data import DEST
CACHE=ROOT/"runs/v3_crossfits"
def prepare(e,parts):
    import tf_keras as k
    net=neural.model();ex=k.Model(net.input,net.layers[-2].input);CACHE.mkdir(exist_ok=True,parents=True)
    for i,row in e.iterrows():
        path=CACHE/f"{row.event_id}.npz"
        if path.exists():continue
        heads=[];zs=[];targets=[]
        for p in parts:
            s,n=p["reference"][i],p["noise"][i]
            for level in LEVELS:
                x=mix(s,n,level);feats,z,_=neural.encode(x)
                h=np.concatenate([ex(a,training=False).numpy()[:,:,59:62] for a in np.array_split(feats,3)])
                _,sig,_=neural.encode(s);_,noise,_=neural.encode(x-s)
                a=abs(sig[:,::12,59:62]);b=abs(noise[:,::12,59:62])
                heads.append(h);zs.append(z[:,:,59:62]);targets.append(np.divide(a,a+b,out=np.full_like(a,.5),where=a+b>0))
        np.savez(path,h=heads,z=zs,target=targets)
    return net.layers[-2].get_weights()[0][0,0]
def run():
    e,p=load(ROOT/"data/prepared");two=dict(np.load(DEST/"aperture2.npz"));parts=[p,two];start=prepare(e,parts)
    bank=[];targets=[]
    for eid in e.event_id:
        with np.load(CACHE/f"{eid}.npz") as z:
            bank.append(z["h"][:,:,::12].reshape(-1,8));targets.append(z["target"].ravel())
    rng=np.random.default_rng(SEED);blocks=[np.array_split(rng.permutation(g.index),5) for _,g in e.groupby("array")]
    rows=[];members=[]
    for j in range(5):
        held=np.sort(np.concatenate([b[j] for b in blocks]))
        fit=[rng.permutation(np.concatenate([g for ix,g in enumerate(b) if ix!=j])) for b in blocks]
        for frac in CFG["crossfit"]["efficiency"]:
            ids=np.concatenate([g[:max(1,round(frac*len(g)))] for g in fit])
            members.extend(dict(fold=j+1,fraction=frac,role="fit",event_id=e.at[i,"event_id"]) for i in ids)
            members.extend(dict(fold=j+1,fraction=frac,role="score",event_id=e.at[i,"event_id"]) for i in held)
            x=np.concatenate([bank[i] for i in ids]);y=np.concatenate([targets[i] for i in ids]);w=(start[:,0]-start[:,1]).copy()
            for step in range(40):w-=.015*(x.T@(expit(x@w)-y)/len(x))
            for i in held:
                with np.load(CACHE/f"{e.at[i,'event_id']}.npz") as z:
                    h=z["h"];spectrum=z["z"]
                mask=expit(h@w);full=np.zeros((*mask.shape[:-1],121),dtype=np.complex64);full[...,59:62]=spectrum*mask
                estimates=[neural.decode(z,1.,2965,70) for z in full]
                for k,lv in enumerate(LEVELS):
                    vals=[]
                    for a,part in enumerate(parts):
                        s,n,q=[part[key][i] for key in ["reference","noise","nominal_p"]]
                        vals.append(all_scores(s,mix(s,n,lv),estimates[4*a+k],q))
                    rows.append(dict(event_id=e.at[i,"event_id"],array=e.at[i,"array"],fold=j+1,fraction=frac,n_fit=len(ids),level=lv,gain_db=np.mean([v["gain_db"] for v in vals]),si_gain_db=np.mean([v["si_gain_db"] for v in vals]),cc=np.mean([v["cc"] for v in vals])))
        print("All-event fold",j+1,flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/"crossfit_events.csv",index=False);pd.DataFrame(members).to_csv(OUT/"crossfit_roles.csv",index=False)
    aggregate(d,["fraction","level"],["gain_db","si_gain_db","cc"]).to_csv(OUT/"crossfit_summary.csv",index=False)
if __name__=="__main__":run()
