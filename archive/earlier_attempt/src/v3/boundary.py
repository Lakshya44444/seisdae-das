"""Fixed passband boundary comparison on development data only."""
import pandas as pd
from ..data import ROOT,LEVELS,load,mix
from ..metrics import bandpass
from ..v2.metrics import all_scores
from .experiments import classical,OUT
def run():
    e,p=load(ROOT/"data/prepared");rows=[]
    for i,r in e[e.split.isin(["train","validation"])].iterrows():
        s,n,q=[p[k][i] for k in ["reference","noise","nominal_p"]]
        for lv in LEVELS:
            x=mix(s,n,lv)
            for context,y in [("short",bandpass(x)),("zero padded",classical(x,"Bandpass","[1,10]"))]:
                rows.append(dict(event_id=r.event_id,level=lv,context=context,**all_scores(s,x,y,q)))
    pd.DataFrame(rows).groupby(["context","level"],as_index=False)[["gain_db","si_gain_db"]].mean().to_csv(OUT/"boundary.csv",index=False)
if __name__=="__main__":run()
