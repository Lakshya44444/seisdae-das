"""Energy-supported masks and measured waveform changes on a prespecified example."""
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, welch
from .. import neural
from ..data import ROOT,load,mix
from ..v2.metrics import si_sdr

OUT=ROOT/"results_v3"
def run():
    ev,p=load(ROOT/"data/prepared")
    i=ev[ev.split=="test"].index[0]
    s,n=p["reference"][i],p["noise"][i]
    x=mix(s,n,0);net=neural.model();rows=[];sam=[]
    for f in [1,4]:
        expanded=x if f==1 else resample_poly(x, f,1,axis=-1)
        features,z,start=neural.encode(expanded)
        mask=np.concatenate([net(a,training=False).numpy()[...,0] for a in np.array_split(features,3)])
        y=neural.decode(z,mask,start,expanded.shape[-1])
        if f>1:y=resample_poly(y,1,f,axis=-1)
        power=abs(z)**2
        weighted=(power*mask).sum()/power.sum()
        rows.append(dict(event_id=ev.at[i,"event_id"],factor=f,weighted_mask=weighted,relative_change=np.linalg.norm(y-x)/np.linalg.norm(x),gain_db=10*np.log10(np.sum((x-s)**2)/np.sum((y-s)**2)),si_gain_db=np.mean(si_sdr(s,y)-si_sdr(s,x))))
        active=np.flatnonzero(power.sum((0,1))>0)
        sam.append((f,mask[:,:,active].mean(0),active))
    pd.DataFrame(rows).to_csv(OUT/"mechanism.csv",index=False)
    np.savez_compressed(OUT/"mask_example.npz",**{f"mask{f}":m for f,m,a in sam},**{f"frame{f}":a for f,m,a in sam})
    station=np.load(ROOT/"data/example/bug2019gbbo_Z.npz")["data"][:neural.N]
    station_out=neural.denoise(net,station[None])[0]
    np.savez_compressed(OUT/"station.npz",raw=station-station.mean(),estimate=station_out)
    # Identical Welch window and frequency grid; report shapes, not cross-sensor amplitudes.
    traces=p["reference"][ev.split=="test"].reshape(-1,70)
    hz,ds=welch(traces,fs=100,nperseg=70,nfft=256,axis=-1)
    _,st=welch(station,fs=100,nperseg=70,nfft=256)
    ds=(ds/ds.sum(1,keepdims=True)).mean(0);st/=st.sum()
    pd.DataFrame(dict(hz=hz,das=ds,station=st)).to_csv(OUT/"spectra.csv",index=False)
if __name__=="__main__":run()
