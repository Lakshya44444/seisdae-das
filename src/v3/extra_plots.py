import matplotlib.pyplot as plt
import numpy as np
from .visuals import panels,save,read,INK,SLATE
from .experiments import OUT
def render():
    fig,(ax,)=panels(size=(9.4,2.15));ax.axis("off")
    for j,(title,label) in enumerate([("Field clip","Select aperture"),("Partition","Arrival + early noise"),("Mix","Set energy ratio"),("Transform","Pad or resample"),("Recover","Predict and crop"),("Compare","Event-level scores")]):
        x=.01+j*.165;ax.text(x,.68,title,fontsize=14,color=INK,weight="bold",transform=ax.transAxes);ax.text(x,.40,label,fontsize=9.4,color=SLATE,transform=ax.transAxes)
        ax.plot([x,x+.13],[.54,.54],color="#087886",lw=2.2,transform=ax.transAxes)
    save(fig,"protocol")
    df=read("crossfit_summary",3).query("level==0");fig,(ax,)=panels(size=(8.6,2.7))
    for m,c,l in [("gain_db",SLATE,"Ordinary gain"),("si_gain_db","#087886","SI-SDR gain")]:
        ax.plot(100*df.fraction,df[m],"o-",color=c)
        ax.fill_between(100*df.fraction,df[m+"_lo"],df[m+"_hi"],color=c,alpha=.10)
        ax.text(102,df[m].iloc[-1],l,color=c,va="center")
    ax.set(xlim=(18,140),xlabel="Percentage of available outer-fold training events",ylabel="OOF gain (dB)",xticks=[25,50,100]);save(fig,"efficiency")
    d=read("spectra",3);ss=read("stead_spectrum",3);fig,(ax,)=panels(size=(8.8,2.7))
    for yy,lab,c in [(d.das,"DAS", "#087886"),(ss.pilot_stead,"STEAD pilot",INK),(d.station,"Station example",SLATE)]:
        ax.plot(d.hz,yy,color=c,lw=1.9);j=yy.idxmax();ax.annotate(lab,(d.hz[j],yy[j]),xytext=(15,2),textcoords="offset points",color=c,fontsize=10)
    ax.set(xlim=(0,25),xlabel="Frequency (Hz)",ylabel="Power fraction / bin");save(fig,"spectra")
if __name__=="__main__":render()
