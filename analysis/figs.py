import numpy as np, pandas as pd, glob, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
S=json.load(open("final_stats.json")); C1,C2,C3="#1f5f99","#d1495b","#555555"
# Fig2 dose-response
fig,ax=plt.subplots(figsize=(5.2,3.3))
xs=["Random\nfolds","Groups\nτ>0.95\n(4,541)","Groups\nτ>0.90\n(393)","Groups\nτ>0.80\n(213)"]
keys=["R50_random","R50_tau095","R50_tau090","R50_scan"]
y=[S[k]["acc"] for k in keys]; lo=[S[k]["acc"]-S[k]["acc_ci"][0] for k in keys]; hi=[S[k]["acc_ci"][1]-S[k]["acc"] for k in keys]
ax.errorbar(range(4),y,yerr=[lo,hi],fmt="o-",color=C1,capsize=3,label="ResNet50")
fx=[0,3]; fk=["Fus_random","Fus_scan"]
ax.errorbar([x+0.08 for x in fx],[S[k]["acc"] for k in fk],yerr=[[S[k]["acc"]-S[k]["acc_ci"][0] for k in fk],[S[k]["acc_ci"][1]-S[k]["acc"] for k in fk]],fmt="s",color=C2,capsize=3,label="EffResNet-FusionNet")
ax.axhline(50,ls="--",color=C3,lw=1); ax.text(1.5,45.5,"majority class 50.0%",ha="center",color=C3,fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels(xs); ax.set_ylabel("Out-of-fold accuracy (%)"); ax.set_ylim(40,100); ax.legend(frameon=False,loc="upper right")
fig.tight_layout(); fig.savefig("fig2_dose.png",dpi=300)
# Fig1 confusion
def cm(k):
    d=glob.glob(f"r{k}/*_oof_y.npy")[0]; y=np.load(d); p=np.load(glob.glob(f"r{k}/*_oof_probs_tta.npy")[0]).argmax(1)
    return np.bincount(y*4+p,minlength=16).reshape(4,4)
names=["Non","VeryMild","Mild","Moderate"]
fig,axs=plt.subplots(1,2,figsize=(7,3.2))
for ax,k,t in zip(axs,["10a","10b"],["Random folds","Scan-level grouped folds (τ>0.80)"]):
    m=cm(k); r=m/m.sum(1,keepdims=True)*100
    ax.imshow(r,cmap="Blues",vmin=0,vmax=100)
    for i in range(4):
        for j in range(4): ax.text(j,i,f"{r[i,j]:.0f}%\n({m[i,j]})",ha="center",va="center",fontsize=7,color="white" if r[i,j]>55 else "black")
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(names,rotation=30,ha="right"); ax.set_yticklabels(names); ax.set_xlabel("Predicted"); ax.set_title(t,fontsize=9)
    ax.spines[:].set_visible(False)
axs[0].set_ylabel("True")
fig.tight_layout(); fig.savefig("fig1_cm.png",dpi=300)
# Fig3 group hist
g=pd.read_csv(glob.glob("r10b/*_folds.csv")[0]).set_index("filepath").group
f=pd.read_csv(glob.glob("r10b/*_oof_files.csv")[0]); yv=np.load(glob.glob("r10b/*_oof_y.npy")[0]); pv=np.load(glob.glob("r10b/*_oof_probs_tta.npy")[0]).argmax(1)
d=pd.DataFrame({"g":f.filepath.map(g),"ok":yv==pv}); ga=d.groupby("g").ok.agg(["mean","size"]); ga=ga[ga["size"]>=20]
fig,ax=plt.subplots(figsize=(4.2,2.8)); ax.hist(ga["mean"]*100,bins=10,range=(0,100),color=C1,edgecolor="white")
ax.set_xlabel("Accuracy within group (%)"); ax.set_ylabel("Number of groups"); fig.tight_layout(); fig.savefig("fig3_groups.png",dpi=300)
# Fig4 OASIS
fig,ax=plt.subplots(figsize=(4.6,3.1)); w=0.35
for i,(k,l,c) in enumerate([("OAS_random","Random folds",C1),("OAS_subject","Subject-grouped folds",C2)]):
    v=[S[k]["acc"],S[k]["f1"]]; e=[[S[k]["acc"]-S[k]["acc_ci"][0],S[k]["f1"]-S[k]["f1_ci"][0]],[S[k]["acc_ci"][1]-S[k]["acc"],S[k]["f1_ci"][1]-S[k]["f1"]]]
    ax.bar(np.arange(2)+(i-0.5)*w,v,w,yerr=e,capsize=3,color=c,label=l)
ax.axhline(S["majority_oasis"],ls="--",color=C3,lw=1); ax.text(1.55,S["majority_oasis"]+1.5,f"majority class {S['majority_oasis']:.1f}%",ha="right",color=C3,fontsize=8)
ax.set_xticks([0,1]); ax.set_xticklabels(["Accuracy","Macro-F1"]); ax.set_ylabel("%"); ax.set_ylim(0,100); ax.legend(frameon=False,loc="lower left",fontsize=8)
fig.tight_layout(); fig.savefig("fig4_oasis.png",dpi=300)
print(len(ga), (ga["mean"]>.9).mean(), (ga["mean"]<.1).mean())
