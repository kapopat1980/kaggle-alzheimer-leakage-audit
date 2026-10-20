import numpy as np, pandas as pd, glob
rng=np.random.default_rng(1)
def f1(y,p):
    cm=np.bincount(y*4+p,minlength=16).reshape(4,4); tp=np.diag(cm); P=cm.sum(0); R=cm.sum(1); m=R>0
    return np.mean((2*tp/np.maximum(P+R,1))[m])*100
def ld(k):
    g=lambda s: glob.glob(f"r{k}/"+s)[0]
    f=pd.read_csv(g("*_oof_files.csv")); y=np.load(g("*_oof_y.npy")); p=np.load(g("*_oof_probs_tta.npy"))
    return pd.DataFrame({"fp":f.filepath,"y":y}).assign(**{f"p{i}":p[:,i] for i in range(4)}).set_index("fp")
def gmap(k):
    if k in("10b","11b","10a","11a"): return pd.read_csv(glob.glob("r10b/*_folds.csv")[0]).set_index("filepath").group
    return None
pairs=[("10a","11a","Kaggle R50 random"),("10b","11b","Kaggle R50 scan-level")]
def stats(d,g,B=2000):
    pr=d[[f"p{i}" for i in range(4)]].values.argmax(1); y=d.y.values
    ug,inv=np.unique(g,return_inverse=True); idx=[np.where(inv==i)[0] for i in range(len(ug))]
    a=[];f=[]
    for _ in range(B):
        s=np.concatenate([idx[i] for i in rng.integers(0,len(ug),len(ug))]); a.append((y[s]==pr[s]).mean()*100); f.append(f1(y[s],pr[s]))
    rec=[(pr[y==c]==c).mean()*100 for c in range(4) if (y==c).any()]
    return (y==pr).mean()*100,np.percentile(a,[2.5,97.5]),f1(y,pr),np.percentile(f,[2.5,97.5]),np.round(rec)
for a,b,name in pairs+[("10c","11c","OASIS random"),("10d","11d","OASIS subject")]:
    A,Bd=ld(a),ld(b)
    if name.startswith("OASIS"):
        for nm,d,k in((a,A,a),(b,Bd,b)):
            sub=d.index.to_series().str.extract(r"(OAS1_\d+)")[0].values
            r=stats(d,sub); print(name,nm,len(d),"acc %.1f [%.1f %.1f] F1 %.1f [%.1f %.1f]"%(r[0],*r[1],r[2],*r[3]),r[4])
        continue
    g=gmap(a)
    for nm,d in((a,A),(b,Bd)):
        r=stats(d,d.index.map(g).values); print(name,nm,"acc %.1f [%.1f %.1f] F1 %.1f [%.1f %.1f]"%(r[0],*r[1],r[2],*r[3]),r[4])
    common=A.index.intersection(Bd.index); ens=(A.loc[common].drop(columns="y")+Bd.loc[common].drop(columns="y"))/2; ens["y"]=A.loc[common].y
    r=stats(ens,common.map(g).values); print(name,"2-seed prob average acc %.1f [%.1f %.1f] F1 %.1f"%(r[0],*r[1],r[2]),r[4])
    pa=A.loc[common][[f"p{i}" for i in range(4)]].values.argmax(1); pb=Bd.loc[common][[f"p{i}" for i in range(4)]].values.argmax(1)
    print("   agreement between seeds %.1f%%"%((pa==pb).mean()*100))
