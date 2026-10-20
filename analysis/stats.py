import numpy as np, pandas as pd, glob, json
NANMOD=False
rng=np.random.default_rng(42); B=3000
def ld(k):
    d=f"r{k}/"; g=lambda s: glob.glob(d+s)[0]
    y=np.load(g("*_oof_y.npy")); p=np.load(g("*_oof_probs_tta.npy")); f=pd.read_csv(g("*_oof_files.csv"))
    return pd.DataFrame({"fp":f.filepath,"y":y,"pred":p.argmax(1)}), p
def f1(yy,pp):
    cm=np.bincount(yy*4+pp,minlength=16).reshape(4,4); tp=np.diag(cm); P=cm.sum(0); R=cm.sum(1)
    m=R>0; return (np.mean((2*tp/np.maximum(P+R,1))[m])*100 if R[3]>0 else np.nan) if NANMOD else np.mean((2*tp/np.maximum(P+R,1))[m])*100
def met(d): return dict(acc=(d.y==d.pred).mean()*100,f1=f1(d.y.values,d.pred.values),
    rec=[float((d.pred[d.y==c]==c).mean()*100) for c in range(4) if (d.y==c).any()])
grp=pd.read_csv(glob.glob("r10b/*_folds.csv")[0]).set_index("filepath").group
_o=pd.read_csv(glob.glob("r10d/*_oof_files.csv")[0]); og=_o.set_index("filepath").subject
R={}
cfg={"R50_random":("10a",grp),"R50_tau095":("10g",grp),"R50_tau090":("10h",grp),"R50_scan":("10b",grp),
 "Fus_random":("10e",grp),"Fus_scan":("10f",grp),"OAS_random":("10c",og),"OAS_subject":("10d",og)}
D={}
for n,(k,gm) in cfg.items():
    d,p=ld(k); d["g"]=d.fp.map(gm); 
    if d.g.isna().any():
        print(n,"unmapped",d.g.isna().sum()); 
    D[n]=d.set_index("fp")
    R[n]=met(d)
    from sklearn.metrics import roc_auc_score
    try: R[n]["auc"]=roc_auc_score(d.y,p,multi_class="ovr",average="macro") if p.shape[1]==4 and len(set(d.y))==4 else None
    except Exception as e: R[n]["auc"]=None
class A_:
    def __init__(s,y,p): s.y=y; s.pred=p
def boot(names,fn):
    base=D[names[0]]; gs=base.g.values; ug,inv=np.unique(gs,return_inverse=True)
    idx=[np.where(inv==i)[0] for i in range(len(ug))]; out=[]
    arr=[(D[n].loc[base.index].y.values,D[n].loc[base.index].pred.values) for n in names]
    for _ in range(B):
        s=np.concatenate([idx[i] for i in rng.integers(0,len(ug),len(ug))])
        out.append(fn([A_(a[0][s],a[1][s]) for a in arr]))
    return np.array(out)
def ci(v): return [float(np.nanpercentile(v,2.5)),float(np.nanpercentile(v,97.5))]
NANMOD=True
for n in D:
    A=boot([n],lambda a:[(a[0].y==a[0].pred).mean()*100,f1(a[0].y,a[0].pred)])
    R[n]["acc_ci"]=ci(A[:,0]); R[n]["f1_ci"]=ci(A[:,1])
def diff(a,b,key):
    global NANMOD
    fn=lambda x:[ (x[0].y==x[0].pred).mean()*100-(x[1].y==x[1].pred).mean()*100, f1(x[0].y,x[0].pred)-f1(x[1].y,x[1].pred)]
    NANMOD=True; v=boot([a,b],fn); NANMOD=False; m=fn([A_(D[a].y.values,D[a].pred.values),A_(D[b].loc[D[a].index].y.values,D[b].loc[D[a].index].pred.values)])
    R[key]=dict(acc=float(m[0]),acc_ci=ci(v[:,0]),f1=float(m[1]),f1_ci=ci(v[:,1]))
for a,b,k in [("R50_random","R50_scan","d_R50_rand_scan"),("R50_random","R50_tau095","d_R50_rand_095"),("R50_tau095","R50_tau090","d_095_090"),
  ("R50_tau090","R50_scan","d_090_scan"),("Fus_random","R50_random","d_fus_r50_rand"),("Fus_scan","R50_scan","d_fus_r50_scan"),
  ("Fus_random","Fus_scan","d_Fus_rand_scan"),("OAS_random","OAS_subject","d_OAS")]:
    diff(a,b,k)
# majority baselines & counts
R["majority_kaggle"]=float((D["R50_scan"].y==0).mean()*100)
R["majority_oasis"]=float((D["OAS_subject"].y==0).mean()*100)
R["oasis_n"]=int(len(D["OAS_subject"])); R["oasis_subjects"]=int(D["OAS_subject"].g.nunique())
R["oasis_class_counts"]=[int(x) for x in np.bincount(D["OAS_subject"].y,minlength=4)]
# subjects per class
o=D["OAS_subject"]; R["oasis_subj_per_class"]=[int(o[o.y==c].g.nunique()) for c in range(4)]
# per-group accuracy
d=D["R50_scan"]; ga=d.groupby("g").apply(lambda x:(x.y==x.pred).mean()); gn=d.groupby("g").size()
big=ga[gn>=20]; R["groups_ge20"]=int(len(big)); R["g_hi90"]=float((big>0.9).mean()*100); R["g_lo10"]=float((big<0.1).mean()*100)
fold=pd.read_csv(glob.glob("r10b/*_per_fold_results.csv")[0]); R["fold_scan_acc"]=fold.to_dict("list")
json.dump(R,open("final_stats.json","w"),indent=1,default=float)
for k,v in R.items():
    if isinstance(v,dict) and "acc" in v: print(k,{a:(np.round(b,1) if not isinstance(b,list) else np.round(b,1).tolist()) for a,b in v.items() if a!='auc'}, 'auc',v.get('auc'))
print({k:R[k] for k in ["majority_kaggle","majority_oasis","oasis_n","oasis_subjects","oasis_class_counts","oasis_subj_per_class","groups_ge20","g_hi90","g_lo10"]})
