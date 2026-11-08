import json, collections, math
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from kvmeaning.model import MODELS
from kvmeaning import ambig as A
s=MODELS["Q3-4B"]; tok=AutoTokenizer.from_pretrained(s.repo, revision=s.revision)
ids=lambda t: tok(A.PRE+t, add_special_tokens=False)["input_ids"]
def lev(a,b):
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        p,d[0]=d[0],i
        for j,y in enumerate(b,1):
            p,d[j]=d[j],min(d[j]+1,d[j-1]+1,p+(x!=y))
    return d[-1]
def tail(a,b):
    n=0
    for x,y in zip(reversed(a),reversed(b)):
        if x!=y: break
        n+=1
    return n
items={it["id"]:it for it in A.load_items()}
R=sorted(Path("runs/ambig/main/items").glob("*.safetensors"))
pts=[]
for f in R:
    i=f.stem; it=items[i]; m=load_file(str(f)); n=load_file(f"runs/ambig/neutral/items/{i}.safetensors")
    def D(t,pair,r):
        return float(torch.tensor([t[f"dist.{c}.{o}.{pair}.{r}.l2"].float().mean() for c in "ABN" for o in ("AB","BA")]).median())
    E=lambda k,c: it[k][c]
    cand=[("neutral",n,"A_pre|nA_pre",E("expl","A"),E("expl_neutral","A")),("neutral",n,"B_pre|nB_pre",E("expl","B"),E("expl_neutral","B")),
          ("AB",m,"A_pre|B_pre",E("expl","A"),E("expl","B"))]
    if "expl_para" in it:
        cand+=[("para",m,"A_pre|pA_pre",E("expl","A"),E("expl_para","A")),("para",m,"B_pre|pB_pre",E("expl","B"),E("expl_para","B"))]
    for kind,t,pair,x,y in cand:
        a,b=ids(x),ids(y)
        pts.append(dict(item=i,type=it["type"],kind=kind,k=lev(a,b),tail=tail(a,b),v=D(t,pair,"v"),resid=D(t,pair,"resid"),k_pre=D(t,pair,"k_pre")))
json.dump(pts, open("runs/ambig/neutral/meaning_pts.json", "w"))
# bins by token edit distance
def b(k): return "1" if k<=1 else "2-3" if k<=3 else "4-6" if k<=6 else "7-10" if k<=10 else "11+"
print("# median v l2 by token edit distance (n)")
for kind in ("neutral","para","AB"):
    g=collections.defaultdict(list)
    for p in pts:
        if p["kind"]==kind: g[b(p["k"])].append(p["v"])
    print(f"  {kind:8s}", "  ".join(f"{k}: {torch.tensor(v).median():.4f} (n={len(v)})" for k,v in sorted(g.items(), key=lambda x: ["1","2-3","4-6","7-10","11+"].index(x[0]))))
# regression log v ~ log k + tail + meaning, with item fixed effect via within-item demeaning
import itertools
X=[];Y=[];grp=[]
for p in pts:
    X.append([math.log(max(p["k"],1)), p["tail"], 1.0 if p["kind"]=="AB" else 0.0]); Y.append(math.log(p["v"])); grp.append(p["item"])
X=torch.tensor(X); Y=torch.tensor(Y)
for g in set(grp):
    idx=torch.tensor([j for j,x in enumerate(grp) if x==g])
    X[idx]-=X[idx].mean(0); Y[idx]-=Y[idx].mean()
beta=torch.linalg.lstsq(X,Y.unsqueeze(1)).solution.squeeze()
res=Y-X@beta; dof=len(Y)-len(set(grp))-3
se=torch.sqrt((res@res/dof)*torch.diag(torch.linalg.inv(X.T@X)))
for nm,bb,ss in zip(("log token edits","tokens after last edit","meaning differs"),beta.tolist(),se.tolist()):
    print(f"  {nm:24s} {bb:+.3f} ± {ss:.3f}")
print("  meaning effect as ratio:", round(math.exp(beta[2].item()),2))
