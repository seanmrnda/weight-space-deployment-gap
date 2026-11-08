"""Post-hoc paired fixed/broken counts and two-sided exact sign tests on legacy workflow slices against L. Run from the repository root."""
import sys, json, math
sys.path.insert(0,'research/weight-space/scripts')
import p2048_weight_space_deploy as d
arms={r['id']:r for r in json.load(open(d.RUNS/'soup/recipes/lockbox.json'))}
rows=d.rows_for('legacy','legacy')
P={n:d.logits('legacy',r) for n,r in arms.items()}
def binom(b,c):
    n=b+c; k=min(b,c)
    return min(1,2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1
sec=lambda r:r['source_family']=='security_incidents' and r['type']=='noul' and int(r['hard_label_index'])==1
out={}
for n in arms:
    if n=='L': continue
    out[n]={}
    for w in d.WORKFLOWS+('secpos',):
        sub=[r for r in rows if (sec(r) if w=='secpos' else r['source_family']==w)]
        b=sum(d.correct(r,P[n][r['id']]) and not d.correct(r,P['L'][r['id']]) for r in sub)
        c=sum(not d.correct(r,P[n][r['id']]) and d.correct(r,P['L'][r['id']]) for r in sub)
        out[n][w]=dict(fixed=b,broken=c,p=round(binom(b,c),4))
    wf=[r for r in rows if r['source_family'] in d.WORKFLOWS]
    b=sum(d.correct(r,P[n][r['id']]) and not d.correct(r,P['L'][r['id']]) for r in wf)
    c=sum(not d.correct(r,P[n][r['id']]) and d.correct(r,P['L'][r['id']]) for r in wf)
    out[n]['all_workflows']=dict(fixed=b,broken=c,p=binom(b,c))
json.dump(out,open('research/weight-space/runs/external/legacy-workflow-paired.json','w'),indent=1)
for n,v in out.items(): print(n, v)
