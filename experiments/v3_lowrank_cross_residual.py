"""Frozen low-rank cross residual adapters using PCA factors × asset exposures."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1]
for p in (str(ROOT),str(ROOT/'experiments')):
    if p not in sys.path: sys.path.insert(0,p)
from lgbm_xs import load_rows
from src.metric import scale_invariant_score, weighted_zero_mean_r2

def args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data-root',default=str(ROOT/'data')); p.add_argument('--oof',default=str(ROOT/'outputs/cache/v3_production_oof_confirm_3s480_phasebal_prodwindow.npz')); p.add_argument('--oof-report',default=str(ROOT/'outputs/experiments/v3_production_oof_confirm_3s480_phasebal_prodwindow.json')); p.add_argument('--output-dir',default=str(ROOT/'outputs/experiments')); p.add_argument('--label',default='v3_lowrank_cross_residual_3s480'); p.add_argument('--meta-fold',type=int,default=0); p.add_argument('--components',type=int,nargs='+',default=[4,8,16]); p.add_argument('--pca-rows',type=int,default=120000); p.add_argument('--ridge-alpha',type=float,default=1000.0); p.add_argument('--force',action='store_true'); return p.parse_args()
def sc(t): s=np.r_[0,np.flatnonzero(t[1:]!=t[:-1])+1]; c=np.diff(np.r_[s,len(t)]); return s,c
def gm(v,t): s,c=sc(t); return np.repeat(np.add.reduceat(v,s)/c,c)
def proj(v,t): return v-gm(v,t)
def design(f,aid,k):
 n=len(aid); rows=np.repeat(np.arange(n),k); cols=(aid[:,None]*k+np.arange(k)[None,:]).ravel(); return sparse.csr_matrix((f[:,:k].ravel(),(rows,cols)),shape=(n,(int(aid.max())+1)*k))
def paired(c,b):
 d=c-b; dr=np.delete(d,int(np.argmax(d))); base=b.mean(); return {'relative_gain':float(d.mean()/base),'positive_folds':int((d>0).sum()),'n_folds':len(d),'relative_gain_drop_best':float(dr.mean()/base),'pass':bool(d.mean()/base>=.01 and (d>0).sum()>=3 and dr.mean()>0),'per_fold_delta':d.tolist()}
def main():
 a=args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); jp=out/f'{a.label}.json'; mp=out/f'{a.label}.md'
 if not a.force and (jp.exists() or mp.exists()): raise SystemExit('exists; use --force')
 started=time.perf_counter(); data=load_rows(Path(a.data_root),5,'phase_balanced')
 report=json.load(open(a.oof_report)); selected=np.array(report['folds'][a.meta_fold]['xs_selected'],dtype=np.int64)
 with np.load(a.oof,allow_pickle=False) as d:
  valid=d['fold']>=0
  for n in ['target','weight','time_id','asset_id']:
   if not np.array_equal(d[n],data[n]): raise AssertionError(f'{n} misaligned')
  y=d['target'][valid].astype(float); w=np.maximum(d['weight'][valid].astype(float),0); t=d['time_id'][valid]; aid=d['asset_id'][valid].astype(np.int64); fold=d['fold'][valid]; market=d['market'][valid].astype(float); e=d['e_lgbm'][valid].astype(float); base=d['prediction_raw'][valid].astype(float)
 x=data['features'][valid][:,selected].astype(np.float32); np.nan_to_num(x,copy=False,nan=0.0,posinf=0.0,neginf=0.0); meta=fold==a.meta_fold
 mean=x[meta].mean(0); scale=x[meta].std(0); scale[scale<1e-8]=1
 rng=np.random.default_rng(2026); idx=np.flatnonzero(meta); idx=rng.choice(idx,size=min(a.pca_rows,len(idx)),replace=False)
 pca=PCA(n_components=max(a.components),svd_solver='randomized',random_state=2026); pca.fit((x[idx]-mean)/scale)
 factors=np.empty((len(x),max(a.components)),dtype=np.float32)
 for start in range(0,len(x),100000): factors[start:start+100000]=pca.transform((x[start:start+100000]-mean)/scale)
 del x,data
 target_cross=y-gm(y,t); residual=target_cross-e
 arms={'baseline':base}; coefs={}
 for k in a.components:
  dm=design(factors[meta],aid[meta],k); model=Ridge(alpha=a.ridge_alpha,fit_intercept=False,solver='lsqr',tol=1e-8,max_iter=2000); model.fit(dm,residual[meta],sample_weight=w[meta]); coefs[str(k)]=model.coef_.tolist(); pred=base.copy()
  for f in np.unique(fold):
   q=fold==f; correction=model.predict(design(factors[q],aid[q],k)); pred[q]=market[q]+e[q]+proj(correction,t[q])
  arms[f'factor_k{k}']=pred
 scales={n:float(scale_invariant_score(y[meta],p[meta],w[meta])['optimal_scale']) for n,p in arms.items()}; evals=[int(f) for f in np.unique(fold) if f!=a.meta_fold]; rows=[]
 for f in evals:
  q=fold==f; row={'fold':f,'arms':{}}
  for n,p in arms.items(): row['arms'][n]={'peak':float(scale_invariant_score(y[q],p[q],w[q])['peak']),'frozen_score':weighted_zero_mean_r2(y[q],p[q]*scales[n],w[q])}
  rows.append(row)
 bp=np.array([r['arms']['baseline']['peak'] for r in rows]); bf=np.array([r['arms']['baseline']['frozen_score'] for r in rows]); summary={}
 for n in arms:
  if n=='baseline': continue
  cp=np.array([r['arms'][n]['peak'] for r in rows]); cf=np.array([r['arms'][n]['frozen_score'] for r in rows]); summary[n]={'peak':paired(cp,bp),'frozen_score':paired(cf,bf)}
 payload={'experiment':'v3_lowrank_cross_residual','meta_fold':a.meta_fold,'selected_features':selected.tolist(),'config':vars(a),'explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'meta_scales':scales,'coefs':coefs,'folds':rows,'summary':summary,'elapsed_seconds':time.perf_counter()-started}; jp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 lines=['# v3 low-rank cross residual','',f'Meta fold `{a.meta_fold}`','', '| Arm | Peak gain | Positive | Drop-best | Gate | Frozen gain |','|---|---:|---:|---:|:---:|---:|']
 for n,v in summary.items(): lines.append(f"| `{n}` | {v['peak']['relative_gain']*100:+.2f}% | {v['peak']['positive_folds']}/{v['peak']['n_folds']} | {v['peak']['relative_gain_drop_best']*100:+.2f}% | {'PASS' if v['peak']['pass'] else 'FAIL'} | {v['frozen_score']['relative_gain']*100:+.2f}% |")
 mp.write_text('\n'.join(lines)+'\n'); print('\n'.join(lines)); print(f"elapsed={payload['elapsed_seconds']:.1f}s")
if __name__=='__main__': main()
