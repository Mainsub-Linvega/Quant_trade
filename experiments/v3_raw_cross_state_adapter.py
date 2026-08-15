"""Screen causal raw-feature cross-sectional dispersion as a residual adapter state."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for p in (str(ROOT),str(ROOT/'experiments'),str(ROOT/'strategies'/'v1_ridge')):
    if p not in sys.path: sys.path.insert(0,p)
from lgbm_xs import load_rows
from src.metric import scale_invariant_score,weighted_zero_mean_r2

def args():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',default=str(ROOT/'data')); p.add_argument('--oof',default=str(ROOT/'outputs/cache/v3_production_oof_confirm_3s480_phasebal_prodwindow.npz')); p.add_argument('--oof-report',default=str(ROOT/'outputs/experiments/v3_production_oof_confirm_3s480_phasebal_prodwindow.json')); p.add_argument('--output-dir',default=str(ROOT/'outputs/experiments')); p.add_argument('--label',default='v3_raw_cross_state_adapter_3s480'); p.add_argument('--meta-fold',type=int,default=0); p.add_argument('--shrink',type=float,default=50); p.add_argument('--force',action='store_true'); return p.parse_args()
def sc(t): s=np.r_[0,np.flatnonzero(t[1:]!=t[:-1])+1]; c=np.diff(np.r_[s,len(t)]); return s,c
def gm(v,t): s,c=sc(t); return np.repeat(np.add.reduceat(v,s)/c,c)
def proj(v,t): return v-gm(v,t)
def slopes(target,e,w,aid,bins,sh):
 out=np.ones((int(aid.max())+1,2))
 for i in range(out.shape[0]):
  for b in range(2):
   m=(aid==i)&(bins==b); den=np.dot(w[m],e[m]**2); num=np.dot(w[m],e[m]*target[m]); out[i,b]=(num+sh)/(den+sh)
 return out
def paired(c,b):
 d=c-b; dr=np.delete(d,np.argmax(d)); return {'relative_gain':float(d.mean()/b.mean()),'positive_folds':int((d>0).sum()),'n_folds':len(d),'drop_best_relative_gain':float(dr.mean()/b.mean()),'pass':bool(d.mean()/b.mean()>=.01 and (d>0).sum()>=3 and dr.mean()>0)}
def main():
 a=args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); jp=out/f'{a.label}.json'; mp=out/f'{a.label}.md'
 if not a.force and (jp.exists() or mp.exists()): raise SystemExit('exists; use --force')
 data=load_rows(Path(a.data_root),5,'phase_balanced'); report=json.load(open(a.oof_report)); selected=np.array(report['folds'][a.meta_fold]['xs_selected'],dtype=int)
 with np.load(a.oof,allow_pickle=False) as d:
  valid=d['fold']>=0
  for n in ['target','weight','time_id','asset_id']:
   if not np.array_equal(d[n],data[n]): raise AssertionError(n)
  y=d['target'][valid].astype(float); w=np.maximum(d['weight'][valid].astype(float),0); t=d['time_id'][valid]; aid=d['asset_id'][valid].astype(int); fold=d['fold'][valid]; market=d['market'][valid].astype(float); e=d['e_lgbm'][valid].astype(float); base=d['prediction_raw'][valid].astype(float)
 x=data['features'][valid][:,selected].astype(np.float32); np.nan_to_num(x,copy=False,nan=0,posinf=0,neginf=0); meta=fold==a.meta_fold; cmean=x[meta].mean(0); cstd=x[meta].std(0); cstd[cstd<1e-8]=1; x=(x-cmean.astype(np.float32))/cstd.astype(np.float32); np.clip(x,-10,10,out=x)
 s,c=sc(t); n_groups=len(s); state=np.zeros(n_groups,dtype=float)
 for j in range(0,x.shape[1],25):
  z=x[:,j:j+25].astype(np.float64); m=np.repeat(np.add.reduceat(z,s,axis=0)/c[:,None],c,axis=0); dev=z-m; group_std=np.sqrt(np.add.reduceat(dev*dev,s,axis=0)/c[:,None]); state+=group_std.mean(1)
 state/=int(np.ceil(x.shape[1]/25)); state=np.repeat(state,c); target=y-gm(y,t); bins=(state>np.median(state[meta])).astype(int); sl=slopes(target[meta]-e[meta],e[meta],w[meta],aid[meta],bins[meta],a.shrink)
 arms={'baseline':base.copy()}; pred=base.copy()
 for f in np.unique(fold):
  q=fold==f; corr=proj(e[q]*sl[aid[q],bins[q]],t[q]); pred[q]=market[q]+corr
 arms['raw_state_asset']=pred
 scales={k:float(scale_invariant_score(y[meta],p[meta],w[meta])['optimal_scale']) for k,p in arms.items()}; evals=[int(f) for f in np.unique(fold) if f!=a.meta_fold]; rows=[]
 for f in evals:
  q=fold==f; rows.append({'fold':f,'baseline':float(scale_invariant_score(y[q],base[q],w[q])['peak']),'candidate':float(scale_invariant_score(y[q],pred[q],w[q])['peak']),'frozen_baseline':weighted_zero_mean_r2(y[q],base[q]*scales['baseline'],w[q]),'frozen_candidate':weighted_zero_mean_r2(y[q],pred[q]*scales['raw_state_asset'],w[q])})
 bp=np.array([r['baseline'] for r in rows]); cp=np.array([r['candidate'] for r in rows]); bf=np.array([r['frozen_baseline'] for r in rows]); cf=np.array([r['frozen_candidate'] for r in rows]); payload={'experiment':'v3_raw_cross_state_adapter','meta_fold':a.meta_fold,'selected_features':selected.tolist(),'state_meta_median':float(np.median(state[meta])),'state_meta_mean':float(state[meta].mean()),'slopes':sl.tolist(),'folds':rows,'peak':paired(cp,bp),'frozen':paired(cf,bf),'config':vars(a)}; jp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 lines=['# Raw cross-sectional state adapter','',f"Peak gain `{payload['peak']['relative_gain']*100:+.2f}%`, positive `{payload['peak']['positive_folds']}/{payload['peak']['n_folds']}`, drop-best `{payload['peak']['drop_best_relative_gain']*100:+.2f}%`, gate **{'PASS' if payload['peak']['pass'] else 'FAIL'}**",'']; mp.write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
if __name__=='__main__': main()
