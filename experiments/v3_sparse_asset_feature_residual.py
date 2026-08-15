"""Sparse asset×feature residual interaction screening on strict v3 OOF.

The adapter predicts only the cross-sectional residual correction. Feature transforms are fit on the
meta fold, top features are selected using meta-fold residual correlation, and all parameters are frozen
on later folds.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1]
for p in (str(ROOT),str(ROOT/'experiments'),str(ROOT/'strategies'/'v1_ridge')):
    if p not in sys.path: sys.path.insert(0,p)
from lgbm_xs import load_rows
from src.metric import scale_invariant_score, weighted_zero_mean_r2

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default=str(ROOT/'data'))
    p.add_argument('--oof',default=str(ROOT/'outputs/cache/v3_production_oof_confirm_3s480_phasebal_prodwindow.npz'))
    p.add_argument('--oof-report',default=str(ROOT/'outputs/experiments/v3_production_oof_confirm_3s480_phasebal_prodwindow.json'))
    p.add_argument('--output-dir',default=str(ROOT/'outputs/experiments'))
    p.add_argument('--label',default='v3_sparse_asset_feature_residual_3s480')
    p.add_argument('--meta-fold',type=int,default=0)
    p.add_argument('--top-k',type=int,nargs='+',default=[4,8,16])
    p.add_argument('--ridge-alpha',type=float,default=1000.0)
    p.add_argument('--force',action='store_true')
    return p.parse_args()

def starts_counts(t):
    s=np.r_[0,np.flatnonzero(t[1:]!=t[:-1])+1]; c=np.diff(np.r_[s,len(t)]); return s,c

def group_mean(v,t):
    s,c=starts_counts(t); return np.repeat(np.add.reduceat(v,s)/c,c)

def project(v,t): return v-group_mean(v,t)

def sparse_asset_design(x,aid,k):
    n=len(aid); rows=np.repeat(np.arange(n),k); cols=(aid[:,None]*k+np.arange(k)[None,:]).ravel()
    return sparse.csr_matrix((x[:,:k].ravel(),(rows,cols)),shape=(n,(int(aid.max())+1)*k))

def paired(c,b):
    d=c-b; drop=np.delete(d,int(np.argmax(d))); base=b.mean()
    return {'relative_gain':float(d.mean()/base),'positive_folds':int((d>0).sum()),'n_folds':len(d),
            'relative_gain_drop_best':float(drop.mean()/base),'per_fold_delta':d.tolist(),
            'pass':bool(d.mean()/base>=.01 and (d>0).sum()>=3 and drop.mean()>0)}

def main():
    a=parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); jp=out/f'{a.label}.json'; mp=out/f'{a.label}.md'
    if not a.force and (jp.exists() or mp.exists()): raise SystemExit('exists; use --force')
    started=time.perf_counter(); data=load_rows(Path(a.data_root),5,'phase_balanced')
    report=json.load(open(a.oof_report)); selected=np.array(report['folds'][a.meta_fold]['xs_selected'],dtype=np.int64)
    with np.load(a.oof,allow_pickle=False) as d:
        valid=d['fold']>=0
        for n in ['target','weight','time_id','asset_id']:
            if not np.array_equal(d[n],data[n]): raise AssertionError(f'{n} misaligned')
        y=d['target'][valid].astype(float); w=np.maximum(d['weight'][valid].astype(float),0); t=d['time_id'][valid]; aid=d['asset_id'][valid].astype(np.int64); fold=d['fold'][valid]
        market=d['market'][valid].astype(float); e=d['e_lgbm'][valid].astype(float); base=d['prediction_raw'][valid].astype(float)
    target_cross=y-group_mean(y,t); correction_target=target_cross-e; meta=fold==a.meta_fold
    x=data['features'][valid][:,selected].astype(np.float32); del data
    np.nan_to_num(x,copy=False,nan=0.0,posinf=0.0,neginf=0.0)
    center=x[meta].mean(axis=0,dtype=np.float64); scale=x[meta].std(axis=0,dtype=np.float64); scale[scale<1e-8]=1.0
    x=(x-center.astype(np.float32))/scale.astype(np.float32); np.clip(x,-10,10,out=x)
    s,c=starts_counts(t)
    x-=np.repeat(np.add.reduceat(x,s,axis=0)/c[:,None],c,axis=0).astype(np.float32)
    residual_meta=correction_target[meta]; wm=w[meta]; xm=x[meta]; aidm=aid[meta]
    cov=(xm.astype(np.float64).T @ (wm[:,None]*residual_meta[:,None]))[:,0]
    den=np.sqrt(np.maximum((xm.astype(np.float64)**2).T@wm,1e-30)*max(float(np.dot(wm,residual_meta**2)),1e-30))
    corr=np.abs(cov/den); order=np.argsort(-corr,kind='stable')
    arms={'baseline':base.copy()}; models={}; selected_top={}
    for k in sorted(set(a.top_k)):
        top=order[:k]; selected_top[str(k)]=selected[top].tolist()
        model=Ridge(alpha=a.ridge_alpha,fit_intercept=False,solver='lsqr',tol=1e-8,max_iter=2000)
        model.fit(sparse_asset_design(xm[:,top],aidm,k),residual_meta,sample_weight=wm); models[str(k)]=model
        pred=base.copy()
        for f in np.unique(fold):
            q=fold==f; corr_pred=model.predict(sparse_asset_design(x[q][:,top],aid[q],k)); pred[q]=market[q]+e[q]+project(corr_pred,t[q])
        arms[f'asset_feature_k{k}']=pred
    scales={n:float(scale_invariant_score(y[meta],p[meta],w[meta])['optimal_scale']) for n,p in arms.items()}
    evals=[int(f) for f in np.unique(fold) if f!=a.meta_fold]; rows=[]
    for f in evals:
        q=fold==f; row={'fold':f,'arms':{}}
        for n,p in arms.items(): row['arms'][n]={'peak':float(scale_invariant_score(y[q],p[q],w[q])['peak']),'frozen_score':weighted_zero_mean_r2(y[q],p[q]*scales[n],w[q])}
        rows.append(row)
    bp=np.array([r['arms']['baseline']['peak'] for r in rows]); bf=np.array([r['arms']['baseline']['frozen_score'] for r in rows]); summary={}
    for n in arms:
        if n=='baseline': continue
        cp=np.array([r['arms'][n]['peak'] for r in rows]); cf=np.array([r['arms'][n]['frozen_score'] for r in rows]); summary[n]={'peak':paired(cp,bp),'frozen_score':paired(cf,bf)}
    payload={'experiment':'v3_sparse_asset_feature_residual','oof':a.oof,'meta_fold':a.meta_fold,'config':vars(a),'top_features':selected_top,'feature_correlations':corr.tolist(),'meta_scales':scales,'folds':rows,'summary':summary,'elapsed_seconds':time.perf_counter()-started}; jp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    lines=['# Sparse asset×feature residual interaction','',f'Meta fold `{a.meta_fold}`','', '| Arm | Peak gain | Positive | Drop-best | Gate | Frozen gain |','|---|---:|---:|---:|:---:|---:|']
    for n,v in summary.items(): lines.append(f"| `{n}` | {v['peak']['relative_gain']*100:+.2f}% | {v['peak']['positive_folds']}/{v['peak']['n_folds']} | {v['peak']['relative_gain_drop_best']*100:+.2f}% | {'PASS' if v['peak']['pass'] else 'FAIL'} | {v['frozen_score']['relative_gain']*100:+.2f}% |")
    mp.write_text('\n'.join(lines)+'\n'); print('\n'.join(lines)); print(f"elapsed={payload['elapsed_seconds']:.1f}s")
if __name__=='__main__': main()
