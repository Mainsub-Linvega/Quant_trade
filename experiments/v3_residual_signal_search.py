"""Frozen OOF search for low-capacity, inference-available cross residual adapters.

All parameters are trained on one meta fold and frozen on the other four. The search is deliberately
small: asset slope, asset×|prediction| bins, asset×cross-RMS regime bins, and a linear soft gate.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.metric import scale_invariant_score, weighted_zero_mean_r2


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oof', default=str(ROOT/'outputs/cache/v3_production_oof_confirm_3s480_phasebal_prodwindow.npz'))
    p.add_argument('--output-dir', default=str(ROOT/'outputs/experiments'))
    p.add_argument('--label', default='v3_residual_signal_search_3s480')
    p.add_argument('--meta-fold', type=int, default=0)
    p.add_argument('--bin-shrink', type=float, default=50.0)
    p.add_argument('--soft-alpha', type=float, default=100.0)
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def starts_counts(t):
    s=np.r_[0,np.flatnonzero(t[1:]!=t[:-1])+1]; c=np.diff(np.r_[s,len(t)]); return s,c

def gmean(v,t):
    s,c=starts_counts(t); return np.repeat(np.add.reduceat(v,s)/c,c)

def project(v,t): return v-gmean(v,t)

def fit_binned(y,e,w,aid,bin_id,n_bins,shrink):
    n_assets=int(aid.max())+1; slopes=np.ones((n_assets,n_bins))
    for a in range(n_assets):
        for b in range(n_bins):
            m=(aid==a)&(bin_id==b); den=float(np.dot(w[m],e[m]*e[m])); num=float(np.dot(w[m],e[m]*y[m]))
            slopes[a,b]=(num+shrink)/(den+shrink)
    return slopes

def apply_binned(e,aid,bins,slopes,t): return project(e*slopes[aid,bins],t)

def state_arrays(t,aid,e,mr,ml,pred):
    s,c=starts_counts(t); cross=np.sqrt(np.add.reduceat(e*e,s)/c); gap=np.abs((ml-mr)[s]); pr=np.sqrt(np.add.reduceat(pred*pred,s)/c)
    cov=c.astype(float); phase=t[s]%10
    group=np.column_stack([cross,gap,pr,cov,np.sin(2*np.pi*phase/10),np.cos(2*np.pi*phase/10)])
    return np.repeat(group,c,axis=0), np.repeat(cross,c)

def standardize_fit(x):
    mean=x.mean(0); scale=x.std(0); scale[scale<1e-12]=1; return mean,scale

def soft_design(e,aid,state,mean,scale):
    z=(state-mean)/scale; n=len(e); n_assets=int(aid.max())+1
    out=np.zeros((n,n_assets+z.shape[1]),dtype=np.float64); out[np.arange(n),aid]=e; out[:,n_assets:]=e[:,None]*z
    return out

def metrics(y,p,w):
    peak=scale_invariant_score(y,p,w); return {'peak':float(peak['peak']),'optimal_scale':float(peak['optimal_scale']),
        'score':weighted_zero_mean_r2(y,p,w)}

def paired(c,b):
    d=c-b; drop=np.delete(d,int(np.argmax(d))); base=float(b.mean())
    return {'relative_gain':float(d.mean()/base),'positive_folds':int((d>0).sum()),'n_folds':len(d),
        'relative_gain_drop_best':float(drop.mean()/base),'per_fold_delta':[float(x) for x in d],
        'pass':bool(d.mean()/base>=.01 and (d>0).sum()>=3 and drop.mean()>0)}

def main():
    a=parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); jp=out/f'{a.label}.json'; mp=out/f'{a.label}.md'
    if not a.force and (jp.exists() or mp.exists()): raise SystemExit('output exists; use --force')
    with np.load(a.oof,allow_pickle=False) as d:
        m=d['fold']>=0; arr={k:d[k][m] for k in ['target','weight','time_id','asset_id','fold','market','market_ridge','market_lgbm','e_lgbm','prediction_raw']}
    y=arr['target'].astype(float); w=np.maximum(arr['weight'].astype(float),0); t=arr['time_id'].astype(np.int64); aid=arr['asset_id'].astype(np.int64); fold=arr['fold']
    market=arr['market'].astype(float); mr=arr['market_ridge'].astype(float); ml=arr['market_lgbm'].astype(float); e=arr['e_lgbm'].astype(float); base=arr['prediction_raw'].astype(float)
    target_cross=y-gmean(y,t); meta=fold==a.meta_fold
    # Asset × |e|: thresholds and parameters are meta-only.
    mag_thr=np.array([np.median(np.abs(e[meta&(aid==i)])) for i in range(int(aid.max())+1)])
    mag_bin=(np.abs(e)>mag_thr[aid]).astype(np.int8)
    mag_slopes=fit_binned(target_cross[meta],e[meta],w[meta],aid[meta],mag_bin[meta],2,a.bin_shrink)
    # Asset × observable cross-RMS regime.
    states,cross_rms=state_arrays(t,aid,e,mr,ml,base); regime_thr=float(np.median(cross_rms[meta])); regime=(cross_rms>regime_thr).astype(np.int8)
    regime_slopes=fit_binned(target_cross[meta],e[meta],w[meta],aid[meta],regime[meta],2,a.bin_shrink)
    # Linear soft gate: per-asset base slope plus global prediction-state interactions.
    sm,ss=standardize_fit(states[meta]); dmeta=soft_design(e[meta],aid[meta],states[meta],sm,ss)
    soft=Ridge(alpha=a.soft_alpha,fit_intercept=False,solver='lsqr',tol=1e-8,max_iter=2000)
    soft.fit(dmeta,target_cross[meta],sample_weight=w[meta])
    arms={k:base.copy() for k in ['baseline','asset_magnitude','asset_regime','soft_gate']}
    for f in np.unique(fold):
        q=fold==f
        arms['asset_magnitude'][q]=market[q]+apply_binned(e[q],aid[q],mag_bin[q],mag_slopes,t[q])
        arms['asset_regime'][q]=market[q]+apply_binned(e[q],aid[q],regime[q],regime_slopes,t[q])
        pred=soft.predict(soft_design(e[q],aid[q],states[q],sm,ss)); arms['soft_gate'][q]=market[q]+project(pred,t[q])
    scales={k:float(scale_invariant_score(y[meta],p[meta],w[meta])['optimal_scale']) for k,p in arms.items()}
    eval_folds=[int(f) for f in np.unique(fold) if f!=a.meta_fold]; rows=[]
    for f in eval_folds:
        q=fold==f; row={'fold':f,'arms':{}}
        for k,p in arms.items(): row['arms'][k]={**metrics(y[q],p[q],w[q]),'frozen_score':weighted_zero_mean_r2(y[q],p[q]*scales[k],w[q])}
        rows.append(row)
    bp=np.array([r['arms']['baseline']['peak'] for r in rows]); bf=np.array([r['arms']['baseline']['frozen_score'] for r in rows]); summary={}
    for k in arms:
        if k=='baseline': continue
        cp=np.array([r['arms'][k]['peak'] for r in rows]); cf=np.array([r['arms'][k]['frozen_score'] for r in rows])
        summary[k]={'peak':paired(cp,bp),'frozen_score':paired(cf,bf)}
    payload={'experiment':'v3_residual_signal_search','oof':a.oof,'meta_fold':a.meta_fold,'eval_folds':eval_folds,
      'config':vars(a),'magnitude_thresholds':mag_thr.tolist(),'magnitude_slopes':mag_slopes.tolist(),
      'regime_threshold':regime_thr,'regime_slopes':regime_slopes.tolist(),'soft_coef':soft.coef_.tolist(),
      'meta_scales':scales,'folds':rows,'summary':summary}
    jp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    lines=['# v3 residual signal search','',f'Meta fold `{a.meta_fold}`, eval `{eval_folds}`','',
      '| Arm | Peak gain | Positive | Drop-best | Peak gate | Frozen gain | Frozen gate |','|---|---:|---:|---:|:---:|---:|:---:|']
    for k,v in summary.items():
        p=v['peak']; fs=v['frozen_score']; lines.append(f"| `{k}` | {p['relative_gain']*100:+.2f}% | {p['positive_folds']}/{p['n_folds']} | {p['relative_gain_drop_best']*100:+.2f}% | {'PASS' if p['pass'] else 'FAIL'} | {fs['relative_gain']*100:+.2f}% | {'PASS' if fs['pass'] else 'FAIL'} |")
    mp.write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
if __name__=='__main__': main()
