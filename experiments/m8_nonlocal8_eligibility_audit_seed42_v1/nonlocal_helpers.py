"""Pinned pure-helper replay; does not execute cache builder or index labels."""
import ast
import __future__
import numpy as np

NAMES=['early_high_gamma_slope','early_line_length_slope','onset_latency_high_gamma',
       'onset_latency_line_length','onset_rank_high_gamma','onset_rank_line_length',
       'high_gamma_top20pct_mean','line_length_top20pct_mean']


def pure_functions(path,names,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    found=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in found}==set(names)
    code=compile(ast.Module(body=found,type_ignores=[]),str(path),'exec',
                 flags=__future__.annotations.compiler_flag,dont_inherit=True)
    exec(code,namespace)
    return namespace


def isolated(module,hg,ll,t):
    assert hg.shape==ll.shape and hg.ndim==2 and hg.shape[0]==len(t)
    assert np.isfinite(hg).all() and np.isfinite(ll).all() and np.isfinite(t).all()
    early=(t>=0)&(t<=10)
    if early.sum()<2:early=t>=0
    hs=module._least_squares_slope(hg,t,early);ls=module._least_squares_slope(ll,t,early)
    hl,hr=module._latency_features(hg,t);llat,lr=module._latency_features(ll,t)
    post=module._post_mask_or_all(t,len(t))
    ht=module._top_pct_mean(hg[post]);lt=module._top_pct_mean(ll[post])
    return np.stack([hs,ls,hl,llat,hr,lr,ht,lt],axis=-1).astype(np.float32)


def tolerance(reference):
    reference=np.asarray(reference,dtype=np.float32)
    return 8*np.abs(np.spacing(reference).astype(float))+2e-6*np.abs(reference.astype(float))


def different(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return np.abs(a.astype(float)-b.astype(float))>tolerance(b)


def diagnostics(module,series,t):
    pre=t<0;post=t>=0
    if not pre.any() or not post.any():
        return dict(pre_windows=int(pre.sum()),post_windows=int(post.sum()),unsupported=True)
    sd=series[pre].std(0);threshold=series[pre].mean(0)+2*sd
    crossing=(series[post]>=threshold).any(0)
    latency,_=module._latency_features(series,t)
    _,counts=np.unique(latency,return_counts=True)
    return dict(pre_windows=int(pre.sum()),post_windows=int(post.sum()),unsupported=False,
                zero_pre_std_channels=int((sd==0).sum()),no_crossing_channels=int((~crossing).sum()),
                tied_latency_channels=int(counts[counts>1].sum()),channels=len(sd),
                early_windows=int(((t>=0)&(t<=10)).sum()))
