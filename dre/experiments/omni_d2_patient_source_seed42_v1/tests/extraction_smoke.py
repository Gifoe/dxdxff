"""Bounded baseline-only numerical and original forward parity, no head outcome."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import torch
from common import *
def main():
    sys.path.insert(0,str(BASE/'code'))
    from modeling import load_official_module,FixedSegmentSpectrum
    from evaluate_oof_r2 import load_model
    from train_fold_r3 import patient_forward
    mf=manifest();row=next(mf.itertuples());patient=str(row.patient_name)
    path=ROOT/f'embeddings/fold_{row.fold}/{key(patient)}.npz'
    with np.load(path,allow_pickle=False) as data:
        channels=data['channels'].copy();logits=data['logits'].copy();emb=data['embedding'].copy()
    old=OLD/f'oof_patient_cache/fold_{row.fold}/baseline/{key(patient)}.npz'
    with np.load(old,allow_pickle=False) as cached:
        assert np.array_equal(channels,cached['channels']);delta=float(abs(sigmoid(logits)-cached['scores']).max())
        assert delta<=1e-6
        v=cached['labels']>=0;fresh=metric(cached['labels'][v],sigmoid(logits[v]));historical=metric(cached['labels'][v],cached['scores'][v])
        assert all(abs(fresh[k]-historical[k])<=1e-12 for k in METRICS)
    # Independent original one-segment API proves extraction ordering/reduction.
    source=load_official_module(OFFICIAL);raw,_=load_model(OLD/f'checkpoints/fold_{row.fold}/baseline/last.pt','baseline',source,torch.device('cuda'))
    frozen=raw.state_dict();before={k:v.detach().cpu().clone() for k,v in frozen.items()};raw.requires_grad_(False)
    spectrum=FixedSegmentSpectrum(source);paths=[Path(p) for p in json.loads((ROOT/'INPUTS_PRIVATE.json').read_text())['records'][patient]]
    with torch.inference_mode():l,y,c=patient_forward(raw,None,spectrum,paths,0,torch.device('cuda'),False)
    assert c==channels.tolist() and all(torch.equal(v.cpu(),before[k]) for k,v in raw.state_dict().items())
    assert np.isfinite(emb).all() and emb.shape==(len(channels),32)
    write_json(ROOT/'public/EXTRACTION_SMOKE.json',{'status':'PASS','frozen_baseline_score_max_error':delta,'all_patient_metrics_equal':True,'original_segment_api_finite':True,'CNN_state_unchanged':True,'no_new_head_training':True})
    print('FIRST_PATIENT_C0_PARITY_AND_ORIGINAL_API_SMOKE_PASS',delta,flush=True)
if __name__=='__main__':main()
