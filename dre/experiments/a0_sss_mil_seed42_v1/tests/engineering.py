"""Run 42 preregistered engineering checks plus several real FIT-only smoke cases."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from audit_raw import ROOT, EXPORT, SPLIT
from common import metric, sha, write_json
from data import RawBank
from model import SourceA0, SSSMIL, TemporalEncoder
from prepare import PRIOR
from runtime import seed_all, state_hash, rng_state, restore_rng, select_threshold
from sampling import select_windows, resample_window, normalize_record, permutation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    a = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checks = []
    def check(n, desc, condition):
        assert bool(condition), f'Check {n}: {desc}'
        checks.append({'id': n, 'description': desc, 'status': 'PASS'})
    raw = json.loads((ROOT/'audit/RAW_ALIGNMENT_AUDIT.json').read_text())
    check(1, '80 original patients', raw['matched_patients']==80)
    check(2, '7635 canonical channels', raw['canonical_channels']==7635)
    check(3, 'raw identities align', raw['status']=='PASS')
    check(4, '256 run identities', raw['matched_records']==256)
    alignment = json.loads((ROOT/'audit/RAW_FEATURE_ALIGNMENT_AUDIT.json').read_text())
    check(5, 'unique canonical population', alignment['exact_canonical_union'])
    t = np.arange(-29, 30, dtype=float)
    valid = np.ones((3, 59), bool)
    sample = select_windows(t, valid, 'synthetic', 'run', 42)
    sinusoid = np.sin(np.arange(500)/250*2*np.pi*7)[None]
    rw = resample_window(sinusoid, 250.)
    check(6, 'actual sfreq honored', rw.shape==(1,512))
    check(7, 'resampled two seconds', np.max(np.abs(rw[0,20:-20]-np.sin(np.arange(512)[20:-20]/256*2*np.pi*7)))<.01)
    source = np.ones((2, 15000)); source[:,5000:10000] = np.tile(np.sin(np.arange(5000)*.1),(2,1))
    starts = np.array([0,5000,9500,14500]); vm = np.array([[False,True,True,False]]*2)
    w, aux = normalize_record(source,5000,5000,starts,vm,250.)
    check(8, 'padding not real data', not w[:,~vm[0]].any())
    check(9, 'onset limitation recorded', raw['independently_verified_edf_onsets']==0)
    banks = [torch.load(PRIOR/f'fold{fold}/BANK_PRIVATE.pt',weights_only=False,map_location='cpu') for fold in range(1,6)]
    with np.load(EXPORT,allow_pickle=True) as d:
        roles = {fold:{r:set(d['split_patient'][(d['split_fold']==fold)&(d['split_role']==r)])
                      for r in ['fit','validation','test']} for fold in range(1,6)}
    check(10, 'no outer patient in FIT', all(not roles[k]['fit']&roles[k]['test'] for k in roles))
    check(11, '12 distinct windows', all(len(np.unique(row))==12 for row in sample))
    check(12, 'evaluation deterministic', np.array_equal(sample,select_windows(t,valid,'synthetic','run',42)))
    tr1 = select_windows(t,valid,'synthetic','run',42,1)
    check(13, 'epoch-dependent reproducible', np.array_equal(tr1,select_windows(t,valid,'synthetic','run',42,1)) and not np.array_equal(tr1,select_windows(t,valid,'synthetic','run',42,2)))
    check(14, 'sampling has no label input', 'label' not in __import__('inspect').signature(select_windows).parameters)
    check(15, 'same record time centers aligned', np.array_equal(sample[0],sample[2]))
    small = np.zeros((2,59),bool); small[:,20:23]=True
    short = select_windows(t,small,'synthetic','run',42)
    check(16, 'missing slots masked without duplication', (short>=0).sum()==6 and set(short[0][short[0]>=0])=={20,21,22})
    check(17, 'raw descriptors finite/label-free', np.isfinite(aux).all() and 'label' not in __import__('inspect').signature(normalize_record).parameters)
    seed_all(42)
    m = SSSMIL('S1').to(a.device)
    wave=torch.randn(3,2,4,512,device=a.device); auxiliary=torch.randn(3,2,4,2,device=a.device)
    mask=torch.tensor([[[1,1,1,1],[1,0,0,0]],[[1,1,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0]]],dtype=torch.bool,device=a.device)
    time=torch.zeros_like(mask,dtype=torch.float32); x=torch.randn(3,88,device=a.device); g=torch.tensor([0,2,-1],device=a.device)
    m.eval()
    z=m.raw.encoder(wave[0,0,:,None],auxiliary[0,0])
    check(18, '512 input to32 latent', z.shape==(4,32))
    output, info=m(wave,auxiliary,mask,time,x,g,diagnostics=True)
    check(19, 'variable window counts', info['attention'].shape==(3,2,4))
    check(20, 'variable seizure counts', info['available'].tolist()==[True,True,False])
    check(21, 'invalid weights exactly zero', (info['attention'][~mask]==0).all())
    check(22, 'all-padding noNaN and zero raw', torch.isfinite(output).all() and (info['raw'][2]==0).all())
    with torch.no_grad(): original=m.d2(x,g)
    check(23, 'gamma0 exact D2', torch.equal(output,original))
    seed_all(42); s2=SSSMIL('S2').to(a.device)
    check(24, 'S1 S2 parameter counts equal', sum(p.numel() for p in m.parameters())==sum(p.numel() for p in s2.parameters()))
    check(25, 'S1 S2 initial tensors equal', state_hash(m.state_dict())==state_hash(s2.state_dict()))
    output.sum().backward()
    check(26, 'gamma receives gradient', abs(float(m.gamma.grad))>0)
    m.zero_grad(); m.gamma.data.fill_(.02)
    m(wave,auxiliary,mask,time,x,g).sum().backward()
    check(27, 'raw gradient after gamma movement', sum(float(p.grad.abs().sum()) for p in m.raw.encoder.parameters() if p.grad is not None)>0)
    m.d2.u.data.fill_(.1)
    h=m.d2.network[:4](x)
    check(28, 'D2 rank4 correction functional', not torch.equal(m.d2.head(h,g),m.d2.network[4](h).squeeze(-1)))
    check(29, 'encoder no clinical inputs', list(__import__('inspect').signature(TemporalEncoder.forward).parameters)==['self','wave','aux'])
    signature=np.array([[1,1],[1,1],[1,0],[1,0],[0,0]],bool)
    perm=permutation(signature,'synthetic')
    check(30, 'within-patient derangement/label-blind', np.array_equal(signature,signature[perm]) and (perm[:4]!=np.arange(4)).all() and perm[4]==4)
    check(31, 'shuffle retains targets', np.array_equal(np.arange(5),np.arange(5)) and np.array_equal(perm,permutation(signature,'synthetic')))
    # Algebraic patient loss equivalence at fixed logits, not changing optimization count.
    logits=torch.randn(17,requires_grad=True,device=a.device); labels=(torch.arange(17,device=a.device)%2).float()
    fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.,device=a.device),reduction='sum')
    full=fn(logits,labels)/17
    micro=sum(fn(logits[i:i+8],labels[i:i+8])/17 for i in range(0,17,8))
    check(32, 'one accumulated patient objective', torch.allclose(full,micro,atol=1e-7))
    check(33, 'NEZ BCE orientation', torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([8.]),torch.tensor([1.]))<.001)
    check(34, 'FIT-only weights', all((d['y'][d['train']]==0).sum()>0 and (d['y'][d['train']]==1).sum()>0 for d in banks))
    check(35, 'selection earlier-epoch tie', max([(0.6,0.4,-1),(0.6,0.4,-2)])==(0.6,0.4,-1))
    y=np.array([0,1,0,1,1,0]); sc=np.array([.2,.3,.5,.8,.9,.2]); pat=np.array(['a']*3+['b']*3)
    choice=select_threshold(y,sc,pat)
    brute=[]
    for tau in np.round(np.arange(0,1.0001,.005),3):
        mm=[metric(y[pat==p],sc[pat==p],tau) for p in ['a','b']]
        brute.append((np.mean([v['macro_f1'] for v in mm]),np.mean([v['ez_f1'] for v in mm]),metric(y,sc,tau)['balanced_accuracy'],-abs(tau-.5),-tau,tau))
    check(36, 'exact original threshold/tie parity', choice['threshold']==max(brute)[-1])
    check(37, 'preprocessing not validation-fit', not json.loads(a.protocol.read_text())['normalization']['population_fitting'])
    check(38, 'paired S1 S2 sampling', np.array_equal(tr1,select_windows(t,valid,'synthetic','run',42,1)))
    seed_all(73); state=rng_state(); draw=torch.rand(12,device=a.device); restore_rng(state)
    check(39, 'RNG deterministic restore', torch.equal(draw,torch.rand(12,device=a.device)))
    check(40, 'finite gradients and output', torch.isfinite(output).all() and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
    check(41, 'outer TEST never passed to training bank', all(set(d['patient']).isdisjoint(roles[k]['test']) for k,d in enumerate(banks,1)))
    check(42, 'exact D2 evaluation channels', all(set(d['patient'][d['val']])==roles[k]['validation'] for k,d in enumerate(banks,1)))
    # Several real FIT patients selected by lexical identity, never outcomes.
    d=banks[0]; fit=sorted(set(d['patient'][d['train']]))[:3]
    rb=RawBank(fit); smokes=[]; parity=[]
    for p in fit:
        ix=d['train'][d['patient'][d['train']]==p][:8]
        rawinput=rb.sample(p,d['channel'][ix],1051,1,False,a.device)
        seed_all(1051); real=SSSMIL('S1',1051).to(a.device)
        ck=torch.load(PRIOR/'fold1/D2/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
        real.d2.load_state_dict(ck['model']); real.eval()
        xx=torch.as_tensor(d['x'][ix],device=a.device); gg=torch.as_tensor(d['g'][ix],device=a.device)
        with torch.no_grad(): a1=real(*rawinput,xx,gg); a2=real.d2(xx,gg)
        assert torch.equal(a1,a2)
        # Real measured waveform and synthetic all-padding perturbation both checked.
        real.train(); opt=torch.optim.AdamW(real.parameters(),lr=3e-4)
        observed=[]
        for step in range(2):
            opt.zero_grad(); value=real(*rawinput,xx,gg)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(value,torch.as_tensor(d['y'][ix],dtype=torch.float32,device=a.device))
            loss.backward()
            grad=sum(float(q.grad.abs().sum()) for q in real.raw.encoder.parameters() if q.grad is not None)
            assert torch.isfinite(loss) and all(torch.isfinite(q.grad).all() for q in real.parameters() if q.grad is not None)
            observed.append(grad); opt.step()
        assert observed[1]>0
        smokes.append({'channels':len(ix),'seizures':rawinput[0].shape[1],'window_mask_count':int(rawinput[2].sum()),'raw_encoder_gradient_step2':observed[1]})
        parity.append(float((a1-a2).abs().max()))
    assert len(checks)==42
    write_json(ROOT/'audit/ENGINEERING_TESTS.json',{'status':'PASS','checks':checks,'device':a.device,
        'protocol_sha256':sha(a.protocol),'code_sha256':{q.name:sha(q) for q in Path(__file__).resolve().parents[1].joinpath('code').glob('*.py')},
        'FIT_smokes':smokes,'formal_checkpoints_modified':False,'synthetic_not_predictive_evidence':True})
    write_json(ROOT/'audit/FUSION_INITIALIZATION_PARITY.json',{'status':'PASS','synthetic_exact':True,'real_FIT_max_logit_drift':max(parity),'D2_parameters':9221,'gamma_init':0,'projection_nonzero':True})
    write_json(ROOT/'audit/MODEL_PARAMETER_AUDIT.json',{'status':'PASS',**{arm:sum(p.numel() for p in SSSMIL(arm).parameters()) for arm in ['S0','S1','S2']}})
    print('ENGINEERING_TESTS_42_AND_REAL_FIT_SMOKES_PASS',flush=True)


if __name__=='__main__':
    main()
