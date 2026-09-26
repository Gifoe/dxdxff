"""Reference runner against the USER-SUPPLIED EpiLENS_Code_Supplement APIs.

This script deliberately reuses that supplement's feature construction,
patient manifests, threshold selector, and metrics. It does not discover server
caches or guess their schema. Only trusted local pickle data should be opened.
Historical server snapshots need a separately audited adapter (see CODEX_PROMPT).
All schedule parameters below must be passed explicitly to prevent accidentally
claiming the supplement's defaults reproduce the paper's training schedule.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from pareset_ez import ModelConfig, PaReSetEZ, patient_objective, parameter_count


def sha256(path):
    with open(path,'rb') as f:
        digest=hashlib.sha256()
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True,help='folder containing the supplied epilens/ package')
    parser.add_argument('--data',type=Path,required=True,help='trusted local PatientRecord-compatible pickle')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--fold',type=int,required=True)
    parser.add_argument('--expected-patients',type=int,default=80)
    parser.add_argument('--variant',choices=['full','base','no_reference','uniform_reference','no_temporal','no_pool','bce_only'],default='full')
    parser.add_argument('--epochs',type=int,required=True)
    parser.add_argument('--minimum-epochs',type=int,required=True)
    parser.add_argument('--patience',type=int,required=True)
    parser.add_argument('--weight-decay',type=float,required=True)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--dropout',type=float,default=.4)
    parser.add_argument('--patient-batch-size',type=int,default=4)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--evaluate-test',action='store_true',help='evaluate the fixed test partition only AFTER locking configuration')
    args=parser.parse_args()
    if not 1 <= args.minimum_epochs <= args.epochs or args.patience<1 or args.patient_batch_size<1:
        parser.error('invalid explicit training schedule')
    sys.path.insert(0,str(args.source_root.resolve()))
    from epilens.data import load_records,load_partition_manifest,validate_protocol,select_records
    from epilens.features import fit_standardizer,four_view_expansion
    from epilens.evaluation import select_threshold,patient_metrics
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA explicitly requested but unavailable')
    if args.output.exists() and any(args.output.iterdir()): raise RuntimeError('Output directory is nonempty; use a fresh experiment directory')
    args.output.mkdir(parents=True,exist_ok=True)
    records=load_records(args.data);manifest=load_partition_manifest(args.manifest)
    validate_protocol(records,manifest,args.expected_patients)
    parts={p:select_records(records,manifest,args.fold,p) for p in ['fit','validation','test']}
    if any(not v for v in parts.values()): raise ValueError('fit/validation/test must each be nonempty')
    # Validate complete fold membership, not just pairwise disjointness.
    if set(r.patient_id for v in parts.values() for r in v) != set(r.patient_id for r in records):
        raise ValueError('selected fold must partition ALL records')
    standardizer=fit_standardizer(parts['fit'])
    cfg=ModelConfig(dropout=args.dropout)
    if args.variant=='base':cfg=replace(cfg,use_temporal_update=False,use_adaptive_seizure_pool=False)
    if args.variant=='no_reference':cfg=replace(cfg,use_reference_contrast=False)
    if args.variant=='uniform_reference':cfg=replace(cfg,reference_mode='uniform')
    if args.variant=='no_temporal':cfg=replace(cfg,use_temporal_update=False)
    if args.variant=='no_pool':cfg=replace(cfg,use_adaptive_seizure_pool=False)
    boundary_weight=0 if args.variant=='bce_only' else .05
    model=PaReSetEZ(cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    # Store tensors on CPU. No test-label-dependent transformation is fitted.
    tensors={}
    for record in records:
        x,v=four_view_expansion(record.descriptors,record.window_times,record.valid)
        x=standardizer.apply(x,v)
        tensors[record.patient_id]=(torch.tensor(x),torch.tensor(v,dtype=torch.bool),
                                  torch.tensor(record.window_times),torch.tensor(record.label_nez))
    def tensorize(record): return tuple(x.to(device) for x in tensors[record.patient_id])
    @torch.no_grad()
    def predict(group):
        model.eval();rows=[]
        for record in group:
            x,v,t,y=tensorize(record);o=model(x,v,t)
            for ch in o['channel_valid'].nonzero(as_tuple=False).flatten().tolist():
                rows.append(dict(patient_id=record.patient_id,center=record.center,
                    channel_name=record.channel_names[ch],label_nez=int(y[ch]),
                    probability_nez=float(o['probability_nez'][ch]),logit_ez=float(o['logit_ez'][ch])))
        if not rows:raise RuntimeError('prediction group has no valid channels')
        return pd.DataFrame(rows)
    provenance=dict(status='TRAINING_RUN',seed=args.seed,outer_fold=args.fold,variant=args.variant,
        model_config=asdict(cfg),boundary_weight=boundary_weight,parameters=parameter_count(model),
        data_sha256=sha256(args.data),manifest_sha256=sha256(args.manifest),
        source_models_sha256=sha256(args.source_root/'epilens'/'models.py'),
        source_features_sha256=sha256(args.source_root/'epilens'/'features.py'),
        source_evaluation_sha256=sha256(args.source_root/'epilens'/'evaluation.py'),
        runner_arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        probability_semantics='probability_nez >= validation_threshold => NEZ',
        test_labels_used_for_selection=False,checkpoint_averaging=False,seizure_dropout=False)
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    best_state=None;best_selection=None;best_key=None;history=[];stale=0;best_epoch=None
    for epoch in range(1,args.epochs+1):
        order=list(parts['fit']);random.Random(args.seed+epoch).shuffle(order)
        model.train();epoch_losses=[]
        for start in range(0,len(order),args.patient_batch_size):
            batch=order[start:start+args.patient_batch_size];opt.zero_grad(set_to_none=True)
            for record in batch:
                x,v,t,y=tensorize(record);o=model(x,v,t)
                loss,_=patient_objective(o['logit_ez'],y,o['channel_valid'],boundary_weight)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('non-finite loss')
                (loss/len(batch)).backward();epoch_losses.append(float(loss.detach()))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            opt.step()
        ledger=predict(parts['validation']);selection=select_threshold(ledger)
        key=(selection.patient_macro_f1,selection.patient_ez_f1,selection.patient_balanced_accuracy)
        if best_key is None or key>best_key:
            best_key=key;best_selection=selection;best_epoch=epoch;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(dict(model=best_state,model_config=asdict(cfg),threshold=selection.threshold,
                standardizer_mean=standardizer.mean.tolist(),standardizer_scale=standardizer.scale.tolist(),
                provenance=provenance,epoch=epoch),args.output/'best.pt')
        else:stale+=1
        row=dict(epoch=epoch,loss=float(np.mean(epoch_losses)),**asdict(selection));history.append(row)
        pd.DataFrame(history).to_csv(args.output/'history.csv',index=False)
        print(json.dumps(row),flush=True)
        if epoch>=args.minimum_epochs and stale>=args.patience:break
    if best_state is None:raise RuntimeError('no valid checkpoint')
    model.load_state_dict(best_state);validation=predict(parts['validation'])
    selection_recomputed=select_threshold(validation)
    if abs(selection_recomputed.threshold-best_selection.threshold)>1e-10:raise RuntimeError('checkpoint threshold reproduction failed')
    for partition in (['validation','test'] if args.evaluate_test else ['validation']):
        frame=validation if partition=='validation' else predict(parts[partition])
        frame['outer_fold']=args.fold;frame['seed']=args.seed;frame['partition']=partition
        frame['threshold']=best_selection.threshold
        frame['predicted_nez']=(frame['probability_nez']>=best_selection.threshold).astype(int)
        frame.to_csv(args.output/f'{partition}_predictions.csv',index=False)
        metrics=patient_metrics(frame,best_selection.threshold)
        metrics.to_csv(args.output/f'{partition}_patient_metrics.csv',index=False)
        means=metrics.select_dtypes(include='number').mean().to_dict()
        (args.output/f'{partition}_summary.json').write_text(json.dumps(means,indent=2),encoding='utf-8')
    (args.output/'selection.json').write_text(json.dumps(dict(epoch=best_epoch,**asdict(best_selection),
        test_evaluated=bool(args.evaluate_test)),indent=2),encoding='utf-8')

if __name__=='__main__':main()
