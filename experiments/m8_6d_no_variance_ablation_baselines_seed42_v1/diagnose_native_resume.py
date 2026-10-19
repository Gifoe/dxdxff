"""Diagnosed complete-epoch native resume audit; never alter scientific configuration."""
from common import *
import shutil

def main():
    binding,core,fac,records,rec,banks=setup()
    assert jr(PUB/'SOURCE_PUSH_BEFORE_TRAIN.json')['source']==sources()==jr(PUB/'ADMISSION.json')['source']
    with (PUB/'TRAIN_CELLS.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    fresh=[r for r in rows if r['reused']=='False'];assert len(fresh)==43
    for r in fresh:
        root=ROOT/'models_private'/r['arm']/('fold'+r['fold']);done=jr(root/'COMPLETE_PRIVATE.json')
        bp=root/'BEST_PRIVATE.pt';vp=root/'VALIDATION_PRIVATE.pkl'
        assert sha(bp)==r['checkpoint_sha256']==done['best_sha256']
        assert sha(vp)==r['validation_sha256']==done['validation_sha256']
        q=read(vp);assert q['binding']['input']==binding and done['binding']==digest(q['binding'])
    arm='WO_LATE_GATE';fold=4;path=ROOT/'models_private'/arm/f'fold{fold}'/'LAST_PRIVATE.pt'
    q=torch.load(path,map_location='cpu',weights_only=False)
    assert q['epoch']==8 and len(q['history'])==8 and q['best']['epoch']==7
    assert q['binding']['input']==binding and q['binding']['arm']==arm and q['binding']['fold']==fold
    assert all(torch.isfinite(v).all() for v in q['model'].values())
    for slot in q['optimizer']['state'].values():
        assert all(torch.isfinite(v).all() for v in slot.values() if torch.is_tensor(v))
    assert {'python','numpy','torch','cuda'}.issubset(q['rng'])
    dest=ROOT/'recovery_private/native_03/LAST_epoch8_before_resume04.pt';dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copy2(path,dest)
    assert sha(dest)==sha(path)
    audit=dict(status='VERIFIED_NATIVE_RECOVERY_FROM_EXACT_LAST',exception='Windows c0000005',arm=arm,fold=fold,
        failure_site='Native access violation during autograd backward; timestamp-matched Windows Application Error names cublasLt64_12.dll; root cause not established',
        last_complete_epoch=8,resume_epoch=9,best_epoch_at_last=7,completed_fresh_cells_preserved=len(fresh),
        failed_run_tag='03',resume_run_tag='04',checkpoint_sha256=sha(path),model_tensors=len(q['model']),
        optimizer_slots=len(q['optimizer']['state']),backup_hash_verified=True,all_completed_hashes_verified=True,
        scientific_source_changed=False,model_data_loss_selection_hyperparameters_changed=False,python_level_error=False)
    write(PUB/'NATIVE_RESUME_AUDIT.json',audit)
    print(json.dumps(audit,sort_keys=True),flush=True)
if __name__=='__main__':main()
