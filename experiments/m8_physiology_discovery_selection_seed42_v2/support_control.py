"""Prospective Raw-only temporal-support control; original sources unchanged."""
import gc
from common import *
from model_io import Inputs,make,restore_cpu_rng
from model import Concat
from train import fit_raw,predict


class ConcatSupport30(Concat):
    def features(self,raw,mask,times,delta,scale,mode='full',perms=None):
        temporary=torch.where(times>=8,8+(times-8)*(12/22),times)
        # No observed time is changed; only interval membership is represented.
        for desired,actual in [(times<0,temporary<0),((times>=0)&(times<8),(temporary>=0)&(temporary<8)),((times>=8)&(times<30),(temporary>=8)&(temporary<20))]:
            assert torch.equal(mask&desired[:,None,:],mask&actual[:,None,:])
        return super().features(raw,mask,temporary,delta,scale,mode='identity',perms=perms)


def promote(model):
    # Same object/state layout, no new trainables, only the checked pool predicate.
    model.__class__=ConcatSupport30
    return model


def main():
    c,core,fac,trainer,records,rec,banks,deps=legal()
    original_restore=core.restore_rng;core.restore_rng=lambda s:original_restore(restore_cpu_rng(s))
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    mainstatus=jr(PUB/'NEURAL_STATUS.json');assert mainstatus['status']=='ALL80_LOGICAL_MODELS_FROZEN'
    fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl');reg=read(ROOT/'POOL_REGISTRY_PRIVATE.pkl');inputs=Inputs(reg)
    binding=dict(main_model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl'),source={n:sha(HERE/n) for n in ['TIME_SUPPORT_AMENDMENT.json','support_control.py','model.py','model_io.py','train.py','common.py']})
    freeze(ROOT/'SUPPORT_INPUT_FREEZE_PRIVATE.json',binding);rows=[];history=[]
    for fold in range(1,6):
        d=torch.load(ROOT/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False);d=dict(d,x=np.zeros_like(d['x']))
        m=promote(make(core,fac,records,fold,'Concat',[]));ss=torch.tensor(d['scale'],device='cuda');cell=ROOT/'support30_private'/f'fold{fold}'
        b,h=fit_raw(m,d,inputs,ss,cell,42+1009*fold,binding,core,'RawSupport30',fold)
        logit=predict(m,d,d['val'],inputs,ss);cp=cell/'BEST_PRIVATE.pt';vp=cell/'VALIDATION_PRIVATE.pkl'
        save(vp,dict(checkpoint_sha256=sha(cp),logit=logit,archive_indices=d['archive_indices'][d['val']],threshold=b['threshold']))
        rows.append(dict(fold=fold,recipe='RAW_SUPPORT30',family='Concat',groups=[],checkpoint=str(cp),checkpoint_sha256=sha(cp),epoch=b['epoch'],threshold=b['threshold'],epochs_run=len(h),validation=str(vp),validation_sha256=sha(vp),signature='RAW_SUPPORT30',alias_of=None))
        history += [dict(fold=fold,**r) for r in h];del m;gc.collect();torch.cuda.empty_cache()
    save(ROOT/'SUPPORT_MODEL_FREEZE_PRIVATE.pkl',dict(binding=binding,selections=rows))
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['checkpoint','validation']} for r in rows]).to_csv(PUB/'SUPPORT30_SELECTION.csv',index=False)
    pd.DataFrame(history).to_csv(PUB/'SUPPORT30_HISTORY.csv',index=False)
    write(PUB/'SUPPORT30_STATUS.json',dict(status='ALL5_SUPPORT_MATCHED_RAW_MODELS_FROZEN',model_freeze_sha256=sha(ROOT/'SUPPORT_MODEL_FREEZE_PRIVATE.pkl'),source=binding['source'],main_models_unchanged=sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl')==binding['main_model_freeze_sha256'],epochs=len(history),before_Outer=True))
    print('ALL5_TEMPORAL_SUPPORT_RAW_CONTROLS_FROZEN',flush=True)


if __name__=='__main__':main()
