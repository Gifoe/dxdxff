"""Call unchanged original trainer for exactly fifteen new development cells."""
import json
import torch
import pandas as pd
from runtime import args,setup,checkpoint


def main():
    a=args();lock,ref,c,core,original,trainer,models,banks,binding=setup(a)
    assert json.loads((a.runtime/'public/PRETRAIN_AUDIT.json').read_text())['binding']==binding
    tests=json.loads((a.runtime/'public/ENGINEERING_TESTS.json').read_text());assert tests['status']=='PASS' and tests['binding']==binding
    pre=json.loads((a.runtime/'public/PRETRAIN_AUDIT.json').read_text());rows=json.loads((a.runtime/'B0_SELECTIONS_PRIVATE.json').read_text());histories=[]
    for fold in range(1,6):
        bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==pre['folds'][fold-1]['bank_sha256']
        d=torch.load(bp,map_location='cpu',weights_only=False)
        for arm in ['B1','B2','B3']:
            factory=models.constructor(original.SourceA0,arm,d['stats_normalizer'])
            old=trainer.SourceA0;trainer.SourceA0=factory
            try:
                trainer.train(d['x'],d['y'],d['patient'],d['g'],d['train'],d['val'],a.runtime/f'fold{fold}/{arm}',
                    42+1009*fold,c.digest({'binding':binding,'bank':c.sha(bp),'arm':arm}),penalty=0.,device='cuda')
            finally:trainer.SourceA0=old
            cp=checkpoint(a,arm,fold);ck=torch.load(cp,map_location='cpu',weights_only=False)
            done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());assert done['best_sha256']==c.sha(cp)
            rows.append({'arm':arm,'fold':fold,'epoch':ck['epoch'],'threshold':ck['threshold'],'checkpoint_sha256':c.sha(cp),'bank_sha256':c.sha(bp),'reused':False})
            last=torch.load(cp.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
            histories.extend({'arm':arm,'fold':fold,**h} for h in last['history'])
            c.write_json(a.runtime/'public/TRAIN_PROGRESS.json',{'status':'RUNNING','new_cells_complete':len(rows)-5,'total':15,'last_arm':arm,'last_fold':fold})
            print('PMA_DEVELOPMENT_CELL_SELECTED',arm,fold,ck['epoch'],ck['threshold'],flush=True)
    frozen=a.runtime/'DEVELOPMENT_FREEZE.json';assert not frozen.exists()
    c.write_json(frozen,{'status':'PASS','binding':binding,'selections':rows,'all20_models_and_preprocessors_locked':True,'outer_accessed':False})
    pd.DataFrame(rows).to_csv(a.runtime/'public/MODEL_SELECTION.csv',index=False)
    pd.DataFrame(histories).to_csv(a.runtime/'public/TRAINING_HISTORY.csv',index=False)
    c.write_json(a.runtime/'public/TRAIN_PROGRESS.json',{'status':'ALL15_NEW_DEVELOPMENT_CELLS_COMPLETE','new_cells_complete':15,'total':15,'outer_accessed':False})
    print('ALL20_DEVELOPMENT_SELECTIONS_FROZEN',flush=True)


if __name__=='__main__':main()
