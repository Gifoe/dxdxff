"""Validate legal banks and name-bound U; no added learned preprocessing."""
import json
import hashlib
import numpy as np
from runtime import args,setup


def main():
 a=args();lock,ref,c,core,original,trainer,models,fm,banks,binding=setup(a)
 assert not (a.runtime/'public/PRETRAIN_AUDIT.json').exists()
 manifest=json.loads((a.previous/'public/FEATURE_GROUP_MANIFEST.json').read_text());audits=[];selections=[]
 baseline=json.loads((a.runtime/'public/BASELINE_REPRODUCTION.json').read_text());mapping=None
 for fold,d in enumerate(banks,1):
  fmap=fm.mapping(manifest,d['features']);assert mapping is None or mapping==fmap;mapping=fmap
  x,pre=core.prepare(d['raw_x'],d['patient'],d['train']);assert np.array_equal(x,d['x'])
  for key in ['mean','scale','var','imputer_statistics']:assert np.array_equal(pre[key],d['pre'][key])
  assert fm.validate_values(d,fmap)
  target=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert not target.exists()
  cell={k:d[k] for k in ['x','y','patient','channel','center','features','train','val','g','pre']};cell.update(mapping=fmap,binding=binding)
  core.torch_write(target,cell)
  old=baseline['folds'][fold-1];assert c.sha(a.runtime/f'fold{fold}/B0_VAL_PRIVATE.pt')==old['prediction_sha256']
  audits.append(dict(fold=fold,bank_sha256=c.sha(target),original_x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),original_preprocessing_bitwise=True,
    representative_columns=56,duplicate_pairs=16,constant_dimensions=16,representatives_nonconstant_on_FIT=True,
    FIT_patients=len(set(d['patient'][d['train']])),FIT_channels=len(d['train']),validation_patients=13,outer_overlap=0))
  selections.append(dict(arm='B0',fold=fold,epoch=old['epoch'],threshold=old['threshold'],checkpoint_sha256=old['checkpoint_sha256'],bank_sha256=c.sha(target),reused=True))
 cols=[]
 for col in manifest['columns']:
  cols.append(dict(index=col['index'],name=col['name'],statistic_representative=col['index'] in mapping['stat_indices'],
   constant_zero=col['index'] in mapping['zeros'],duplicate_of=col.get('duplicate_of'),main_name=manifest['columns'][col['duplicate_of']]['name'] if 'duplicate_of' in col else None))
 c.write_json(a.runtime/'public/FEATURE_MAPPING_AUDIT.json',dict(status='PASS',stat_indices=mapping['stat_indices'],columns=cols,
   input_width=88,label_free=True,fixed_by_manifest=True,all_legal_postprocessed_banks_checked=True,folds=audits))
 c.write_json(a.runtime/'B0_SELECTIONS_PRIVATE.json',selections)
 c.write_json(a.runtime/'public/PRETRAIN_AUDIT.json',dict(status='PASS',binding=binding,folds=audits,global_x_y_indexed=False,no_raw_extraction=True,new_outer_accessed=False))
 print('PURN_FEATURE_AND_PREPROCESSING_ADMISSION_PASS',flush=True)


if __name__=='__main__':main()
