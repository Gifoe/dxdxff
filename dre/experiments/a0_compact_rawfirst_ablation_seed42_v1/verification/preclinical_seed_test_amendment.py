"""Preserve initial freeze; repair only preclinical test-seed metadata binding."""
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from audit_sources import arguments,context


def main():
 tag=sys.argv.pop(1) if len(sys.argv)>1 and not sys.argv[1].startswith('--') else 'seed'
 assert tag in ['seed','gradient_and_resume_fixture']
 a=arguments();o=context(a);c=o[2];newbind=o[-1];pub=a.runtime/'public';xp=a.runtime/'EXPERIMENT_FREEZE.json';old=json.loads(xp.read_text())
 assert not (a.runtime/'raw_training').exists() and not (pub/'RAW_OUTER_STATUS.json').exists()
 assert old['binding']['original']==newbind['original']
 before=old['binding']['new'];after=newbind['new'];assert set(before)==set(after)
 changed=[k for k in before if before[k]!=after[k]];assert changed==['tests_contract.py']
 archived_test=a.source/f'a0_compact_rawfirst_ablation_seed42_v1/engineering_archive/contract_before_{tag}_gate.py'
 assert c.sha(archived_test)==before['tests_contract.py']
 archive=a.runtime/f'engineering_archive/{tag}_test_amendment';archive.mkdir(parents=True,exist_ok=False)
 paths=[xp,pub/'BASELINE_REPRODUCTION.json',pub/'EXPERIMENT_MATRIX.json',pub/'FEATURE_PRETRAIN_AUDIT.json']
 if (pub/'PRECLINICAL_ENGINEERING_AMENDMENT.json').exists():paths.append(pub/'PRECLINICAL_ENGINEERING_AMENDMENT.json')
 source_hashes={p.name:c.sha(p) for p in paths}
 for p in paths:shutil.copy2(p,archive/p.name);assert c.sha(p)==c.sha(archive/p.name)
 for r in json.loads((pub/'FEATURE_PRETRAIN_AUDIT.json').read_text())['folds']:
  assert c.sha(a.runtime/f"features/{r['arm']}/fold{r['fold']}/BANK_PRIVATE.pt")==r['bank_sha256']
 for name in ['BASELINE_REPRODUCTION.json','FEATURE_PRETRAIN_AUDIT.json']:
  p=pub/name;d=json.loads(p.read_text());assert d['binding']==old['binding'];d['binding']=newbind;d['engineering_test_seed_amendment_only']=True;c.write_json(p,d)
 old['binding']=newbind;old['previous_experiment_freeze_sha256']=source_hashes['EXPERIMENT_FREEZE.json'];old['engineering_test_seed_amendment_only']=True;c.write_json(xp,old)
 d=json.loads((pub/'EXPERIMENT_MATRIX.json').read_text());d['experiment_freeze_sha256']=c.sha(xp);d['previous_experiment_freeze_sha256']=source_hashes['EXPERIMENT_FREEZE.json'];c.write_json(pub/'EXPERIMENT_MATRIX.json',d)
 c.write_json(pub/'PRECLINICAL_ENGINEERING_AMENDMENT.json',dict(status='PASS',amendment_type=tag,changed_binding_keys=changed,previous_generated_hashes=source_hashes,
  previous_test_sha256=before['tests_contract.py'],corrected_test_sha256=after['tests_contract.py'],new_experiment_freeze_sha256=c.sha(xp),
  training_started=False,new_raw_metrics_viewed=False,no_data_model_loss_split_threshold_or_hyperparameter_change=True,
  raw_registry_still_binds_initial_same_scientific_freeze=True,initial_freeze_and_audits_preserved=True,script_sha256=c.sha(Path(__file__))))
 print('PRECLINICAL_TEST_SEED_ONLY_AMENDMENT_PASS',flush=True)


if __name__=='__main__':main()
