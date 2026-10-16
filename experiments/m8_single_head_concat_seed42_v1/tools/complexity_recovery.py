"""Reporting-only historical Raw checkpoint key repair; immutable source retained."""
import sys
import hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *

def main():
    binding=jr(ROOT/'INPUT_FREEZE_PRIVATE.json');original=HERE/'complexity.py'
    assert sha(original)==binding['source']['complexity.py']
    source=original.read_text(encoding='utf-8')
    old="rm.load_state_dict(torch.load(r['checkpoint'],map_location='cuda',weights_only=False)['model'])"
    new="rm.load_state_dict(torch.load(r['checkpoint'],map_location='cuda',weights_only=False)['state'])"
    assert source.count(old)==1
    modified=source.replace(old,new)
    namespace={'__file__':str(original),'__name__':'single_head_complexity_reporting_recovery'}
    exec(compile(modified,str(original),mode='exec'),namespace)
    namespace['main']()
    write(PUB/'COMPLEXITY_RECOVERY_AUDIT.json',dict(status='PASS',original_source_sha256=sha(original),
        executed_source_sha256=hashlib.sha256(modified.encode()).hexdigest(),recovery_source_sha256=sha(__file__),
        only_change="Historical Raw-only benchmark checkpoint ['model'] -> ['state']",new_models_or_training_changed=False,
        candidate_sha256_unchanged=sha(PUB/'CANDIDATE_FROZEN.json'),original_failure='KeyError:model',
        failure_stage='label_free_FIT_complexity_before_any_new_Outer',original_source_preserved=True))
    print('SINGLE_HEAD_REPORTING_ONLY_CHECKPOINT_NAMESPACE_RECOVERY_PASS',flush=True)

if __name__=='__main__':main()
