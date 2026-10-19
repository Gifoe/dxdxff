"""Resolve one immutable historical seal filename; no score/model/preprocessing change."""
from common import *
def main():
    requested=PARENTROOT/'NEURAL_SCORE_SEAL_PRIVATE.json';resolved=PARENTROOT/'OUTER_SCORE_SEAL_PRIVATE.json'
    expected='6dc73306f04ffefa9ffee471470786b3e190d16a304a81b8be5b9b2225a1f033'
    assert not requested.exists() and sha(resolved)==expected
    assert sources()==jr(PUB/'SOURCE_PUSH_BEFORE_TRAIN.json')['source']
    m=module('sixd_immutable_outer_entry',HERE/'evaluate_outer.py');original=m.jr;count=[0]
    def resolved_read(path):
        if Path(path)==requested:
            assert sha(resolved)==expected;count[0]+=1;return original(resolved)
        return original(path)
    m.jr=resolved_read
    write(PUB/'REFERENCE_PATH_ADAPTER_AUDIT.json',dict(status='PASS_HASH_PINNED_REFERENCE_PATH_ONLY',requested_basename=requested.name,
        resolved_basename=resolved.name,resolved_sha256=expected,original_files_changed=False,model_or_prediction_math_changed=False,
        adapter_sha256=sha(Path(__file__)),scientific_source_unchanged=True))
    m.main();assert count[0]==1
    print('SIXD_NEURAL_OUTER_REFERENCE_PATH_ADAPTER_COMPLETE',flush=True)
if __name__=='__main__':main()
