"""Synthetic atomic cache/hash and binding tamper controls; no clinical outcomes."""
import tempfile
from evaluate_final_v2 import patient_cache_binding
from integrated_data_v2 import *
scratch=OUT/'synthetic_test_scratch';scratch.mkdir(exist_ok=True)
x=np.arange(60,dtype=np.float32).reshape(3,20);indices=np.asarray([1,3,4]);scale=np.ones(10,np.float32)
arguments=['synthetic_freeze','synthetic_checkpoint',x,indices,{'raw':'synthetic_hash'},'integrated',list(range(10)),False,scale]
binding=patient_cache_binding(*arguments)
assert binding==patient_cache_binding(*arguments)  # Aliases have no model-name cache discriminator.
changes={0:'other_freeze',1:'other_checkpoint',2:x+1,3:indices+1,4:{'raw':'changed_hash'},5:'legacy',6:list(range(9))+[None],7:True,8:scale*2}
for i,value in changes.items():
    other=list(arguments);other[i]=value;assert patient_cache_binding(*other)!=binding,i
with tempfile.TemporaryDirectory(prefix='final_cache_unit_',dir=scratch) as directory:
    p=Path(directory)/'SYNTHETIC.pkl';save(p,dict(binding=binding,logit=np.asarray([.1,.2,.3],np.float32)))
    q=read(p);assert q['binding']==binding
    original=p.read_bytes();p.write_bytes(original+b'tamper')
    try:read(p)
    except AssertionError:pass
    else:raise AssertionError('Tampered content accepted')
write(OUT/'public/FINAL_CACHE_PROTOCOL_UNIT_TEST.json',dict(status='PASS',synthetic_only=True,
    all_nine_input_binding_changes_rejected=True,alias_binding_exact=True,atomic_roundtrip=True,content_tampering_rejected=True,
    source_sha256=sha(TASK/'evaluate_final_v2.py'),test_sha256=sha(__file__),clinical_labels_predictions_unread=True))
print('FINAL_PRIVATE_CACHE_SOURCE_FREEZE_INPUT_SCALE_ALIAS_AND_TAMPER_PASS',flush=True)
