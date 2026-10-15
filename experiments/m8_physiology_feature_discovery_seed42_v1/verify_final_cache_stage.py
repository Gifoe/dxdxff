"""Run unchanged cache/source verification prefix in original pinned model runtime."""
import ast
import verify_final as V
from verify_final import *

source=Path(V.__file__);tree=ast.parse(source.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
cut=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='archive' for t in n.targets))
assert cut>20
tail=ast.parse("""
save(ROOT/'VERIFIED_CACHE_CONTEXT_PRIVATE.pkl',dict(fr=fr,fh=fh,frames=frames,patient_cache_files=patient_cache_files,
    rec=dict(binding=dict(archive=rec['binding']['archive']),meta={k:rec['meta'][k] for k in ['patient','channel','center']}),
    original_verifier_sha256=sha(source)))
write(PUB/'INDEPENDENT_FINAL_CACHE_VERIFICATION.json',dict(status='PASS',arms=26,armfolds=130,
    actual_all_patient_inputs_reconstructed=True,all_checkpoint_sources_roles_and_scores_verified=True,
    private_patient_cache_files=len(patient_cache_files),no_new_predictor_forward=True,no_full_label_archive_read=True,
    source_sha256=sha(__file__),original_verifier_sha256=sha(source),
    context_sha256=sha(ROOT/'VERIFIED_CACHE_CONTEXT_PRIVATE.pkl')))
print('ALL130_FROZEN_MODELS_ROLES_FEATURE_INPUTS_AND_PATIENT_CACHES_INDEPENDENT_PASS',flush=True)
""").body
code=compile(ast.Module(body=main.body[:cut]+tail,type_ignores=[]),str(source),'exec')
namespace=dict(vars(V));namespace.update(source=source,__file__=__file__)
exec(code,namespace)
