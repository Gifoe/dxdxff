"""Outcome-free native-versus-safe equivalence, Unicode/score ties/singletons."""
import faulthandler
faulthandler.enable()
from audit_old7 import *
from ranking_safe import ranking
rng=np.random.default_rng(42);cases=0
for n in [1,2,3,5,16,64,127,128,256,512,1024]:
    for kind in ['random','ties','all_equal']:
        p=rng.random(n) if kind=='random' else np.round(rng.random(n),1) if kind=='ties' else np.full(n,.5)
        y=rng.integers(0,2,n);y[0]=1
        ch=np.asarray([('β' if i%3==0 else 'é' if i%3==1 else 'ASCII')+f'_{i:04d}' for i in rng.permutation(n)])
        print('RANKING_SYNTHETIC_CASE',n,kind,flush=True)
        native=EV.ranking(y,p,ch);safe=ranking(y,p,ch)
        assert native.keys()==safe.keys()
        assert all(np.isclose(native[k],safe[k],atol=1e-12,rtol=0,equal_nan=True) for k in native)
        cases+=1
assert all(np.isnan(v) for v in ranking(np.zeros(7,int),np.arange(7)/7,[str(i) for i in range(7)]).values())
write(OUT/'public/RANKING_SORT_EQUIVALENCE.json',dict(status='PASS',synthetic_cases=cases,all_outputs_equal_atol1e_12=True,
    unicode_and_exact_score_ties=True,singletons_and_no_positive_support=True,clinical_scores_labels_unread=True,
    source_sha256=sha(TASK/'ranking_safe.py'),test_sha256=sha(__file__)))
print('ALL33_NATIVE_AND_SAFE_CANONICAL_RANKING_CASES_PASS',flush=True)
