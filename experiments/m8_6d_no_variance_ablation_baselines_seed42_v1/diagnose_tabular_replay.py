"""Quantify frozen classifier replay without any Outer inference or parameter change."""
from common import *
import tabular
def main():
    binding,core,fac,records,rec,banks=setup();q=read(ROOT/'TABULAR_FREEZE_PRIVATE.pkl');assert q['input']==binding
    grid=np.round(np.arange(0,1.0001,.005),3);rows=[]
    for r in q['cells']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        d=bank(r['fold'],banks,'FULL_F0');saved=read(r['checkpoint'])
        z=tabular.view(core,rec['R14'][d['archive_indices']],d['patient'],r['PR'])
        p=tabular.scores(r['classifier'],saved['bundle'],z[d['val']]);oldp=saved['validation_probability']
        sel=tabular.BL.EV.threshold((d['y'][d['val']]==0).astype(int),p,d['patient'][d['val']])
        rows.append(dict(arm=r['arm'],fold=r['fold'],max_absolute_error=float(np.max(np.abs(p-oldp))),
            different_probabilities=int((p!=oldp).sum()),all201_decisions_exact=bool(np.array_equal(p[:,None]>=grid,oldp[:,None]>=grid)),
            selected_decisions_exact=bool(np.array_equal(p>=r['threshold'],oldp>=r['threshold'])),selection_identical=sel==saved['selected']))
    csv_out('TABULAR_REPLAY_DIAGNOSIS',rows)
    print(json.dumps(dict(cells=len(rows),max_absolute_error=max(r['max_absolute_error'] for r in rows),
        changed_cells=[r for r in rows if r['different_probabilities']],all201_exact=all(r['all201_decisions_exact'] for r in rows),
        all_selected_exact=all(r['selected_decisions_exact'] for r in rows),all_selections_identical=all(r['selection_identical'] for r in rows)),sort_keys=True),flush=True)
if __name__=='__main__':main()
