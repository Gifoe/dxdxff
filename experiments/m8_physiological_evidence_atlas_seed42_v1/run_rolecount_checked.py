"""One precisely scoped assertion repair; frozen statistical body unchanged."""
import ast
import atlas as q


def main():
    rec=q.read(q.BASE/'RECONSTRUCTION_PRIVATE.pkl');meta=rec['meta']
    expected={f:len(set(meta['split_patient'][(meta['split_fold']==f)&(meta['split_role']=='fit')])) for f in range(1,6)}
    assert expected=={1:51,2:51,3:50,4:52,5:51}
    source=ast.parse((q.HERE/'atlas.py').read_text(encoding='utf-8'))
    function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='audit_old')
    wanted=ast.parse('assert len(g)==g.patient.nunique()==51').body[0];replacements=0
    for i,node in enumerate(function.body):
        if not isinstance(node,ast.For):continue
        for j,statement in enumerate(node.body):
            if ast.dump(statement,include_attributes=False)==ast.dump(wanted,include_attributes=False):
                node.body[j]=ast.parse('assert len(g)==g.patient.nunique()==expected_fit_patient_counts[key[0]]').body[0];replacements+=1
    assert replacements==1
    q.expected_fit_patient_counts=expected
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(q.HERE/'atlas.py'),'exec'),q.__dict__)
    q.ROOT=q.ROOT/'rolecount_checked';q.PUB=q.ROOT/'public';q.ROOT.mkdir(exist_ok=True);q.PUB.mkdir(exist_ok=True)
    original=q.freeze
    def freeze(path,value):
        if path.name=='INPUT_FREEZE_PRIVATE.json':
            value=dict(value,admission_guard_amendment={n:q.sha(q.HERE/n) for n in ['ADMISSION_AMENDMENT.json','run_rolecount_checked.py']},
                exact_original_FIT_counts=expected)
        original(path,value)
    q.freeze=freeze;q.main()


if __name__=='__main__':main()
