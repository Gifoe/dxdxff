"""Release-only AST parity: new seed plumbing must not change frozen math."""
import ast
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
OLD=REPO/'experiments/drechan_loco_five_models_seed42_v1'
NEW=REPO/'experiments/drechan_loco_five_models_3seed_v1'

class Normalize(ast.NodeTransformer):
    def visit_Name(self,node):
        return ast.copy_location(ast.Constant(42),node) if node.id=='TRAIN_SEED' else node
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='dict':
            node.keywords=[k for k in node.keywords if k.arg!='train_seed']
        return node

checks=[]
for filename,names in {
    'common.py':['split_roles','labels','feature_fit','feature_query','make_source','make','rng_state','restore_rng','forward','predict','pn','select'],
    'pipeline.py':['cellpaths','fit','patient_metrics','bootstrap_draws','boot','tables'],
    'svm.py':['weights','fit','predict']
}.items():
    def read(path):
        return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    a=read(OLD/filename);b=read(NEW/filename)
    for name in names:
        x=ast.dump(Normalize().visit(a[name]),include_attributes=False)
        y=ast.dump(Normalize().visit(b[name]),include_attributes=False)
        assert x==y,(filename,name,'MATH_OR_SCIENTIFIC_LOGIC_CHANGED')
        checks.append(filename+':'+name)
assert (OLD/'timeconv_engine.py').read_text()==(NEW/'timeconv_engine.py').read_text()
print('UNCHANGED_PARENT_MATH_AST_PASS',len(checks),'functions; only training seed and summary metadata differ; TimeConv engine exact')
