"""Read hash-pinned upstream component definitions without unrelated dependencies."""
import ast
import copy
import types
import warnings
from pathlib import Path
from typing import Literal,Optional,Union
import torch
from torch import Tensor,nn
from torch.nn.parameter import Parameter
from base import sha

NAMES={'_check_positive_integer','_check_input_ndim','_check_ensemble_input_shape','_init_rsqrt_uniform_','_init_random_signs_','init_scaling_','ensemble_view','EnsembleView','LinearEnsemble','LinearBatchEnsemble'}


class Strip(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
        return node

    visit_ClassDef=visit_FunctionDef


def definitions(path):
    t=ast.parse(Path(path).read_text(encoding='utf-8'))
    return {n.name:n for n in t.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in NAMES}


def official(path):
    assert sha(path)=='fc654af6a16bac53d893a8265c79d7af4ebddcb95ad0d600cc6b6bc6b7317ade'
    orig=definitions(path); vend=definitions(Path(__file__).with_name('tabm_layers.py'))
    assert set(orig)==set(vend)==NAMES
    for name in NAMES:
        assert ast.dump(Strip().visit(copy.deepcopy(orig[name])),include_attributes=False)==ast.dump(vend[name],include_attributes=False),name
    ns={'torch':torch,'nn':nn,'Tensor':Tensor,'Parameter':Parameter,'Literal':Literal,'Optional':Optional,'Union':Union,'warnings':warnings,
        '_INTERNAL_ERROR_MESSAGE':'Internal error (this code must be unreachable; please, report a bug)',
        'ScalingRandomInitialization':Literal['random-signs','normal'],'ScalingInitialization':Literal['random-signs','normal','ones']}
    module=ast.Module(body=list(orig.values()),type_ignores=[])
    exec(compile(module,str(path),'exec'),ns)
    return types.SimpleNamespace(**ns)
