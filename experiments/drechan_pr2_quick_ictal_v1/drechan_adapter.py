"""One audited AST observation exposes original Raw96 without changing math."""
import ast,copy,inspect,textwrap
import torch
from torch import nn

def expose_raw96(model):
    owner=next(c for c in type(model).__mro__ if 'features' in c.__dict__ and
               'vector' in c.__dict__['features'].__code__.co_varnames)
    if getattr(owner.features,'_pr2_observer',False):return
    old=owner.features;tree=ast.parse(textwrap.dedent(inspect.getsource(old)))
    function=tree.body[0];assert isinstance(function.body[-1],ast.Return)
    original=ast.dump(tree,include_attributes=False)
    observe=ast.parse('self._pr2_raw96 = vector.to(raw.dtype)').body[0]
    function.body.insert(-1,observe);ast.fix_missing_locations(tree)
    check=copy.deepcopy(tree);check.body[0].body.pop(-2)
    assert ast.dump(check,include_attributes=False)==original
    local={};exec(compile(tree,inspect.getsourcefile(old),'exec'),old.__globals__,local)
    replacement=local['features'];replacement._pr2_observer=True
    owner.features=replacement

class DREChanAdapter(nn.Module):
    def __init__(self,base,plugin=None,ranking=True,calibration=True):
        super().__init__();self.base=base;self.plugin=plugin
        self.ranking_enabled=ranking;self.calibration_enabled=calibration;expose_raw96(base)
    def forward(self,x,raw,mask,times,delta,scale,phys12,enabled=True):
        original=self.base(x,raw,mask,times,delta,scale)
        base_ez=-original['logit']
        h=torch.cat([self.base._pr2_raw96,phys12],1)
        assert h.shape==(len(x),108) and torch.isfinite(h).all()
        if self.plugin is None or not enabled:
            zero=torch.zeros_like(base_ez)
            return dict(anchor=base_ez,rank=base_ez,final=base_ez,delta_rank=zero,delta_cal=zero,h=h)
        out=self.plugin(h,base_ez,ranking=self.ranking_enabled,calibration=self.calibration_enabled);out['h']=h;return out
