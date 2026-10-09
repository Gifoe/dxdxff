# Copyright 2024 Yandex LLC. Apache-2.0; see ../LICENSE_TABM.txt.
# Modified: isolated components, docstrings removed; operations/initialization unchanged.
# Source: yandex-research/tabm@28e47ae301c92ec37787dde1ce923a0793f405b4
import warnings
from typing import Literal,Optional,Union
import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
_INTERNAL_ERROR_MESSAGE="Internal error (this code must be unreachable; please, report a bug)"
ScalingRandomInitialization=Literal["random-signs","normal"]
ScalingInitialization=Literal[ScalingRandomInitialization,"ones"]

def _check_positive_integer(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer, however: {name}={value}')

def _check_input_ndim(value: Tensor, name: str, required_ndim: int) -> None:
    if value.ndim != required_ndim:
        raise ValueError(f'The input must have {required_ndim} dimensions, however: {name}.ndim={value.ndim}')

def _check_ensemble_input_shape(name: str, value: Tensor, k: int) -> None:
    _check_input_ndim(value, name, 3)
    if value.shape[-2] != k:
        raise ValueError(f'The penultimate input dimension must be equal to the ensemble size k={k!r}, however: {name}.shape[-2]={value.shape[-2]}')

def _init_rsqrt_uniform_(tensor: Tensor, d: int) -> Tensor:
    assert d > 0, _INTERNAL_ERROR_MESSAGE
    d_rsqrt = d ** (-0.5)
    return nn.init.uniform_(tensor, -d_rsqrt, d_rsqrt)

@torch.inference_mode()
def _init_random_signs_(tensor: Tensor) -> Tensor:
    return tensor.bernoulli_(0.5).mul_(2).add_(-1)

def init_scaling_(x: Tensor, distribution: ScalingInitialization, chunks: Optional[list[int]]=None) -> Tensor:
    if distribution == 'ones':
        if chunks is not None:
            raise ValueError(f'When distribution={distribution!r}, chunks must be None')
        init_fn = nn.init.ones_
    elif distribution == 'normal':
        init_fn = nn.init.normal_
    elif distribution == 'random-signs':
        init_fn = _init_random_signs_
    else:
        raise ValueError(f'Unknown distribution={distribution!r}')
    if chunks is None:
        return init_fn(x)
    else:
        if x.ndim < 1:
            raise ValueError(f'When chunks is not None, the input tensor must have at least onedimension, however: x.ndim={x.ndim!r}')
        if sum(chunks) != x.shape[-1]:
            raise ValueError(f'The tensor shape and chunks are incompatible: x.shape[-1]={x.shape[-1]!r} != sum(chunks)={sum(chunks)!r}')
        with torch.inference_mode():
            chunk_start = 0
            for chunk_size in chunks:
                x[..., chunk_start:chunk_start + chunk_size] = init_fn(torch.empty(*x.shape[:-1], 1))
                chunk_start += chunk_size
        return x

def ensemble_view(x: Tensor, k: int, training: bool) -> Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(-2).expand(-1, k, -1)
    elif x.ndim == 3:
        if x.shape[-2] != k:
            raise ValueError(f'The penultimate input dimension must be equal to k={k}, however: x.shape[-2]={x.shape[-2]!r}')
        if not training:
            warnings.warn(f'When training=False, the input should usually have the shape (batch_size, d), i.e. hold one representation per object. However: x.shape={x.shape!r}. Is this intentional?')
    else:
        raise ValueError(f'The must must have either two or three dimensions, however: x.ndim={x.ndim!r}')
    return x

class EnsembleView(nn.Module):

    def __init__(self, *, k: int) -> None:
        super().__init__()
        self._k = k

    @property
    def k(self) -> int:
        return self._k

    def forward(self, x: Tensor) -> Tensor:
        return ensemble_view(x, self.k, self.training)

class LinearEnsemble(nn.Module):
    bias: Optional[Tensor]

    def __init__(self, in_features: int, out_features: int, bias: bool=True, *, k: int, dtype: Optional[torch.dtype]=None, device: Optional[Union[str, torch.dtype]]=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty(k, in_features, out_features, **factory_kwargs))
        self.register_parameter('bias', Parameter(torch.empty(k, out_features, **factory_kwargs)) if bias else None)
        self.reset_parameters()

    @classmethod
    def from_linear(cls, module: nn.Linear, **kwargs) -> 'LinearEnsemble':
        kwargs.setdefault('dtype', module.weight.dtype)
        kwargs.setdefault('device', module.weight.device)
        return cls(module.in_features, module.out_features, module.bias is not None, **kwargs)

    @property
    def in_features(self) -> int:
        return self.weight.shape[-2]

    @property
    def out_features(self) -> int:
        return self.weight.shape[-1]

    @property
    def k(self) -> int:
        return self.weight.shape[0]

    def reset_parameters(self):
        d = self.in_features
        _init_rsqrt_uniform_(self.weight, d)
        if self.bias is not None:
            _init_rsqrt_uniform_(self.bias, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_ensemble_input_shape('x', x, self.k)
        if x.shape[-1] != self.in_features:
            raise ValueError(f'The last input dimension must be equal to {self.in_features}, however: {x.shape[-1]}')
        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x

class LinearBatchEnsemble(nn.Module):
    bias: Optional[Tensor]

    def __init__(self, in_features: int, out_features: int, bias: bool=True, *, k: int, scaling_init: Union[ScalingInitialization, tuple[ScalingInitialization, ScalingInitialization]], first_scaling_init_chunks: Optional[list[int]]=None, dtype: Optional[torch.dtype]=None, device: Optional[Union[str, torch.dtype]]=None):
        _check_positive_integer(in_features, 'in_features')
        _check_positive_integer(out_features, 'out_features')
        _check_positive_integer(k, 'k')
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.r = Parameter(torch.empty(k, in_features, **factory_kwargs))
        self.s = Parameter(torch.empty(k, out_features, **factory_kwargs))
        self.register_parameter('bias', Parameter(torch.empty(k, out_features, **factory_kwargs)) if bias else None)
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        if isinstance(scaling_init, tuple):
            self._first_scaling_init = scaling_init[0]
            self._second_scaling_init = scaling_init[1]
        else:
            self._first_scaling_init = scaling_init
            self._second_scaling_init = scaling_init
        self._first_scaling_init_chunks = first_scaling_init_chunks
        self.reset_parameters()

    @classmethod
    def from_linear(cls, module: nn.Linear, **kwargs) -> 'LinearBatchEnsemble':
        kwargs.setdefault('dtype', module.weight.dtype)
        kwargs.setdefault('device', module.weight.device)
        return cls(module.in_features, module.out_features, module.bias is not None, **kwargs)

    def reset_parameters(self) -> None:
        _init_rsqrt_uniform_(self.weight, self.in_features)
        init_scaling_(self.r, self._first_scaling_init, self._first_scaling_init_chunks)
        init_scaling_(self.s, self._second_scaling_init, None)
        if self.bias is not None:
            bias_init = torch.empty(self.out_features, dtype=self.weight.dtype, device=self.weight.device)
            bias_init = _init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        _check_input_ndim(x, 'x', 3)
        x = x * self.r
        x = x @ self.weight.T
        x = x * self.s
        if self.bias is not None:
            x = x + self.bias
        return x

