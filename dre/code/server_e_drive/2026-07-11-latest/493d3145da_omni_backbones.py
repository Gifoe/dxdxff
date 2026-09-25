"""Omni-iEEG-inspired encoders adapted to 4 s / 200 Hz Task 1 windows.

upstream_repository = "Omni-iEEG/Omni-iEEG"
adaptation = "architecture adapted to 4-second, 200-Hz Task 1 windows"
"""
from __future__ import annotations

from typing import Any
import torch
from torch import nn
from torch.nn import functional as F


def samples_from_ms(ms: float, sfreq: float, *, minimum: int, odd: bool = False) -> int:
    value = max(int(minimum), int(round(float(ms) * float(sfreq) / 1000.0)))
    return value + 1 if odd and value % 2 == 0 else value


def _check(x: torch.Tensor, n_times: int) -> None:
    if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] != n_times:
        raise ValueError(f"Expected [B,1,{n_times}], got {tuple(x.shape)}.")
    if not torch.isfinite(x).all():
        raise ValueError("Raw input contains NaN or Inf.")


class _SE(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__(); self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, max(1, channels // 8), 1), nn.ReLU(), nn.Conv1d(max(1, channels // 8), channels, 1), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x * self.net(x)


class OmniSEEGNetEncoder(nn.Module):
    output_dim = 256
    def __init__(self, *, n_times: int = 800, sfreq: float = 200.0, dropout: float = .5, **_: Any) -> None:
        super().__init__(); self.n_times = int(n_times)
        kernels = [samples_from_ms(ms, sfreq, minimum=3, odd=True) for ms in (250, 80, 12.5)]
        strides = [samples_from_ms(ms, sfreq, minimum=1) for ms in (20, 10, 2.5)]
        self.kernel_sizes, self.strides = kernels, strides
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv1d(1, 32, k, stride=s, padding=k//2), nn.BatchNorm1d(32), nn.ELU()) for k,s in zip(kernels,strides)])
        self.dropout = nn.Dropout(dropout); self.residual = nn.Sequential(nn.Conv1d(96, 96, 1), nn.BatchNorm1d(96)); self.se = _SE(96)
        self.lstm = nn.LSTM(96, 128, batch_first=True, bidirectional=True); self.attn = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check(x, self.n_times); parts=[b(x) for b in self.branches]; t=min(p.shape[-1] for p in parts); z=torch.cat([p[..., :t] for p in parts], dim=1); z=F.elu(self.se(self.residual(z))+z); z=self.dropout(z).transpose(1,2); z,_=self.lstm(z); a=torch.softmax(self.attn(z).squeeze(-1),dim=1); return torch.sum(z*a.unsqueeze(-1),dim=1)


class TimeFrequencyFrontEnd(nn.Module):
    def __init__(self, *, n_fft: int = 256, hop_length: int = 20, n_bins: int = 224) -> None:
        super().__init__(); self.n_fft=n_fft; self.hop_length=hop_length; self.n_bins=n_bins; self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec=torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(x), return_complex=True, center=True).abs(); spec=torch.log1p(spec); spec=F.interpolate(spec.unsqueeze(1), size=(self.n_bins, spec.shape[-1]), mode="bilinear", align_corners=False); return (spec-spec.mean((2,3),keepdim=True))/spec.std((2,3),keepdim=True).clamp_min(1e-6)


class OmniTimeConvCNNEncoder(nn.Module):
    output_dim = 128
    def __init__(self, *, n_times: int=800, sfreq: float=200., timeconv_pretrained_resnet: bool=False, timeconv_frontend: str="stft", **_: Any) -> None:
        super().__init__(); self.n_times=int(n_times); self.frontend_type=str(timeconv_frontend)
        if self.frontend_type != "stft": raise ValueError("Only stable STFT frontend is implemented in this portable bundle.")
        self.frontend=TimeFrequencyFrontEnd(); self.temporal=nn.Sequential(nn.Conv2d(1,16,(3,5),(1,5),padding=(1,0)),nn.ReLU(),nn.Conv2d(16,32,(3,2),(1,2),padding=(1,0)),nn.ReLU())
        try:
            from torchvision.models import ResNet18_Weights, resnet18
            weights=ResNet18_Weights.IMAGENET1K_V1 if timeconv_pretrained_resnet else None
            backbone=resnet18(weights=weights); old=backbone.conv1; backbone.conv1=nn.Conv2d(32,64,old.kernel_size,old.stride,old.padding,bias=False); backbone.fc=nn.Identity(); self.resnet=backbone
        except Exception as exc: raise RuntimeError("Omni-TimeConv-CNN requires torchvision and requested ResNet weights must be locally available.") from exc
        self.projection=nn.Linear(512,128)
    def forward(self,x:torch.Tensor)->torch.Tensor: _check(x,self.n_times); return self.projection(self.resnet(self.temporal(self.frontend(x))))


class OmniCLAPEncoder(nn.Module):
    def __init__(self, *, n_times:int=800, sfreq:float=200., clap_model_id:str="laion/clap-htsat-fused", clap_local_files_only:bool=False, clap_train_mode:str="audio_finetune", **_:Any)->None:
        super().__init__(); self.n_times=int(n_times); self.sampling_rate=float(sfreq)
        if self.n_times != 800 or self.sampling_rate != 200.: raise ValueError("Omni-CLAP requires the adapted 800-sample, 200-Hz contract.")
        try:
            from transformers import ClapAudioModelWithProjection, ClapFeatureExtractor
            self.feature_extractor=ClapFeatureExtractor(sampling_rate=200, chunk_length_s=4, max_length_s=4, n_mels=64, f_min=0, f_max=100, n_fft=256, hop_length=20)
            self.clap=ClapAudioModelWithProjection.from_pretrained(clap_model_id, local_files_only=clap_local_files_only)
        except Exception as exc: raise RuntimeError(f"CLAP weights unavailable for {clap_model_id}; run prepare_checkpoints.sh --clap or disable local-files-only explicitly.") from exc
        self.output_dim=int(self.clap.config.projection_dim); self._set_mode(clap_train_mode)
    def _set_mode(self, mode:str)->None:
        if mode not in {"frozen","audio_finetune","full"}: raise ValueError("clap_train_mode must be frozen, audio_finetune, or full.")
        for p in self.clap.parameters(): p.requires_grad=(mode=="full")
        if mode=="audio_finetune":
            for name,p in self.clap.named_parameters(): p.requires_grad=("audio_model" in name or "audio_projection" in name)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        _check(x,self.n_times); device=next(self.clap.parameters()).device; feat=self.feature_extractor(x.squeeze(1).detach().cpu().numpy(), sampling_rate=200, return_tensors="pt", padding=True); feat={k:v.to(device) for k,v in feat.items()}; return self.clap(**feat).audio_embeds


def build_omni_raw_encoder(name:str, *, n_times:int, sfreq:float, config:dict|None=None)->nn.Module:
    key=name.lower().replace("_","").replace("-",""); config=config or {}
    if key=="omniseegnet": return OmniSEEGNetEncoder(n_times=n_times,sfreq=sfreq,**config)
    if key=="omnitimeconvcnn": return OmniTimeConvCNNEncoder(n_times=n_times,sfreq=sfreq,**config)
    if key=="omniclap": return OmniCLAPEncoder(n_times=n_times,sfreq=sfreq,**config)
    raise ValueError(f"Unknown Omni raw encoder: {name}")
