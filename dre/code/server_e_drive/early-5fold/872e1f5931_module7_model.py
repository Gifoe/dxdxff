import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        
        self.downsample = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        res = self.downsample(x)
        out = self.fc(x)
        out = self.ln(out)
        out = self.act(out)
        out = self.drop(out)
        return out + res

class TemporalDynamicGraphModel(nn.Module):
    def __init__(self, abs_in_dim, diff_in_dim, edge_in_dim, hidden_dim=32, num_layers=2):
        super().__init__()
        
        self.abs_encoder = ResidualBlock(abs_in_dim, hidden_dim)
        self.diff_encoder = ResidualBlock(diff_in_dim, hidden_dim)

        # Output branch channel logits
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn_fusion = nn.LayerNorm(hidden_dim)

        self.gat_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_in_dim)
            for _ in range(num_layers)
        ])
        
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.ez_head = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.GELU(),
            nn.Linear(int(hidden_dim/2), 1)
        )
        
        # New branch: Run-level pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, node_abs, node_diff, edge_indices, edge_attrs, return_embeddings=False):
        # T windows
        T = node_abs.size(0)
        N = node_abs.size(1)

        # Vectorized feature computation across time
        x_a = self.abs_encoder(node_abs)  # (T, N, H)
        x_d = self.diff_encoder(node_diff)  # (T, N, H)
        
        x_f = torch.cat([x_a, x_d], dim=-1)
        x_f = self.fusion_fc(x_f)
        
        if N > 1:
            x_f = self.bn_fusion(x_f)
        x_f = F.gelu(x_f)

        if T > 0:
            x_f_flat = x_f.view(T * N, -1)
            batch_ei, batch_ea = [], []
            for t in range(T):
                batch_ei.append(edge_indices[t] + t * N)
                batch_ea.append(edge_attrs[t])
                
            b_ei = torch.cat(batch_ei, dim=1)
            b_ea = torch.cat(batch_ea, dim=0)
            
            x = x_f_flat
            for gat in self.gat_layers:
                x_new = gat(x, b_ei, b_ea)
                x = F.gelu(x_new) + x
                
            window_embs = x.view(T, N, -1)
        else:
            window_embs = x_f

        # Shape: (N, T, H) for GRU
        window_embs = window_embs.transpose(0, 1)
        
        gru_out, _ = self.temporal_gru(window_embs)
        
        # Takes last timestep as channel embedding
        chan_embs = gru_out[:, -1, :] # (N, H)
        
        channel_logits = self.ez_head(chan_embs).squeeze(-1) # (N,)
        
        if return_embeddings:
            # 1. Mean pool
            mean_p = chan_embs.mean(dim=0)
            # 2. Max pool
            max_p = chan_embs.max(dim=0)[0]
            # 3. Attention pool
            attn_scores = self.attn_pool(chan_embs) # (N, 1)
            attn_weights = F.softmax(attn_scores, dim=0)
            attn_p = (chan_embs * attn_weights).sum(dim=0) # (H,)
            
            run_emb = torch.cat([mean_p, max_p, attn_p], dim=-1) # (3*H,)
            
            return {
                "channel_logits": channel_logits,
                "chan_embs": chan_embs,
                "run_emb": run_emb
            }
        
        return channel_logits