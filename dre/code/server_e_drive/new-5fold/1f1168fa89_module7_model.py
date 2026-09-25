import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class LocalSpectralEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim, dropout=0.3):
        super().__init__()
        # 使用 1D-CNN 代替单纯的 MLP，捕捉频带的局部平滑性和异常突增
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=2)
        self.fc = nn.Linear(16 * in_features, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (T, N, F)
        T, N, F = x.shape
        x_flat = x.view(T * N, 1, F) # (Batch, Channels, Length) for Conv1d
        
        # 1D CNN over frequency domain
        out = self.act(self.conv1(x_flat))
        out = self.act(self.conv2(out))
        
        out = out.view(T * N, -1) # Flatten
        out = self.fc(out)
        out = self.ln(out)
        out = self.act(out)
        out = self.drop(out)
        
        return out.view(T, N, -1)

class GraphSpatialEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, heads=4, num_layers=2):
        super().__init__()
        self.gats = nn.ModuleList([
            GATv2Conv(
                in_channels=node_dim if i == 0 else hidden_dim, 
                out_channels=hidden_dim // heads, 
                heads=heads, 
                concat=True, 
                edge_dim=edge_dim
            )
            for i in range(num_layers)
        ])
        
    def forward(self, x, edge_index, edge_attr):
        res = x
        for gat in self.gats:
            out = gat(res, edge_index, edge_attr)
            out = F.gelu(out)
            res = out + res if res.shape == out.shape else out
        return res

class TemporalDynamicGraphModel(nn.Module):
    def __init__(self, abs_in_dim, diff_in_dim, edge_in_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        
        # 1. Local Spectral Encoder
        self.spectral_enc = LocalSpectralEncoder(abs_in_dim, hidden_dim)
        
        # 2. Graph Spatial Encoder
        self.diff_proj = nn.Sequential(
            nn.Linear(diff_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.graph_enc = GraphSpatialEncoder(hidden_dim, edge_in_dim, hidden_dim, num_layers=num_layers)

        # 3. Cross-Modal Attention Fusion
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.fusion_ln = nn.LayerNorm(hidden_dim)

        # 4. Sequential Temporal Dynamics Modeling
        self.temporal_gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # 5. Multi-Task Optimization and Readout
        self.ez_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Readout B: Graph-Level Patient Embedding using Attention Pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, node_abs, node_diff, edge_indices, edge_attrs, return_embeddings=False):
        T = node_abs.size(0)
        N = node_abs.size(1)

        # 1. Local Spectral Encoding
        x_spec = self.spectral_enc(node_abs)  # (T, N, H)

        # 2. Graph Spatial Encoding
        x_diff = self.diff_proj(node_diff)  # (T, N, H)

        # === 终极 0 循环图并行魔法 ===
        T, N, H = x_diff.shape
        x_diff_flat = x_diff.view(T * N, H)  # 拍平节点特征

        if edge_indices.numel() > 0:
            # 自动计算每个时间窗口的节点 ID 偏移量，将所有瞬间的图连成一个巨无霸网络
            offsets = (torch.arange(T, device=x_diff.device) * N).view(T, 1, 1)
            b_ei = (edge_indices + offsets).permute(1, 0, 2).reshape(2, -1)  # (2, T * E)
            b_ea = edge_attrs.view(-1, edge_attrs.size(-1))  # (T * E, 4)

            # 5090 咆哮吧：一次性计算全量图结构！
            h_g_flat = self.graph_enc(x_diff_flat, b_ei, b_ea)
            x_graph = h_g_flat.view(T, N, H)  # 完美还原回三维 (T, N, H)
        else:
            x_graph = x_diff
        
        # 3. True Cross-Modal Attention Fusion
        # 将 T 作为 Batch，N 作为 Sequence Length，H 作为 Embedding Dim
        q = x_graph # 形状 (T, N, H)
        k = x_spec  # 形状 (T, N, H)
        v = x_spec  # 形状 (T, N, H)
        
        fused_out, attn_weights_cross = self.cross_attn(q, k, v) # fused_out 形状为 (T, N, H)
        fused = self.fusion_ln(fused_out + x_graph) # (T, N, H) 残差连接
        
        # 4. Sequential Temporal Dynamics Modeling
        seq_input = fused.transpose(0, 1) # (N, T, H)
        
        gru_out, _ = self.temporal_gru(seq_input) # (N, T, H)
        
        # Extract the last time step
        chan_embs = gru_out[:, -1, :] # (N, H)
        
        # 5. Multi-Task Readout
        channel_logits = self.ez_head(chan_embs).squeeze(-1) # (N,)
        
        if return_embeddings:
            mean_p = chan_embs.mean(dim=0)
            max_p = chan_embs.max(dim=0)[0]
            
            attn_scores = self.attn_pool(chan_embs) # (N, 1)
            attn_weights = F.softmax(attn_scores, dim=0) # (N, 1)
            attn_p = (chan_embs * attn_weights).sum(dim=0) # (H,)
            
            run_emb = torch.cat([mean_p, max_p, attn_p], dim=-1) # (3*H,)
            
            return {
                "channel_logits": channel_logits,
                "chan_embs": chan_embs,
                "run_emb": run_emb
            }
        
        return channel_logits
