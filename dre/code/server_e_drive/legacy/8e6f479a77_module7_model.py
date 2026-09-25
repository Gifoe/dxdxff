import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class TemporalDynamicGraphModel(nn.Module):
    def __init__(self, abs_in_dim, diff_in_dim, edge_in_dim, hidden_dim, dropout_rate=0.3):
        super().__init__()
        
        # 1 & 2. Feature Encoders
        self.abs_encoder = nn.Sequential(
            nn.Linear(abs_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.diff_encoder = nn.Sequential(
            nn.Linear(diff_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 3. Fusion block
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 4. Graph Encoder
        self.gnn = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=edge_in_dim, heads=2, concat=False)
        self.gnn_bn = nn.BatchNorm1d(hidden_dim)
        self.gnn_dropout = nn.Dropout(dropout_rate)
        
        # 5. Temporal Encoder
        self.temporal_gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        
        # 6. Channel level EZ head
        self.ez_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, node_abs, node_diff, edge_indices, edge_attrs):
        T, N, _ = node_abs.shape
        
        # 1. 将时间步与节点维度展平，一次性让 GPU 并行计算所有的全连接特征提取
        x_abs_flat = node_abs.view(T * N, -1)
        x_diff_flat = node_diff.view(T * N, -1)
        
        x_abs_enc = self.abs_encoder(x_abs_flat)
        x_diff_enc = self.diff_encoder(x_diff_flat)
        
        x_fused = self.fusion(torch.cat([x_abs_enc, x_diff_enc], dim=-1))
        
        # 2. 将所有时间步的小动态图平移拼接成一张巨大的互相隔离的图（这是 GPU 并行 GNN 的灵魂！）
        batched_edge_indices = [edge_indices[t] + t * N for t in range(T)]
        big_edge_index = torch.cat(batched_edge_indices, dim=1)
        big_edge_attr = torch.cat(edge_attrs, dim=0)
        
        # 3. 5090 仅需通过 1 条指令全功率执行这巨大图结构的 GAT 卷积
        x_graph = self.gnn(x_fused, big_edge_index, big_edge_attr)
        x_graph = self.gnn_bn(x_graph)
        x_graph = torch.relu(x_graph)
        x_graph = self.gnn_dropout(x_graph)
        
        # 4. 重新还原到对应的时间步维度 (T, N, Hidden) 并送入 GRU 网络
        seq_input = x_graph.view(T, N, -1).transpose(0, 1) # 修改形状为 (N, T, Hidden)
        gru_out, h_n = self.temporal_gru(seq_input)
        
        chan_embs = torch.mean(gru_out, dim=1)
        logits = self.ez_head(chan_embs)
        return logits.squeeze(-1)
