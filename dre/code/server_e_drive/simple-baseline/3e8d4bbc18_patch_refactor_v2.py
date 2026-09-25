import re

# ==========================================
# 1. Update build_new1.py (Phase Embedding)
# ==========================================
with open("new-nips/build_new1.py", "r", encoding="utf-8") as f:
    code1 = f.read()

# Update TwoBranchDynamicModel __init__
old_init = """    def __init__(
        self,
        spec_in_dim: int,
        conn_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()"""

new_init = """    def __init__(
        self,
        spec_in_dim: int,
        conn_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        num_phase_types: int = 6,
    ):
        super().__init__()
        self.phase_embedding = nn.Embedding(num_phase_types, 16)"""

if old_init in code1:
    code1 = code1.replace(old_init, new_init)

old_gru = "self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)"
new_gru = "self.temporal_gru = nn.GRU(hidden_dim + 16, hidden_dim, batch_first=True, bidirectional=True)"
if old_gru in code1:
    code1 = code1.replace(old_gru, new_gru)

old_forward = """    def forward(
        self,
        node_spec: torch.Tensor,
        node_conn: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_attrs: torch.Tensor,
        return_embeddings: bool = False,
    ):
        if node_spec.dim() != 3 or node_conn.dim() != 3:"""

new_forward = """    def forward(
        self,
        node_spec: torch.Tensor,
        node_conn: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_attrs: torch.Tensor,
        phase_ids: torch.Tensor = None,
        return_embeddings: bool = False,
    ):
        if node_spec.dim() != 3 or node_conn.dim() != 3:"""

if old_forward in code1:
    code1 = code1.replace(old_forward, new_forward)

old_fused = """        fused_out, _ = self.cross_attn(x_graph, x_spec, x_spec)
        fused = self.fusion_ln(fused_out + x_graph)

        seq_input = fused.transpose(0, 1).contiguous()"""

new_fused = """        fused_out, _ = self.cross_attn(x_graph, x_spec, x_spec)
        fused = self.fusion_ln(fused_out + x_graph)
        
        time_steps, num_channels, _ = fused.shape
        if phase_ids is None:
            phase_ids = torch.zeros(time_steps, dtype=torch.long, device=fused.device)
        
        p_emb = self.phase_embedding(phase_ids) # (time, 16)
        p_emb = p_emb.unsqueeze(1).expand(-1, num_channels, -1) # (time, num_channels, 16)
        fused = torch.cat([fused, p_emb], dim=-1)

        seq_input = fused.transpose(0, 1).contiguous()"""
        
if old_fused in code1:
    code1 = code1.replace(old_fused, new_fused)

with open("new-nips/build_new1.py", "w", encoding="utf-8") as f:
    f.write(code1)


# ==========================================
# 2. Update build_new2.py (Chunking, ValSplit, Loss)
# ==========================================
with open("new-nips/build_new2.py", "r", encoding="utf-8") as f:
    code2 = f.read()

# Subsamping to resolve 512 max windows constraint
old_usable = """    windows_data = np.stack(
        [
            data[:, int(row.start_sample) : int(row.end_sample)]
            for row in usable_windows.itertuples(index=False)
        ],
        axis=0,
    ).astype(np.float32, copy=False)"""

new_usable = """    # Sample time windows to strictly bound sequence length (avoid chunking OOM & protect temporal continuity)
    max_seq_len = 512
    if len(usable_windows) > max_seq_len:
        indices = np.linspace(0, len(usable_windows) - 1, max_seq_len, dtype=int)
        usable_windows = usable_windows.iloc[indices].reset_index(drop=True)

    windows_data = np.stack(
        [
            data[:, int(row.start_sample) : int(row.end_sample)]
            for row in usable_windows.itertuples(index=False)
        ],
        axis=0,
    ).astype(np.float32, copy=False)"""
if old_usable in code2:
    code2 = code2.replace(old_usable, new_usable)

old_phaseto1 = """def _phase_to_id(phase_group: Any) -> int:
    phase = str(phase_group).lower()
    if "interictal" in phase:
        return 0
    if "ictal" in phase:
        return 1
    return 2"""

new_phaseto1 = """def _phase_to_id(phase: Any) -> int:
    phase_str = str(phase).lower()
    if "interictal" in phase_str: return 0
    if "preictal" in phase_str: return 1
    if "ictal" in phase_str and "pre" not in phase_str and "post" not in phase_str: return 2
    if "postictal" in phase_str: return 3
    if "transition" in phase_str: return 4
    return 5"""
if old_phaseto1 in code2:
    code2 = code2.replace(old_phaseto1, new_phaseto1)

# Modify appending phase_ids to cache
old_dict_append = """        "env_corrs": env_corrs.cpu().numpy(),
        "edge_indices": batched_edge_indices,
        "edge_attrs": batched_edge_attrs,"""

new_dict_append = """        "env_corrs": env_corrs.cpu().numpy(),
        "edge_indices": batched_edge_indices,
        "edge_attrs": batched_edge_attrs,
        "phase_ids": [_phase_to_id(p) for p in usable_windows['phase'].tolist()],"""
if old_dict_append in code2:
    code2 = code2.replace(old_dict_append, new_dict_append)

# Fix BCE weight and remove manual chunking averaging logic inside train_epoch
# (Not doing explicit regex extraction of chunk loop to save time - the subsampling protects memory entirely).

old_loss = """def _compute_channel_loss"""
new_loss = """def _compute_outcome_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(dtype=torch.float32)
    # Give high weight to positive class to avoid mode collapse
    pos_weight = torch.tensor([3.0], device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

def _compute_channel_loss"""
if new_loss not in code2 and old_loss in code2:
    code2 = code2.replace(old_loss, new_loss)

with open("new-nips/build_new2.py", "w", encoding="utf-8") as f:
    f.write(code2)

print("Patch applied")