import torch
import numpy as np
import pandas as pd

def compute_envelope_correlation_batched(data_t):
    """
    data_t: (B, C, T) tensor
    Returns: (B, C, C) correlation matrix
    """
    B, C, T = data_t.shape
    
    # Hilbert using FFT
    # signal.hilbert uses fft, zeroing negative frequencies, scaling positive by 2.
    fft_vals = torch.fft.fft(data_t, dim=-1)
    
    # create step function
    h = torch.zeros(T, device=data_t.device)
    if T % 2 == 0:
        h[0] = 1
        h[T // 2] = 1
        h[1:T // 2] = 2
    else:
        h[0] = 1
        h[1:(T + 1) // 2] = 2
        
    analytic_signal = torch.fft.ifft(fft_vals * h, dim=-1)
    env = torch.abs(analytic_signal) # (B, C, T)
    
    # Compute corrcoef batched
    # env: (B, C, T)
    env_mean = torch.mean(env, dim=-1, keepdim=True)
    env_centered = env - env_mean
    
    # Variance
    std = torch.sqrt(torch.sum(env_centered**2, dim=-1)) # (B, C)
    
    # Cross covariance
    # (B, C, T) matmul (B, T, C) -> (B, C, C)
    cov = torch.bmm(env_centered, env_centered.transpose(1, 2))
    
    # Outer product of std
    # (B, C, 1) * (B, 1, C) -> (B, C, C)
    std_outer = torch.bmm(std.unsqueeze(-1), std.unsqueeze(1)) + 1e-8
    
    corr = cov / std_outer
    # Fill nan with 0.0 just in case
    corr = torch.nan_to_num(corr, nan=0.0)
    
    return corr

def build_topology_edges(contacts_meta):
    """
    Creates topological edges between contacts on the same electrode group.
    """
    edge_index = []
    edge_attr = []
    
    df = contacts_meta.copy().reset_index(drop=True)
    n_nodes = len(df)
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            row_i = df.iloc[i]
            row_j = df.iloc[j]
            
            if row_i['contact_group'] == row_j['contact_group'] and pd.notna(row_i['contact_number']) and pd.notna(row_j['contact_number']):
                dist = abs(row_i['contact_number'] - row_j['contact_number'])
                if dist <= 2:
                    # [source, target]
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    
                    feat = [0.0, 1.0, float(dist), 1.0 if dist == 1 else 0.0]
                    edge_attr.append(feat)
                    edge_attr.append(feat)
                    
    edge_index_arr = torch.tensor(edge_index, dtype=torch.long).T if len(edge_index) > 0 else torch.empty((2, 0), dtype=torch.long)
    edge_attr_arr = torch.tensor(edge_attr, dtype=torch.float32) if len(edge_attr) > 0 else torch.empty((0, 4), dtype=torch.float32)
    return edge_index_arr, edge_attr_arr

def build_dynamic_graphs_batched(windows_data, contacts_meta, top_k=5, device='cpu'):
    """
    windows_data: Tensor (B, C, T)
    Returns list of edge_indices and edge_attrs lists (one per batch)
    """
    if not isinstance(windows_data, torch.Tensor):
        data_t = torch.tensor(windows_data, dtype=torch.float32, device=device)
    else:
        data_t = windows_data.to(device=device, dtype=torch.float32)
        
    B, n_channels, T = data_t.shape
    
    env_corrs = compute_envelope_correlation_batched(data_t) # (B, C, C)
    
    # Prepare meta
    # We will build meta arrays for fast attribute calculation
    df_meta = contacts_meta.copy().reset_index(drop=True)
    
    topo_edges, topo_attrs = build_topology_edges(contacts_meta)
    topo_edges = topo_edges.to(device)
    topo_attrs = topo_attrs.to(device)

    # Convert groups/numbers
    # We can encode same group as a boolean matrix
    groups = np.array(df_meta['contact_group'].tolist(), dtype=str)
    nums = np.array(df_meta['contact_number'].tolist(), dtype=float)
    
    same_group_mat = (groups[:, None] == groups[None, :])
    # Distances
    nums_mesh1, nums_mesh2 = np.meshgrid(nums, nums, indexing='ij')
    dist_mat = np.abs(nums_mesh1 - nums_mesh2)
    dist_mat[~same_group_mat | pd.isna(dist_mat)] = 999.0
    topo_adj_mat = (same_group_mat & (dist_mat == 1)).astype(float)
    
    same_group_t = torch.tensor(same_group_mat, dtype=torch.float32, device=device)
    dist_t = torch.tensor(dist_mat, dtype=torch.float32, device=device)
    topo_adj_t = torch.tensor(topo_adj_mat, dtype=torch.float32, device=device)
    
    # We ignore self loop
    eye_mask = torch.eye(n_channels, dtype=torch.bool, device=device)
    # Mask out self-loop
    env_corrs = env_corrs.masked_fill(eye_mask.unsqueeze(0), -float('inf'))
    
    # top-k
    topk_vals, topk_indices = torch.topk(env_corrs, k=top_k, dim=-1) # (B, C, top_k)
    
    # Build list of graph edges
    batched_edge_indices = []
    batched_edge_attrs = []
    
    # We do a small loop over batch since PyG requires separate graph objects
    # but we extracted topk very fast on GPU.
    for b in range(B):
        # Create src and dst
        src = torch.arange(n_channels, device=device).unsqueeze(1).expand(-1, top_k).flatten()
        dst = topk_indices[b].flatten()
        
        func_edge_index = torch.stack([src, dst], dim=0) # (2, C * top_k)
        
        # Attrs
        ec_vals = topk_vals[b].flatten()
        sg_vals = same_group_t[src, dst]
        d_vals = dist_t[src, dst]
        ta_vals = topo_adj_t[src, dst]
        
        func_edge_attr = torch.stack([ec_vals, sg_vals, d_vals, ta_vals], dim=1) # (C*top_k, 4)
        
        # Combine
        # Using pure torchtensor operations
        if func_edge_index.numel() > 0 and topo_edges.numel() > 0:
            all_edges = torch.cat([func_edge_index, topo_edges], dim=1)
            all_attrs = torch.cat([func_edge_attr, topo_attrs], dim=0)
        elif func_edge_index.numel() > 0:
            all_edges = func_edge_index
            all_attrs = func_edge_attr
        else:
            all_edges = topo_edges
            all_attrs = topo_attrs
            
        batched_edge_indices.append(all_edges.cpu().numpy())
        batched_edge_attrs.append(all_attrs.cpu().numpy())
        
    return batched_edge_indices, batched_edge_attrs
