import numpy as np
import pandas as pd

def compute_envelope_correlation(data):
    """
    Computes broadband envelope correlation.
    data: (n_channels, n_samples)
    """
    from scipy.signal import hilbert
    env = np.abs(hilbert(data, axis=-1))
    
    # Correlation matrix
    corr = np.corrcoef(env)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr

def build_topology_edges(contacts_meta):
    """
    creates topological edges between contacts on the same electrode.
    """
    edge_index = []
    edge_attr = []
    
    df = contacts_meta.copy().reset_index()
    n_nodes = len(df)
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            row_i = df.iloc[i]
            row_j = df.iloc[j]
            
            if row_i['stem'] == row_j['stem'] and not pd.isna(row_i['index']) and not pd.isna(row_j['index']):
                dist = abs(row_i['index'] - row_j['index'])
                if dist <= 2:
                    # Keep edge
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    
                    # same stem=1, dist=dist, topo_adj=1 if dist==1
                    feat = [1.0, float(dist), 1.0 if dist==1 else 0.0]
                    edge_attr.append(feat)
                    edge_attr.append(feat)
                    
    edge_index_arr = np.array(edge_index).T if len(edge_index) > 0 else np.empty((2, 0), dtype=int)
    return edge_index_arr, np.array(edge_attr)

def build_dynamic_graph(win_data, contacts_meta, top_k=5):
    """
    Builds the graph structure for a single window.
    Includes functional and topological edges.
    """
    n_channels = win_data.shape[0]
    
    # 1. Functional edges (Envelope correlation)
    env_corr = compute_envelope_correlation(win_data)
    
    # sparsify using top-k per node
    func_edge_index = []
    func_edge_attr = []
    
    for i in range(n_channels):
        row = env_corr[i]
        # Ignore self loop for top-k
        row[i] = -np.inf
        top_indices = np.argsort(row)[-top_k:]
        for j in top_indices:
            func_edge_index.append([i, j])
            # is_same_stem?
            stem_i = contacts_meta.iloc[i]['stem']
            stem_j = contacts_meta.iloc[j]['stem']
            dist = abs(contacts_meta.iloc[i]['index'] - contacts_meta.iloc[j]['index']) if stem_i == stem_j and pd.notna(contacts_meta.iloc[i]['index']) and pd.notna(contacts_meta.iloc[j]['index']) else 999.0
            
            same_stem = 1.0 if stem_i == stem_j else 0.0
            topo_adj = 1.0 if (same_stem and dist == 1) else 0.0
            
            # feature: [env_corr, same_stem, dist, topo_adj]
            feat = [env_corr[i, j], same_stem, dist, topo_adj]
            func_edge_attr.append(feat)
            
    # Combines func and topology edges
    topo_edges, topo_attrs = build_topology_edges(contacts_meta)
    
    # format edge attr to match: [env_corr, same_stem, dist, topo_adj]
    formatted_topo_attrs = []
    for t_attr in topo_attrs:
        formatted_topo_attrs.append([0.0, t_attr[0], t_attr[1], t_attr[2]])
        
    func_edge_index = np.array(func_edge_index).T if len(func_edge_index)>0 else np.empty((2,0))
    func_edge_attr = np.array(func_edge_attr)
    formatted_topo_attrs = np.array(formatted_topo_attrs)
    
    if func_edge_index.size > 0 and topo_edges.size > 0:
        all_edges = np.concatenate([func_edge_index, topo_edges], axis=1)
        all_attrs = np.concatenate([func_edge_attr, formatted_topo_attrs], axis=0)
    elif func_edge_index.size > 0:
        all_edges = func_edge_index
        all_attrs = func_edge_attr
    else:
        all_edges = topo_edges
        all_attrs = formatted_topo_attrs
        
    return all_edges, all_attrs
