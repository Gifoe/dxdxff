import torch
import torch.nn.functional as F
import numpy as np
import copy

def pairwise_ranking_loss(logits, targets):
    pos_mask = targets == 1
    neg_mask = targets == 0
    
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
        
    pos_logits = logits[pos_mask]
    neg_logits = logits[neg_mask]
    
    # Compare every pos with every neg
    # We want pos_logits > neg_logits
    diff = neg_logits.unsqueeze(1) - pos_logits.unsqueeze(0) # (num_neg, num_pos)
    
    # Softplus or margin ranking
    loss = F.softplus(diff).mean()
    return loss

def train_one_epoch(model, dataloader_list, optimizer, device):
    model.train()
    total_loss = 0
    
    for run_data in dataloader_list:
        node_abs = run_data['gpu_node_abs']
        node_diff = run_data['gpu_node_diff']
        edge_indices = run_data['gpu_edge_indices']
        edge_attrs = run_data['gpu_edge_attrs']
        labels = run_data['gpu_labels']
        
        optimizer.zero_grad()
        
        logits = model(node_abs, node_diff, edge_indices, edge_attrs)
        
        # 【优化方向二】动态权重计算，防御严重的正负样本不平衡
        num_pos = labels.sum().float()
        num_neg = (labels == 0).sum().float()
        
        if num_pos > 0:
            dyn_pos_weight = torch.clamp(num_neg / num_pos, min=1.0, max=50.0)
        else:
            # 如果极端情况一个 run 里没有正样本
            dyn_pos_weight = torch.tensor(1.0, device=device)
            
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), 
            pos_weight=torch.tensor([dyn_pos_weight], device=device)
        )
        rank_loss = pairwise_ranking_loss(logits, labels)
        
        loss = bce_loss + 0.5 * rank_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / max(len(dataloader_list), 1)

def validate_one_epoch(model, val_run_data, device):
    model.eval()
    with torch.no_grad():
        node_abs = val_run_data['gpu_node_abs']
        node_diff = val_run_data['gpu_node_diff']
        edge_indices = val_run_data['gpu_edge_indices']
        edge_attrs = val_run_data['gpu_edge_attrs']
        labels = val_run_data['gpu_labels'].cpu().numpy()
        
        logits = model(node_abs, node_diff, edge_indices, edge_attrs)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        preds = (probs > 0.5).astype(int)
        
        eps = 1e-8
        tp = np.logical_and(preds == 1, labels == 1).sum()
        fp = np.logical_and(preds == 1, labels == 0).sum()
        fn = np.logical_and(preds == 0, labels == 1).sum()
        
        precision = float(tp / (tp + fp + eps))
        recall = float(tp / (tp + fn + eps))
        f1 = float(2 * (precision * recall) / (precision + recall + eps))
        jaccard = float(tp / (tp + fp + fn + eps))
        
        return f1, jaccard, probs

def leave_one_run_out_cv(all_runs_data, model_class, model_kwargs, epochs=100, device='cpu'):
    print(f"Preloading {len(all_runs_data)} runs to GPU to blast avoiding PCIe bottlenecks...")
    for run in all_runs_data:
        if 'gpu_node_abs' not in run:
            run['gpu_node_abs'] = torch.stack(run['node_abs']).to(device)
            run['gpu_node_diff'] = torch.stack(run['node_diff']).to(device)
            run['gpu_edge_indices'] = [e.to(device) for e in run['edge_indices']]
            run['gpu_edge_attrs'] = [e.to(device) for e in run['edge_attrs']]
            run['gpu_labels'] = run['labels'].to(device)

    n_runs = len(all_runs_data)
    fold_results = {}
    
    for test_idx in range(n_runs):
        print(f"--- Fold {test_idx + 1}/{n_runs} ---")
        test_run = all_runs_data[test_idx]
        
        val_idx = (test_idx + 1) % n_runs
        val_run = all_runs_data[val_idx]
        
        train_runs = [all_runs_data[i] for i in range(n_runs) if i != test_idx and i != val_idx]
        
        if len(train_runs) == 0:
            train_runs = [val_run] 
        
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        best_val_f1 = -1
        best_model_state = None
        patience_counter = 0
        early_stop_patience = 25
        
        for ep in range(epochs):
            train_loss = train_one_epoch(model, train_runs, optimizer, device)
            val_f1, val_jaccard, _ = validate_one_epoch(model, val_run, device)
            scheduler.step(val_f1)
            
            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {ep+1}")
                break
                
        # Evaluate on test
        if best_model_state:
            model.load_state_dict(best_model_state)
            
        test_f1, test_jaccard, test_probs = validate_one_epoch(model, test_run, device)
        
        print(f"Test Run {test_run['run_id']} F1: {test_f1:.4f}, Jaccard: {test_jaccard:.4f}")
        
        fold_results[test_run['run_id']] = {
            'f1': test_f1,
            'jaccard': test_jaccard,
            'probs': test_probs.tolist()
        }
        
    return fold_results

if __name__ == "__main__":
    pass
