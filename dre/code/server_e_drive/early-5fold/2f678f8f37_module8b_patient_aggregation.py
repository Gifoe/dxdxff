import torch
import torch.nn as nn
import torch.nn.functional as F

class PatientOutcomeAggregator(nn.Module):
    def __init__(self, run_emb_dim, hidden_dim=64, num_phase_types=3, dropout_rate=0.3):
        super().__init__()
        
        self.phase_embedding = nn.Embedding(num_phase_types, 16)
        
        in_dim = run_emb_dim + 16
        
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim/2), 1)
        )
        
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, run_embs, phase_ids):
        """
        run_embs: (R, run_emb_dim)
        phase_ids: (R,)
        """
        R = run_embs.size(0)
        
        p_emb = self.phase_embedding(phase_ids) # (R, 16)
        
        x = torch.cat([run_embs, p_emb], dim=-1) # (R, run_emb_dim+16)
        h = self.proj(x) # (R, hidden_dim)
        
        if R > 1:
            attn_scores = self.attn_mlp(h)   # (R, 1)
            attn_weights = F.softmax(attn_scores, dim=0) # (R, 1)
            patient_emb = (h * attn_weights).sum(dim=0)  # (hidden_dim,)
        else:
            attn_weights = None
            patient_emb = h[0]
            
        outcome_logit = self.outcome_head(patient_emb)
        
        # Outcome logit should return (1,)
        return {
            "patient_emb": patient_emb,
            "outcome_logit": outcome_logit,
            "attn_weights": attn_weights
        }