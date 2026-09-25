import re

with open("build_new1.py", "r", encoding="utf-8") as f:
    code = f.read()

forward_old = r'''        gru_out, _ = self.temporal_gru(seq_input)
        attn_weights = F.softmax(self.temp_attn(gru_out), dim=1)
        channel_embeddings = torch.sum(gru_out * attn_weights, dim=1)
        channel_embeddings = self.temp_proj(channel_embeddings)
        channel_logits = self.ez_head(channel_embeddings).squeeze(-1)

        if return_embeddings:
            return channel_logits, channel_embeddings
        return channel_logits'''

forward_new = r'''        gru_out, _ = self.temporal_gru(seq_input)
        attn_weights = F.softmax(self.temp_attn(gru_out), dim=1)
        channel_embeddings = torch.sum(gru_out * attn_weights, dim=1)
        channel_embeddings = self.temp_proj(channel_embeddings)
        channel_logits = self.ez_head(channel_embeddings).squeeze(-1)
        
        run_embedding = channel_embeddings.mean(dim=0)
        raw_count_run = self.count_reg_head(run_embedding)
        predicted_ratio_run = torch.sigmoid(self.count_ratio_head(run_embedding))

        if return_embeddings:
            return channel_logits, channel_embeddings, raw_count_run, predicted_ratio_run
        return channel_logits, raw_count_run, predicted_ratio_run'''

code = code.replace(forward_old, forward_new)

with open("build_new1.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Updated build_new1.py")
