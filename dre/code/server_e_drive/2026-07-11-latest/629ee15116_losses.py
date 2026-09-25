import torch
def patient_bce(outputs,labels):
    logits=torch.stack([x['logit'] for x in outputs]).float();targets=torch.as_tensor(labels,dtype=torch.float32,device=logits.device);return torch.nn.functional.binary_cross_entropy_with_logits(logits,targets)
