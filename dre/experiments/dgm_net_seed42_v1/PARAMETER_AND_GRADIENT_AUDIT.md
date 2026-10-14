# Parameter and gradient admission

Before formal training, check exact budgets, absence of recurrent/attention modules, synthetic gate identity and actual legal-FIT initialization identity. Auxiliary BCE at weight.25 permits CNN/head and gate output gradients despite alpha0. Earlier gate weights are expected zero-gradient on the first update because the last projection is zero, and must receive nonzero gradient after it moves.

Actual two-update smoke uses a legal original FIT patient only, FIT class weights and the prescribed optimizer/clipping. The smoke model/optimizer is discarded; fold construction/training seeds are reset, preserving all selected D1 states. D1 gradients must remain absent, eval state frozen and tensors unchanged. All active gradients/outputs finite; trained gate identity removal must change raw-head logits. Numerical results and norms are written only after execution to `PARAMETER_AND_GRADIENT_AUDIT.json`.
