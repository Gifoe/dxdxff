# Exact Raw-only model and interpretation boundary

Each fresh candidate is one Raw encoder and one classifier, not a D1 residual
or late ensemble. Four fixed configurations change only waveform normalization
and pooling of learned responses. The mean and energy operators have no fitted
parameters; their **convolutional inputs and classification head are learned**.
Calling this a learnable pool-parameter module would be inaccurate.

| Stage | Exact operation | What is not supplied |
|---|---|---|
| Input | All original valid 500-sample Raw250 windows, valid mask, fixed peri-onset times | No feature-bank measurements, old logits or embeddings |
| Normalization | Original centered window SD/clip, or fixed pre-only common-scale asinh/clip | No appended RMS/variance/bandpower coordinate |
| Encoder | Conv1→12 k9/s4, GELU, Conv12→24 k7/s4, GELU | No preset frequency-band filter definitions |
| Window pool | Mean, or log1p(mean of squared learned responses) | Not measured raw bandpower or a clinical frequency-band claim |
| Patient representation | Pre-latent subtraction, two phase means, available-seizure mean/SD, complete-patient Z | No patient/source ID as predictor |
| Classifier | 160→64 GELU→1; only 96 learned coordinates plus 64 literal zeros | No independent physiological branch or D1 score |

There are 12,529 total parameters in every fresh model: 120 in the first
convolution, 2,040 in the second, 10,304 in the first head layer and 65 in its
output layer. The 64 literal-zero slots preserve the historical head shape and
initialization semantics. Their 4,096 incident weights are counted in the total
but cannot carry physiological information; no measured values fill those
slots. This is not a claim that all counted parameters are computationally
necessary. The four candidates share exact random tensors before fitting and
do not accept a trained-state constructor argument.

The squared-response operation is a fixed neural nonlinearity/statistic on
jointly learned filters. It must not be renamed a recovered delta/theta/alpha
bandpower, an interpretable high-gamma measure, or an automatically discovered
clinical biomarker. The unchanged Raw250 input and its acquisition/preprocessing
limitations remain; this experiment makes no low-frequency provenance repair
or full 80–150 Hz measurement claim.

Labels enter only the supervised TRAIN+CHOICE loss and CAL model/threshold
selection, never normalization or the prediction function. Patient grouping
defines complete-channel normalization; source identity is used only for
reported stability/selection penalties, not supplied to the classifier. The
frozen historical scores are reference outcomes only, not residual features.

The 2x2 design is necessary for attribution: improvement over historical Native
alone could be due to different random initialization, longer Macro-aligned
training or more legal inner training patients. Energy versus mean **within the
same normalization** uses matched fresh initialization/data/budget. Its
interaction with normalization is a paired performance contrast, not proof of
a specific physiological mechanism. No external novelty claim is made for
response-energy pooling.

The final primary procedure is the per-ownFIT selected policy. Fixed-arm Outer
tables remain prespecified secondary evaluations, not a source of post-test
architecture or threshold selection. The target is full 80-patient equal-weight
Macro-F1 >0.65, not a first-path internal point, a favorable fold or an oracle.
