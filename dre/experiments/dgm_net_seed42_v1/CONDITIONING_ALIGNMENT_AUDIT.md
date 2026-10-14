# Alignment and inference controls

Raw and physiology originate from the exact same hashed500-sample window and canonical patient/seizure/channel map. Labels do not form pre-reference, scales or shuffles. Original missing masks and actual times remain unchanged.

Diagnostic shuffle uses a hash of seed42 + patient + compound seizure + canonical channel + stage; only genuine windows inside each[0,8)/[8,20) stage participate. All7 coordinates move together, multiset preserved, nonidentity enforced when at least2 eligible; singleton counts disclosed. Pre-reference is unchanged. Static intervention averages unnormalized DELTA within stage then applies the same locked scale/clip. Neutral-family interventions set each normalized coordinate0 separately.

Selected A2 checkpoints/scales/thresholds are frozen before interventions; no retraining or threshold refitting. A2 dependence under a wrong-window intervention is not equivalent to superiority over a separately trained A3.
