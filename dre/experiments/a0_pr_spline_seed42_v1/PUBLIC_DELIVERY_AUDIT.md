# Public delivery audit

Delivery is restricted to this new experiment directory. Allowed files are
source/test `.py`, protocol/hash/audit `.json`, anonymous aggregate `.csv`,
report `.md`, and local `.gitattributes`/`.gitignore`. No existing unrelated
file or experimental result is changed.

The CSV units are arm, fold, center, training epoch, named feature/fold, or
aggregate contrast. No CSV contains a patient ID, individual patient metric,
channel identity, record name, individual score, clinical waveform or label
array. Feature names, aggregate FIT channel counts, model statistics, source
digests and selected checkpoint hashes are allowed; they are not clinical
record identity or checkpoint contents. Small center aggregates are reported
as aggregates, with no individual patient linkage.

Private banks, input arrays, raw EEG, all frozen model/optimizer checkpoints,
patient/channel predictions, per-patient metric tables, resume artifacts,
logs and credentials remain on the trusted server. Bytecode is ignored, as
are private/binary/log extensions. A scoped filename/extension check, source
credential/identity-pattern scan and staged diff review precede publication.
No credential or private data is needed by the GitHub publication mechanism.

Executed code/tests and protocol are verified against the pretraining binding
in `results/PRETRAIN_AUDIT.json` before staging. UTF-8 LF normalization keeps
the published executable source/protocol equal to the bound source bytes.
Transferred result hashes are server-artifact provenance; they are not claims
that Git-normalized aggregate CSV bytes equal every runtime file byte.

Only already-computed aggregates were fetched. No new classification,
threshold fitting, model training or outer inference is run for packaging.
The report's descriptive summaries of knots/slopes/timing derive directly
from fetched aggregate rows, not from additional private outcome access.
Public git-tree parity is checked after publication against the reviewed
local commit. No pull request or additional experimental branch is requested.
