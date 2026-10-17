# Single-head7D goal status,2026-10-17

The requested actual full80equal-patient Macro-F1>0.64 remains **unproven**.
Retained C2 is0.6396579900303192,not a rounded success. No newOuter was
authorized by either completed study in this goal turn.

| Current-turn candidate | Internal patientMacro | AUROC | AP | Gate |
|---|---:|---:|---:|---|
| Retained C2 | .673076 | .762169 | .561626 | Reference |
| Pointwise matched-capacity block | .672011 | .761336 | .561546 | Failed |
| Query-dependent channel relation | .674225 | .765829 | .562818 | Failed |
| C2 normalized global Macro objective | .669890 | .758296 | .559045 | Failed |
| C2 patient-local balanced Macro objective | .673135 | .759028 | .555161 | Failed |

These internal figures cover65appearances/47people,not80Outer patients.
SameDevelopment selects epoch/theta/configuration,so these are
selection-conditioned estimates. Neither study's predeclared+.003Macro/
AUROC-AP safety gate was relaxed. Four candidate point estimates and their
nominalCI upper bounds are not substituted for a selectedfull80result.

Actual new work this turn:20fresh end-to-end fits/215epochs. Each study
independently replayed15logicalcheckpoint/control cells,195patientrows/
1755metric entries and54intervals. The same5oldC2fits recur across studies;
they are not10independent new baselines. Prior Raw/FiLM/context negative
studies remain preserved and are not counted as new work here.

Channel-relational versus pointwise Macro is+.002214,
CI[-.004678,+.010461];patient-local versus normalized-global is+.003244,
CI[-.002897,+.009401]. Neither establishes a reliable mechanism gain or
improvement over retained C2. Higher EZ-F1 can be canceled by lowerNEZ-F1;
better ranking alone does not satisfy the F1 objective.

The channel relation is an architecture test. The subsequent objective
comparison deliberately retains the designed C2 architecture and is a
training-method test,not structural novelty. All7measurements,Rawinput,
patientroles and singleclassificationhead remain unchanged in both.

The training comparison's first bootstrap verifier had a confirmed native
pandas access violation. Allmodels/checkpoints were alreadycomplete and
hash-unchanged. Fresh-process verification-only recovery succeeded with
no retraining or scientific/source/tolerance change;the original logs remain.

The goal remains active,not complete or blocked by these negative findings.
Any next experiment needs a distinct prospective mechanism and legal
internal selection rather than changing these gates or fishing onOuter.
Only actual code/hash/anonymousresults are public;private inputs and scores
are preserved on the trusted server. Any futureOuter remains exploratory
repeated-test,not independentconfirmation.
