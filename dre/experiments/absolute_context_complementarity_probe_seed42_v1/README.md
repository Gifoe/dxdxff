# Frozen A1 absolute-context complementarity probe

This development-only experiment asks whether patient-level embedding mean/scale, removed by A1's patient-relative normalization, adds information beyond a parameter-matched residual head on the relative representation alone. P0 is the frozen A1 model; P1 is the relative-only residual control; P2 has the same head with fit-normalized absolute patient context.

The protocol is locked in `PROTOCOL_LOCK.json` before probe training. The 150 original A1 checkpoints are reused; patient-level representations and probe checkpoints remain in the private server runtime. Only aggregate development results are published. No outer-test data is accessed.
