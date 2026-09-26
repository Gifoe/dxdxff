# Frozen A1 absolute-context complementarity probe

This development-only experiment asks whether patient-level embedding mean/scale, removed by A1's patient-relative normalization, adds information beyond a parameter-matched residual head on the relative representation alone. P0 is the frozen A1 model; P1 is the relative-only residual control; P2 has the same head with fit-normalized absolute patient context.

The protocol was locked in `PROTOCOL_LOCK.json` before probe training. The 150 original A1 checkpoints were reused; patient-level representations and probe checkpoints remain in the private server runtime. Only aggregate development results are published. No outer-test loader, predictions, or performance evaluation were used.

All 300 matched probe cells completed. P0's VLOO Macro-F1 reproduced exactly (0.625996); P1 reached 0.618627 and P2 reached 0.617182. P2 minus P1 was -0.001445, so the absolute-context complementarity gate failed. See `FINAL_REPORT.md` for fold-level results and diagnostics. Terminal: `ABSOLUTE_CONTEXT_COMPLEMENTARITY_NOT_SUPPORTED`.
