# Frozen Outer TP/FP case with positive AUROC gain

The user requested another case extending CASE-F01 with positive mean AUROC.
Preserve the same TP/FP requirements and fixed thresholds; exclude the four
previous cases. Use the previously verified independent Python formulas from
the start. The protocol is retrospective and retains all per-seed endpoints.

Run execute.py in the trusted server; only CASE-A01 / CH aliases are public.

Completed and independently verified on 2026-10-30. See
[seed42 Chinese report](results/seed42/README_ZH.md),
[seed42 confusion counts](results/seed42/CASE_CONFUSION.csv), and
[seed42 anonymous channel data](results/seed42/DRECHANNET_SEED42_CHANNELS_WIDE.csv).

CASE-A01: HUP, Outer fold4, 49 channels / 10 EZ. Mean DRE TP 2->3.6667,
FP 0->0.3333, AUROC 0.752137->0.769231. Two seeds improve AUROC; seed44
declines. Mean precision declines from 1 to 0.944444. All results are retained.
The user requested seed42 after viewing the three-seed results. The GitHub
publication is restricted to the DREChanNet seed42 view. The complete original
three-seed results remain on the trusted server and locally. This selected case
and selected seed are retrospective illustrations, not independent confirmation.

Verify the published subset with:

```powershell
python export_seed42.py --verify results/seed42
```
