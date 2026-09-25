# Isolated HUP Phys-NeuroEZ experiments

This folder is an overlay copy of the modified Phys-NeuroEZ code. It is kept
separate from the original baseline folder.

Default cache:

```text
/root/nips-E/outputs_dgneuroez_nez_v3_small/cache/techez_prepost_dynamic_cache_v5_9cb761e87145.pkl
```

Default output:

```text
/root/nips-E/hup_phys_compare_results
```

Run on the server:

```bash
python phys_neuroez_hup_overlay/run_hup_two_versions.py
```

Run with explicit paths:

```bash
python phys_neuroez_hup_overlay/run_hup_two_versions.py \
  --cache-path /root/nips-E/outputs_dgneuroez_nez_v3_small/cache/techez_prepost_dynamic_cache_v5_9cb761e87145.pkl \
  --output-root /root/nips-E/hup_phys_compare_results
```

It runs these two variants in order:

```text
NeuroEZ-B-PhysFeat
Phys-NeuroEZ-B
```

By default each variant runs only one seed:

```text
seed42
```

Important: `run_neuroez_v2.py --window_cache_path` uses the whole cache file.
So this is HUP-only only if the cache itself is HUP-only.
