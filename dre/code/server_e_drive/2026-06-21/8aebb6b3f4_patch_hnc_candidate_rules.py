from pathlib import Path

src = Path("neuroez_c/v3_hnc_features.py")
text = src.read_text(encoding="utf-8")

old = '''        n_channels = len(group)
        top20 = max(1, int(math.ceil(0.20 * n_channels)))
        if candidate_rule == "top12":
            candidate_names = set(ordered.head(min(12, n_channels))["channel_name_norm_hnc"])
        elif candidate_rule == "top8":
            candidate_names = set(ordered.head(min(8, n_channels))["channel_name_norm_hnc"])
        else:
            candidate_names = set(ordered.head(top20)["channel_name_norm_hnc"])
        subject_id = str(group["subject_id"].iloc[0])
        topology_map = patient_topology.get(subject_id, {})
        if candidate_rule == "top20pct_plus_neighbors":
            for _, top_row in ordered.head(min(5, n_channels)).iterrows():
'''

new = '''        n_channels = len(group)
        top20 = max(1, int(math.ceil(0.20 * n_channels)))

        rule = str(candidate_rule)
        add_neighbors = rule.endswith("_plus_neighbors")
        base_rule = rule[: -len("_plus_neighbors")] if add_neighbors else rule

        if base_rule == "all_channels":
            candidate_count = n_channels
        elif base_rule.startswith("top") and base_rule.endswith("pct"):
            pct_text = base_rule[len("top") : -len("pct")]
            try:
                pct = float(pct_text) / 100.0
            except ValueError:
                pct = 0.20
            pct = min(max(pct, 0.0), 1.0)
            candidate_count = max(1, int(math.ceil(pct * n_channels)))
        elif base_rule.startswith("top"):
            try:
                candidate_count = int(base_rule[len("top") :])
            except ValueError:
                candidate_count = max(1, int(math.ceil(0.20 * n_channels)))
        else:
            candidate_count = max(1, int(math.ceil(0.20 * n_channels)))

        candidate_count = max(1, min(candidate_count, n_channels))
        candidate_names = set(ordered.head(candidate_count)["channel_name_norm_hnc"])

        subject_id = str(group["subject_id"].iloc[0])
        topology_map = patient_topology.get(subject_id, {})

        if add_neighbors and base_rule != "all_channels":
            neighbor_anchor_count = min(max(5, int(math.ceil(0.05 * n_channels))), candidate_count, n_channels)
            for _, top_row in ordered.head(neighbor_anchor_count).iterrows():
'''

if old in text:
    backup = src.with_suffix(".py.before_candidate_grid_patch")
    backup.write_text(text, encoding="utf-8")
    src.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched: {src}")
    print(f"backup: {backup}")
elif 'base_rule == "all_channels"' in text and 'top30pct_plus_neighbors' not in text:
    print("candidate-rule patch appears already installed.")
elif 'base_rule == "all_channels"' in text:
    print("candidate-rule patch appears already installed.")
else:
    raise SystemExit("Patch failed: target block not found. Inspect neuroez_c/v3_hnc_features.py manually.")
