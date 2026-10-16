"""Post hoc, public-only name-definition sensitivity; never merges model inputs."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Groups share measurement concepts, not numerical or predictive equivalence.
ATTRIBUTE = {
    'log_bp_beta':'beta_power', 'stable_log_beta_sameband':'beta_power',
    'log_bp_low_gamma':'low_gamma_power', 'stable_log_low_gamma_sameband':'low_gamma_power',
    'partial_high_gamma_80_125':'partial_hg_power', 'stable_log_partial_high_gamma_sameband':'partial_hg_power',
    'rms_legacy':'rms', 'variance':'variance', 'log_variance_stable':'variance',
    'line_length_per_sec':'line_length', 'normalized_line_length':'normalized_line_length',
    'spectral_entropy_10_125':'spectral_entropy', 'hjorth_mobility_per_second':'hjorth_mobility',
    'hjorth_mobility_legacy':'hjorth_mobility', 'hjorth_complexity_stable':'hjorth_complexity',
    'hjorth_complexity_legacy':'hjorth_complexity', 'permutation_entropy_m3_tau1':'ordinal_entropy',
    'spectral_centroid_13_95':'spectral_centroid', 'log_gamma_beta_ratio':'gamma_beta_ratio',
    'partial_high_gamma_fraction_80_125':'partial_hg_fraction', 'spectral_flatness_13_95':'spectral_flatness',
    'spectral_slope_proxy_20_95_notchgap':'spectral_slope', 'clipped_excess_kurtosis':'kurtosis',
    'partial_alpha_10_13_legacy':'partial_alpha_power', 'log_total_power_10_125_legacy':'retained_total_power',
    'rebuilt_degree_norm':'graph_degree', 'rebuilt_strength_norm':'graph_strength',
    'rebuilt_clustering_coeff':'graph_clustering', 'rebuilt_eigenvector_centrality':'graph_eigenvector',
    'rebuilt_pagerank':'graph_pagerank', 'rebuilt_kcore_norm':'graph_kcore', 'rebuilt_local_efficiency':'graph_efficiency',
}
FAMILIES = {
    'retained_band_power':['beta_power','low_gamma_power','partial_hg_power','partial_alpha_power'],
    'amplitude':['rms','variance','retained_total_power'],
    'temporal_roughness':['line_length','normalized_line_length','hjorth_mobility'],
    'spectral_distribution':['spectral_entropy','spectral_centroid','gamma_beta_ratio','partial_hg_fraction','spectral_flatness','spectral_slope'],
    'temporal_complexity':['hjorth_complexity','ordinal_entropy'],
    'tail_shape':['kurtosis'],
    'network':['graph_degree','graph_strength','graph_clustering','graph_eigenvector','graph_pagerank','graph_kcore','graph_efficiency'],
}


def main():
    pool = HERE.parent / 'm8_physiology_discovery_selection_seed42_v2'
    catalogue = pd.read_csv(pool / 'FEATURE_CATALOGUE.csv')
    assert set(catalogue.feature) == set(ATTRIBUTE) and len(ATTRIBUTE) == 32
    family = {attribute:label for label, names in FAMILIES.items() for attribute in names}
    assert set(family) == set(ATTRIBUTE.values())
    out = HERE / 'results'; out.mkdir(exist_ok=True)
    pd.DataFrame([dict(feature=n,attribute=ATTRIBUTE[n],coarse_family=family[ATTRIBUTE[n]],
                       numerical_equivalence_claim=False) for n in catalogue.feature]).to_csv(out/'ATTRIBUTE_MAP.csv',index=False)
    studies = {
        'physiology_only':pd.read_csv(pool/'results/FROZEN_FEATURE_SELECTION.csv').rename(columns={'outer_fold':'fold','arm':'recipe'}),
        'Raw_conditional':pd.read_csv(HERE.parent/'m8_rawconditional_feature_selection_seed42_v1/results/dense_reference/FROZEN_INTERNAL_SELECTION.csv'),
    }
    pairs=[];summary=[]
    for study, table in studies.items():
        choices=table[table.recipe=='SELECTED'].sort_values('fold')
        names=[set(s.split(';')) if pd.notna(s) else set() for s in choices.features]
        for level,transform in [('exact',lambda n:n),('attribute',lambda n:ATTRIBUTE[n]),('coarse_family',lambda n:family[ATTRIBUTE[n]])]:
            sets=[{transform(n) for n in ns} for ns in names]
            values=[]
            for (i,a),(j,b) in itertools.combinations(enumerate(sets),2):
                value=len(a&b)/len(a|b) if a|b else 1.
                values.append(value);pairs.append(dict(study=study,level=level,fold_a=i+1,fold_b=j+1,jaccard=value,
                                                      posthoc_descriptive=True,overlapping_training=True))
            common=set.intersection(*sets)
            summary.append(dict(study=study,level=level,mean_pairwise_jaccard=float(np.mean(values)),
                                all5_fold_shared_count=len(common),shared=';'.join(sorted(common)),
                                posthoc_descriptive=True,inputs_not_changed=True))
    pd.DataFrame(pairs).to_csv(out/'ATTRIBUTE_OVERLAP_PAIRS.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'ATTRIBUTE_OVERLAP_SUMMARY.csv',index=False)
    (out/'ATTRIBUTE_STATUS.json').write_text(json.dumps(dict(status='COMPLETE_PUBLIC_DEFINITION_SENSITIVITY',
        all32_definitions_mapped=True,no_new_labels_or_predictions=True,posthoc_not_feature_selection=True,
        coarsening_does_not_prove_substitutability=True),indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
