"""Single views derived strictly inside the new36-coordinate B2 main model."""
from folding import mapping,retained


def configurations(columns):
    mp=mapping(columns)
    removed=[q['group'] for q in mp['groups'] if q['view']=='delta' and q['feature'] in ['log_bp_beta','log_bp_low_gamma','log_bp_high_gamma']]
    assert len(removed)==3
    main=retained(mp,removed)
    configs={'MAIN_B2':main}
    for arm,view in [('ABS_ONLY','abs'),('DELTA_ONLY','delta'),('RATIO_ONLY','ratio')]:
        configs[arm]=[q for q in main if q['view']==view]
    assert {k:len(v) for k,v in configs.items()}==dict(MAIN_B2=36,ABS_ONLY=14,DELTA_ONLY=8,RATIO_ONLY=14)
    assert {q['physiological_feature'] for q in configs['DELTA_ONLY']}=={'rms','variance','line_length_per_sec','spectral_entropy'}
    ids=[{q['index'] for q in configs[k]} for k in ['ABS_ONLY','DELTA_ONLY','RATIO_ONLY']]
    assert all(ids[i].isdisjoint(ids[j]) for i in range(3) for j in range(i))
    assert set.union(*ids)=={q['index'] for q in main}
    return configs
