from neuroez_c.task2.tech_outcome_c1.sampling import patient_batches
from neuroez_c.task2.tech_outcome_c1.split import development_split,assert_isolated
def test_sampler_once_without_replacement_and_split_isolation():
 keys=[str(i) for i in range(20)];meta={k:{'center':'a','outcome_success':i%2} for i,k in enumerate(keys)};train,val=development_split(keys,meta,1,42);assert_isolated(train,val,['z']);b=patient_batches(train,{k:meta[k]['outcome_success'] for k in train},4,42);flat=sum(b,[]);assert len(flat)==len(set(flat))==len(train)
