from __future__ import annotations
from neuroez_c.task2.tech_outcome_c1.data import CachedCohort,TeChC1PatientViewBuilder
from neuroez_c.task2.tech_outcome_c1.schema import ViewConfig
from .view_planner import Capacity,DeterministicEvalViewPlanner,stable_hash,view_identity,jaccard
class AlignedCohort(CachedCohort):
    def __init__(self,cache_dir,seed=42,capacity=Capacity()):super().__init__(cache_dir,seed,ViewConfig(capacity.max_seizures,capacity.max_channels,capacity.max_windows_per_phase));self.seed=seed;self.capacity=capacity;self.planner=DeterministicEvalViewPlanner(seed,capacity)
    def train_views(self,key,outer_fold,epoch,n_views=2):
        patient=self.load(key);return [TeChC1PatientViewBuilder(stable_hash(self.seed,outer_fold,epoch,key,index),ViewConfig(self.capacity.max_seizures,self.capacity.max_channels,self.capacity.max_windows_per_phase)).build(patient,epoch=0,training=True) for index in range(n_views)]
    def eval_views(self,key,outer_fold,n_views=8):return self.planner.views(self.load(key),outer_fold,n_views)
    def coverage(self,key,outer_fold,n_views=8):return {'patient_key':key,'outer_fold':outer_fold,**self.planner.coverage(self.load(key),outer_fold,n_views)}
    def diversity(self,key,outer_fold,epoch=1):
        patient=self.load(key);a,b=self.train_views(key,outer_fold,epoch,2);ia,ib=view_identity(a),view_identity(b);over_capacity=len(patient['seizures'])>self.capacity.max_seizures or any(len(s['channel_names'])>self.capacity.max_channels or any(int((s['window_mask'][c] & ((s['phase_ids'][None].expand_as(s['window_mask']) if s['phase_ids'].ndim==1 else s['phase_ids'])[c]==p)).sum())>self.capacity.max_windows_per_phase for c in range(len(s['channel_names'])) for p in range(3)) for s in patient['seizures']);different=ia!=ib;return {'patient_key':key,'outer_fold':outer_fold,'epoch':epoch,'data_exceed_capacity':over_capacity,'at_least_one_level_differs':different,'seizure_jaccard':jaccard(ia[0],ib[0]),'channel_jaccard':jaccard(ia[1],ib[1]),'window_jaccard':jaccard(ia[2],ib[2])}
