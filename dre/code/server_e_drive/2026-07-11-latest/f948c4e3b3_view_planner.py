from __future__ import annotations
import hashlib,torch
from dataclasses import dataclass

def stable_hash(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big')
def cyclic(values,limit,start):
    values=list(values)
    if len(values)<=limit:return values
    return [values[(start+i)%len(values)] for i in range(limit)]
@dataclass(frozen=True)
class Capacity:
    max_seizures:int=3;max_channels:int=96;max_windows_per_phase:int=6
class DeterministicEvalViewPlanner:
    def __init__(self,base_seed=42,capacity=Capacity()):self.base_seed,self.capacity=int(base_seed),capacity
    def view(self,patient,outer_fold,view_index):
        seizures=list(patient['seizures']);selected=cyclic(seizures,self.capacity.max_seizures,view_index*self.capacity.max_seizures%len(seizures));output=[]
        for seizure in selected:
            source_mask=seizure['window_mask'].bool();valid=torch.where(source_mask.any(1))[0].tolist();base_offset=stable_hash(self.base_seed,outer_fold,seizure['seizure_id'])%max(1,len(valid));offset=(base_offset+view_index*self.capacity.max_channels)%max(1,len(valid));channels=cyclic(valid,self.capacity.max_channels,offset);phase=seizure['phase_ids'];times=seizure['relative_times_sec'];phase2=phase[None].expand_as(source_mask) if phase.ndim==1 else phase;time2=times[None].expand_as(source_mask) if times.ndim==1 else times;chosen_mask=torch.zeros_like(source_mask)
            for channel in channels:
                for phase_id in range(3):
                    candidates=torch.where(source_mask[channel]&(phase2[channel]==phase_id))[0].tolist();candidates=sorted(candidates,key=lambda i:float(time2[channel,i]));window_base=stable_hash(self.base_seed,outer_fold,seizure['seizure_id'],channel,phase_id)%max(1,len(candidates));window_start=(window_base+view_index*self.capacity.max_windows_per_phase)%max(1,len(candidates));chosen=cyclic(candidates,self.capacity.max_windows_per_phase,window_start);chosen_mask[channel,chosen]=True
            keep_time=chosen_mask[channels].any(0);c=torch.tensor(channels);t=torch.where(keep_time)[0]
            output.append({'seizure_id':seizure['seizure_id'],'windows':seizure['windows'][c][:,t].float(),'window_mask':chosen_mask[c][:,t],'phase_ids':phase2[c][:,t].long(),'relative_times_sec':time2[c][:,t].float(),'channel_names':[seizure['channel_names'][i] for i in channels]})
        return {'patient_key':patient['patient_key'],'center':patient['center'],'view_index':view_index,'seizures':output}
    def views(self,patient,outer_fold,n_views=8):return [self.view(patient,outer_fold,index) for index in range(n_views)]
    def coverage(self,patient,outer_fold,n_views=8):
        views=self.views(patient,outer_fold,n_views);total_s={s['seizure_id'] for s in patient['seizures']};covered_s={s['seizure_id'] for v in views for s in v['seizures']};total_c={(s['seizure_id'],name) for s in patient['seizures'] for name in s['channel_names']};covered_c={(s['seizure_id'],name) for v in views for s in v['seizures'] for name in s['channel_names']};total_w=set()
        for seizure in patient['seizures']:
            times=seizure['relative_times_sec'];times=times[None].expand_as(seizure['window_mask']) if times.ndim==1 else times
            total_w|={(seizure['seizure_id'],seizure['channel_names'][c],float(times[c,t])) for c in range(len(seizure['channel_names'])) for t in torch.where(seizure['window_mask'][c])[0].tolist()}
        covered_w={(s['seizure_id'],s['channel_names'][c],float(s['relative_times_sec'][c,t])) for v in views for s in v['seizures'] for c in range(len(s['channel_names'])) for t in torch.where(s['window_mask'][c])[0].tolist()}
        return {'n_total_seizures':len(total_s),'n_covered_seizures':len(covered_s),'seizure_coverage_fraction':len(covered_s)/max(1,len(total_s)),'mean_total_channels':len(total_c)/max(1,len(total_s)),'mean_covered_channels':len(covered_c)/max(1,len(total_s)),'channel_coverage_fraction':len(covered_c)/max(1,len(total_c)),'mean_total_windows':len(total_w)/max(1,len(total_c)),'mean_covered_windows':len(covered_w)/max(1,len(total_c)),'window_coverage_fraction':len(covered_w)/max(1,len(total_w))}
def view_identity(view):
    seizures={s['seizure_id'] for s in view['seizures']};channels={(s['seizure_id'],n) for s in view['seizures'] for n in s['channel_names']};windows={(s['seizure_id'],s['channel_names'][c],float(s['relative_times_sec'][c,t])) for s in view['seizures'] for c in range(len(s['channel_names'])) for t in torch.where(s['window_mask'][c])[0].tolist()};return seizures,channels,windows
def jaccard(a,b):return len(a&b)/max(1,len(a|b))
