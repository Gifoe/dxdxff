import numpy as np

from neuroez_c.task2.ngbr.hfo_detection import compute_hfo,hfo_eligibility
from neuroez_c.task2.ngbr.schema import RawRunRecord


def _record(fs):
    onset=int(31*fs);n=onset+int(fs);rng=np.random.default_rng(7);signal=.1*rng.normal(size=(4,n));time=np.arange(int(.1*fs))/fs;start=onset-int(2*fs);signal[0,start:start+len(time)]+=4*np.sin(2*np.pi*120*time)
    return RawRunRecord("p","c","s",signal,float(fs),onset,[f"A{i}" for i in range(4)],np.ones(4,bool),np.array([1,0,0,0],bool),0,31)


def test_hfo_sampling_rate_gates():
    assert not hfo_eligibility(_record(500))[0]
    assert hfo_eligibility(_record(1000))[0] and not hfo_eligibility(_record(1000))[2]
    assert hfo_eligibility(_record(2000))[2]


def test_synthetic_ripple_is_detected():
    result=compute_hfo(_record(1000))
    assert result.quality["hfo_valid"] and result.channel_table.loc[0,"ripple_rate"]>0


def test_global_broadband_like_coincidence_is_rejected():
    record=_record(1000);start=record.onset_sample-int(3*record.sampling_rate);time=np.arange(100)/record.sampling_rate
    record.signal[:,start:start+len(time)]+=8*np.sin(2*np.pi*120*time)
    result=compute_hfo(record)
    # The simultaneous event must not survive on a majority of channels.
    assert int((result.channel_table.ripple_rate>0).sum())<=2
