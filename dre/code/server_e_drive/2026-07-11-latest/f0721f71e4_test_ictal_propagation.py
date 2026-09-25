import numpy as np

from neuroez_c.task2.ngbr.ictal_propagation import propagation_scores_from_times,recruitment_time_from_score


def test_early_recruitment_scores_higher_and_unrecruited_zero():
    score=propagation_scores_from_times(np.array([.2,1.,3.,np.nan]),np.ones(4,bool))
    assert score[0]>score[1]>score[2] and score[3]==0


def test_recruitment_persistence_and_target_outside_delay():
    times=np.arange(-1,2,.05);values=np.zeros((2,len(times)));values[0,(times>=.2)&(times<.6)]=4;values[1,(times>=1)&(times<1.5)]=4
    detected=recruitment_time_from_score(values,times)
    assert abs((detected[1]-detected[0])-.8)<.06
