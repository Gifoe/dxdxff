"""Post-hoc internal context only; never selection or final model mutation."""
import common as c


def rows(root,seal,arm,label):
    result=[]
    for fold in range(1,6):
        path=root/'models_private'/f'fold{fold}_screen1'/arm/'HELD_PRIVATE.pkl'
        expected=next(r['sha256'] for r in seal['cells'] if (r['fold'],r['screen'],r['arm'])==(fold,1,arm))
        assert c.sha(path)==expected;q=c.read(path)
        result.extend([dict(r,arm=label) for r in q['rows']])
    frame=c.pd.DataFrame(result);assert len(frame)==86 and frame.patient.nunique()==59
    return frame


def main():
    c.setup();current=c.jr(c.PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json')
    previous=c.jr(c.PREVIOUS/'public/INDEPENDENT_INTERNAL_VERIFICATION.json')
    assert current['status']==previous['status']=='PASS'
    choices=c.sha(c.ROOT/'OWN_FIT_SELECTION_PRIVATE.json')
    assert choices==current['ownFIT_selection_sha256']
    new_seal=c.jr(c.ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json');old_seal=c.jr(c.PREVIOUS/'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    values={arm:rows(c.ROOT,new_seal,arm,arm) for arm in c.TRAINED}
    values.update({f'PREVIOUS_{arm}':rows(c.PREVIOUS,old_seal,arm,f'PREVIOUS_{arm}') for arm in ['PRE_MEAN','PRE_LOGENERGY']})
    result=[]
    for left in c.TRAINED:
        for right in ['PREVIOUS_PRE_MEAN','PREVIOUS_PRE_LOGENERGY']:
            pair=values[left].merge(values[right],on=['patient','center','fold','screen'],suffixes=('_l','_r'),validate='one_to_one')
            assert len(pair)==86 and pair.patient.nunique()==59
            diff=c.pd.DataFrame(dict(patient=pair.patient,**{m:pair[m+'_l']-pair[m+'_r'] for m in c.METRICS}))
            result.extend([dict(left=left,right=right,**r) for r in c.parent.cluster_interval(diff.to_dict('records'))])
    c.pd.DataFrame(result).to_csv(c.PUB/'HISTORICAL_PRE_COMPARISON.csv',index=False)
    assert c.sha(c.ROOT/'OWN_FIT_SELECTION_PRIVATE.json')==choices
    c.write(c.PUB/'HISTORICAL_PRE_COMPARISON_AUDIT.json',dict(status='PASS',paired_metric_rows=len(result),appearances_per_comparison=86,
        unique_patients_per_comparison=59,current_internal_seal_sha256=c.sha(c.ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),
        previous_internal_seal_sha256=c.sha(c.PREVIOUS/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),
        choices_sha256_before_and_after=choices,posthoc_internal_only_not_selection=True,
        no_new_Development_Outer_data_read=True,no_training_prediction_or_threshold_change=True,
        scope='Allthree newarms versus both priorcompactPREoperators; role-matched internalcontext,not a new independenttest or parameter-matched attribution.'))
    print('JOINT_RAW_HISTORICAL_INTERNAL_CONTEXT_COMPLETE',len(result),flush=True)


if __name__=='__main__':main()
