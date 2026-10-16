"""Prepare20 frozen selector views with HELD labels physically omitted."""
from study_io import *


def main():
    ROOT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    protocol=jr(HERE/'PROTOCOL_LOCK.json');qstatus=jr(QUAL/'public/QUALIFICATION_STATUS.json')
    assert sha(QUAL/'MEASUREMENTS_PRIVATE.pkl')==protocol['qualified_private_sha256']==qstatus['private_measurements_sha256']
    assert sha(QUAL/'INPUT_FREEZE_PRIVATE.json')==protocol['qualification_input_sha256']
    assert jr(QUAL/'public/INDEPENDENT_QUALIFICATION_VERIFICATION.json')['status']=='PASS'
    qbind=jr(QUAL/'INPUT_FREEZE_PRIVATE.json')
    for section in ['code','numeric_amendment']:
        for name,digest in qbind[section].items():assert sha(QUALCODE/name)==digest
    assert sha(RAWROOT/'ENCODER_REGISTRY_PRIVATE.pkl')==protocol['raw_encoder_registry_sha256']
    registry=read(RAWROOT/'ENCODER_REGISTRY_PRIVATE.pkl');measurements=read(QUAL/'MEASUREMENTS_PRIVATE.pkl')
    rec=read(BASE/'RECONSTRUCTION_PRIVATE.pkl');pool=read(OLDROOT/'POOL_PRIVATE.pkl')
    assert sha(BASE/'RECONSTRUCTION_PRIVATE.pkl')==qbind['reconstruction'] and np.array_equal(pool['summary'][:,:14],rec['R14'])
    bound=dict(source=source_binding(),qualification=protocol['qualified_private_sha256'],qualification_input=protocol['qualification_input_sha256'],
        Raw_registry=protocol['raw_encoder_registry_sha256'],old_pool=sha(OLDROOT/'POOL_PRIVATE.pkl'),
        reconstruction=sha(BASE/'RECONSTRUCTION_PRIVATE.pkl'),parent_Outer_seal=sha(OLDROOT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'))
    freeze(ROOT/'INPUT_FREEZE_PRIVATE.json',bound);entries=[];audits=[]
    for fold in range(1,6):
        identity=fit_identity_bank(fold,rec);ix=identity['archive_indices'];pid=identity['patient']
        old=patient_z(pool['summary'][ix,:14].astype(float),pid);new=patient_z(measurements['values'][ix],pid)
        for screen in range(4):
            ent=next(e for e in registry['entries'] if e['fold']==fold and e['screen']==screen)
            assert sha(Path(ent['path']))==ent['sha256'];raw=read(Path(ent['path']))
            assert np.array_equal(raw['archive_indices'],ix) and raw['original_validation_and_Outer_excluded']
            role=raw['roles'];record=next(r for r in measurements['roles'] if r['fold']==fold and r['screen']==screen)
            for name,indices in role.items():assert np.array_equal(ix[indices],record['indices'][name])
            sets=[set(pid[indices]) for indices in role.values()]
            assert all(sets[a].isdisjoint(sets[b]) for a in range(len(sets)) for b in range(a))
            tr=role['train'];source_finite,baseline=source_rates(measurements,ix,tr)
            values=measurements['values'][ix];nonconstant=nonconstant_rate(values[tr],pid[tr]);allowed=[]
            for j,name in enumerate(GROUPS):
                coverage=np.isfinite(values[tr,2*j:2*j+2]).mean(0)
                good=source_finite[j]>=.99 and baseline[j]>=.99 and coverage[1]>=.99 and (j in [2,3] or coverage[0]>=.99) and max(nonconstant[2*j:2*j+2])>=.5
                duplicate=None
                if good:
                    duplicate=next((g for g in allowed if np.array_equal(new[tr,2*j:2*j+2],new[tr,2*g:2*g+2],equal_nan=True)),None)
                good=good and duplicate is None
                if good:allowed.append(j)
                audits.append(dict(fold=fold,screen=screen,feature=name,eligible=bool(good),selection_train_patients=len(set(pid[tr])),
                    source_finite_rate=source_finite[j],baseline_eligible_rate=baseline[j],coordinate0_coverage=coverage[0],coordinate1_coverage=coverage[1],
                    nonconstant_patient_fraction=float(max(nonconstant[2*j:2*j+2])),duplicate_of=duplicate,
                    reason='ELIGIBLE' if good else 'EXACT_TRAIN_DUPLICATE' if duplicate is not None else 'OWN_TRAIN_QUALITY_OR_VARIATION',
                    original_validation_and_Outer_excluded=True))
            # Only these three label subsets are serialized; no HELD label member.
            train_labels={name:identity['own_fit_ez'][role[name]] for name in ['train','cal','choice']}
            view=dict(binding=bound,fold=fold,screen=screen,archive_indices=ix,patient=pid,channel=identity['channel'],center=identity['center'],
                old14=old,new12=new,raw=raw['raw'].astype(float),roles=role,role_labels=train_labels,eligible=allowed,
                bank_sha256=identity['bank_sha256'],raw_entry_sha256=ent['sha256'],raw_checkpoint_sha256=raw['checkpoint_sha256'])
            target=ROOT/'inputs_private'/f'fold{fold}_screen{screen}.pkl'
            if target.exists():assert read(target)['binding']==bound
            else:save(target,view)
            entries.append(dict(fold=fold,screen=screen,path=str(target),sha256=sha(target)))
    save(ROOT/'INPUT_REGISTRY_PRIVATE.pkl',dict(binding=bound,entries=entries))
    pd.DataFrame(audits).to_csv(PUB/'OWN_TRAIN_ELIGIBILITY.csv',index=False)
    write(PUB/'INPUT_ADMISSION.json',dict(status='PASS',original_inner_partitions=20,group_partition_checks=120,
        new_group_eligibility_counts={str(i):sum(row['eligible'] and row['feature']==name for row in audits) for i,name in enumerate(GROUPS)},
        contexts=CONTEXTS,uniform_old_and_new_physiology_preprocessing=True,HELD_labels_omitted_from_selector_inputs=True,
        new_encoders=0,new_Outer_access=False,registry_sha256=sha(ROOT/'INPUT_REGISTRY_PRIVATE.pkl')))
    print('ALL20_INPUTS_ADMITTED_HELD_LABELS_OMITTED',flush=True)


if __name__=='__main__':main()
