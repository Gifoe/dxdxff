"""Label-free FIT magnitude audit and exact matched B0 validation replay."""
import json
import numpy as np
import pandas as pd
import torch
from runtime import args,setup,checkpoint,patient_rows


def quantiles(x,w):
    ix=np.argsort(x,kind='stable');v=x[ix];ww=w[ix];cdf=np.cumsum(ww)-ww/2
    return np.interp([0,.05,.25,.5,.75,.95,1],cdf,v)


def main():
    a=args();lock,ref,c,core,original,trainer,models,banks,binding=setup(a)
    out=a.runtime/'public/PRETRAIN_AUDIT.json';assert not out.exists()
    summaries=[];geometry=[];audits=[];baseline_rows=[];baseline_selections=[]
    for fold,d in enumerate(banks,1):
        x,pre=core.prepare(d['raw_x'],d['patient'],d['train'])
        assert np.array_equal(x,d['x'])
        duplicate_main=[44,45,50,51,46,47,48,49,26,27,32,33,28,29,30,31]
        zero=[0,1,2,3,18,19,20,21,36,37,38,39,54,55,56,57]
        assert np.array_equal(d['raw_x'][:,72:88],d['raw_x'][:,duplicate_main],equal_nan=True)
        assert np.array_equal(x[:,72:88],x[:,duplicate_main]) and (x[:,zero]==0).all()
        for k in ['mean','scale','var','imputer_statistics']:assert np.array_equal(pre[k],d['pre'][k])
        stats,s,w=models.fit_statistics(x,d['patient'],d['train']);tr=d['train'];va=d['val']
        eps=original.SourceA0().network[0].eps;assert eps==1e-5
        for j,name in enumerate(['channel_mean','channel_log_std']):
            q=quantiles(s[tr,j],w[tr]);summaries.append({'fold':fold,'statistic':name,'FIT_patients':stats['FIT_patients'],'FIT_channels':len(tr),
                'patient_weighted_mean':stats['mean'][j],'patient_weighted_std':stats['scale'][j],
                'within_patient_variance':stats['within_patient_variance'][j],'between_patient_variance':stats['between_patient_variance'][j],
                **dict(zip(['minimum','q05','q25','median','q75','q95','maximum'],q)),
                'nonfinite_fraction':float((~np.isfinite(s[tr,j])*w[tr]).sum()),
                'degenerate_input_variance_fraction':float((models.magnitude(torch.from_numpy(x))[tr,1].exp().square().numpy()<=eps+1e-12)@w[tr])})
        col_std=np.std(x[tr].astype(float),axis=0)
        ck=torch.load(checkpoint(a,'B0',fold),map_location='cpu',weights_only=False)
        seed=42+1009*fold;core.seed_all(seed);b0=models.constructor(original.SourceA0,'B0',stats)(seed).cuda().eval()
        b0.load_state_dict(ck['model'],strict=True)
        with torch.inference_mode():
            ll=b0(torch.as_tensor(x[va],device='cuda'));logits=ll.cpu().numpy();scores=torch.sigmoid(ll).cpu().numpy()
        selected=core.select_threshold(d['y'][va],scores,d['patient'][va])
        for k in ['macro_f1','ez_f1','threshold']:assert abs(selected[k]-ck[k])<1e-12,('B0 replay',fold,k)
        baseline_rows.extend(patient_rows(c,d,va,scores,ck['threshold'],'B0',fold))
        bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert not bp.exists()
        cell={k:d[k] for k in ['x','patient','channel','center','features','train','val','y','g','pre']}
        cell.update({'stats_normalizer':stats,'binding':binding});core.torch_write(bp,cell)
        core.torch_write(bp.parent/'B0_VAL_PRIVATE.pt',{'binding':binding,'indices':va,'logits':logits,'scores':scores,
            'threshold':ck['threshold'],'checkpoint_sha256':c.sha(checkpoint(a,'B0',fold))})
        baseline_selections.append({'arm':'B0','fold':fold,'epoch':ck['epoch'],'threshold':ck['threshold'],
            'checkpoint_sha256':c.sha(checkpoint(a,'B0',fold)),'bank_sha256':c.sha(bp),'reused':True})
        # Label-free rescaling of fixed synthetic input under frozen B0.
        z=torch.as_tensor(np.random.default_rng(42+fold).normal(size=(8,88)).astype('float32'),device='cuda')
        with torch.inference_mode():base_ln=b0.network[0](z);base_logit=b0(z)
        for scale in [.25,.5,1.,2.,4.]:
            with torch.inference_mode():ln=b0.network[0](z*scale);logit=b0(z*scale)
            geometry.append({'fold':fold,'positive_scale':scale,'input_norm_ratio':scale,
                'LN_output_max_absolute_change':float((ln-base_ln).abs().max()),'A0_logit_max_absolute_change':float((logit-base_logit).abs().max()),
                'LN_eps':eps,'mathematical_diagnostic_not_clinical_benefit':True})
        audits.append({'fold':fold,'bank_sha256':c.sha(bp),'stats_normalizer_digest':c.digest({k:np.asarray(stats[k]).tolist() for k in ['mean','scale','variance']}),
            'B0_validation_predictions_sha256':c.sha(bp.parent/'B0_VAL_PRIVATE.pt'),
            'auxiliary_duplicates_preserved':16,'constant_zero_dimensions_preserved':16,
            'FIT_only_preprocessing_bitwise':True,'B0_validation_selection_exact_replay':True,'retained_features':88,
            'FIT_nonfinite_stats':0,'FIT_stats_nonconstant':True,'preprocessed_feature_std_min':float(col_std.min()),
            'preprocessed_feature_std_max':float(col_std.max()),'preprocessed_feature_min':float(x[tr].min()),'preprocessed_feature_max':float(x[tr].max())})
    assert len(baseline_rows)==65 and len(set(r['patient'] for r in baseline_rows))==47
    core.torch_write(a.runtime/'B0_DEVELOPMENT_METRICS_PRIVATE.pt',baseline_rows)
    c.write_json(a.runtime/'B0_SELECTIONS_PRIVATE.json',baseline_selections)
    pd.DataFrame(summaries).to_csv(a.runtime/'public/MAGNITUDE_STATISTICS.csv',index=False)
    pd.DataFrame(geometry).to_csv(a.runtime/'public/LAYERNORM_SCALE_AUDIT.csv',index=False)
    c.write_json(out,{'status':'PASS','binding':binding,'folds':audits,'validation_appearances':65,'validation_unique_patients':47,
        'global_feature_archive_x_or_y_loaded':False,'outer_scores_or_outcomes_loaded':False,'stats_descriptions_use_FIT_only':True})
    print('FIT_MAGNITUDE_AND_MATCHED_B0_REPLAY_PASS',flush=True)


if __name__=='__main__':main()
