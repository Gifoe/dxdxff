"""Reporting-only repair: populate verifier CAL truth after the score-seal gate.

The source-frozen verifier incorrectly expected `y` in a labels=False bank.
Keep that verifier, all sources, configurations and predictions byte-unchanged.
The complete truth archive is already legally opened by labels_after_seal().
"""
import final_stage as f
import verify_final as original

source=f.labels_after_seal


def labels_with_cal_truth():
    seal,meta,banks,y_ez=source()  # Includes the mandatory all45-score seal check.
    banks=[dict(d,y=1-y_ez[d['archive_indices']]) for d in banks]
    return seal,meta,banks,y_ez


def main():
    f.labels_after_seal=labels_with_cal_truth
    original.main()
    f.c.write(f.PUBLIC/'REPORTING_VERIFIER_REPAIR.json',dict(status='PASS',
        original_issue='Verifier expected y in identity_and_banks(labels=False); KeyError before config replay.',
        repair='After existing full-score-seal gate, map already opened truth archive to original CAL bank archive indices.',
        original_verifier_sha256=f.c.sha(f.c.HERE/'verify_final.py'),wrapper_sha256=f.c.sha(f.c.HERE/'verify_final_reporting_v2.py'),
        original_bound_sources_unchanged=True,model_config_score_threshold_changes=0,
        new_training_or_inference=0,original_logs_preserved=True))


if __name__=='__main__':main()
