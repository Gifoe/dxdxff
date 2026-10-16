"""Outcome-free diagnosis of cross-backend and batch-shape replay roundoff."""
import copy
from common import *
from head import Head, groups


def main():
    setup()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reports = []
    for ctx in contexts():
        z, cp = inputs(ctx)
        raw = z['1111']; patient = ctx['data']['patient']
        reference = ctx['stored']['raw_logit_nez']; model = Head('REFIT', cp['model'], ctx['fold']).eval()
        modes = {}
        with torch.inference_mode():
            modes['cpu_full'] = model(torch.from_numpy(raw), groups(patient)).numpy()
            for device, dtype, label in [('cpu', torch.float32, 'cpu_patient'),
                                         ('cpu', torch.float64, 'cpu64_patient'),
                                         ('cuda', torch.float32, 'cuda_patient')]:
                m = copy.deepcopy(model).to(device=device, dtype=dtype)
                prediction = np.empty(len(raw), np.float32)
                for pid in sorted(set(patient)):
                    ix = np.flatnonzero(patient == pid)
                    prediction[ix] = m(torch.as_tensor(raw[ix], device=device, dtype=dtype),
                                       groups(patient[ix]).to(device)).float().cpu().numpy()
                modes[label] = prediction
            with torch.backends.mkldnn.flags(enabled=False):
                modes['cpu_no_mkldnn'] = model(torch.from_numpy(raw), groups(patient)).numpy()
        cal = ctx['roles']['cal']; grid = np.round(np.arange(0, 1.0001, .005), 3)
        r = dict(fold=ctx['fold'], screen=ctx['screen'])
        for label, prediction in modes.items():
            a = probability(prediction[cal]); b = probability(reference[cal])
            r[label] = dict(logit_max_error=float(np.max(np.abs(prediction-reference))),
                            probability_max_error=float(np.max(np.abs(probability(prediction)-probability(reference)))),
                            calibration_grid_disagreements=int(np.count_nonzero((a[:,None]>=grid)!=(b[:,None]>=grid))))
        reports.append(r)
        print('NUMERICAL_REPLAY_DIAGNOSIS', r, flush=True)
    write(PUB/'REPLAY_DIAGNOSIS.json', dict(outcome_metrics_read=False, new_fits=0, reports=reports))


if __name__ == '__main__': main()
