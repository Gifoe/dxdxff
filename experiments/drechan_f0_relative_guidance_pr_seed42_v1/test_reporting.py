"""Synthetic reporting arithmetic, not experimental performance evidence."""
import ast
from pathlib import Path
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
METRICS = ['macro_f1', 'auroc', 'auprc', 'f1', 'nez_f1', 'accuracy', 'mrr', 'hit_at_1', 'ez_fraction_bias']


def functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


class ReportingTests(unittest.TestCase):
    def test_chunked_intervals_match_scalar_endpoint_resampling(self):
        ns = dict(np=np, METRICS=METRICS)
        functions(HERE/'common.py', ['interval'], ns)
        rng = np.random.default_rng(891)
        values = rng.normal(size=(80, 9))
        draws = np.random.default_rng(42).integers(0, 80, (10000, 80))
        for levels in [(.025, .975), (.05/28, 1-.05/28)]:
            mean, limits = ns['interval'](values, draws, levels)
            for j in range(9):
                expected = np.quantile(values[:, j][draws].mean(1), levels)
                self.assertLess(np.max(np.abs(limits[:, j]-expected)), 1e-14)
                self.assertAlmostEqual(mean[j], values[:, j].mean(), places=14)

    def test_component_failure_is_not_promoted_from_f1_only(self):
        ns = dict(np=np, METRICS=METRICS, PRIMARY=['RAW_PR'], CONTRASTS={'RAW_PR': {'FULL': 1, 'WO_RAW_PR': -1}})
        functions(HERE/'common.py', ['vector'], ns)
        functions(HERE/'report_outer.py', ['component_evidence'], ns)
        rows, matrices = [], {}
        for arm in ['FULL', 'WO_RAW_PR']:
            value = np.full((80, 9), .6)
            if arm == 'FULL':
                value[:, 0] += .02
                value[:, 1] -= .01
            matrices[arm] = value
            for i in range(80):
                rows.append(dict(arm=arm, fold=i//16+1, **dict(zip(METRICS, value[i]))))
        family = [dict(contrast='RAW_PR', metric=m, low=low) for m, low in [('macro_f1', .01), ('auroc', -.02)]]
        result = ns['component_evidence'](rows, matrices, family)[0]
        self.assertFalse(result['both_point_estimates_positive'])
        self.assertFalse(result['both_family_lower_bounds_positive'])
        self.assertEqual(result['folds_both_positive'], 0)
        self.assertFalse(result['stable_within_this_seed_protocol'])
        matrices['FULL'][:, 1] += .03
        for r in rows:
            if r['arm'] == 'FULL':
                r['auroc'] += .03
        family[1]['low'] = .01
        result = ns['component_evidence'](rows, matrices, family)[0]
        self.assertTrue(result['stable_within_this_seed_protocol'])
        self.assertEqual(result['folds_both_positive'], 5)


if __name__ == '__main__':
    unittest.main()
