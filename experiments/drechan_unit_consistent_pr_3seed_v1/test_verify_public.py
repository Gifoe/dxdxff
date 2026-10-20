"""Real admission plus negative release checks; no private data or toy metrics."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ucp_public_checker', HERE / 'verify_public.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PublicReleaseChecks(unittest.TestCase):
    def test_real_published_admission(self):
        protocol, cells, reuse, anchors, anchor_keys, roles = checker.check_admission(HERE / 'results')
        self.assertEqual(len(cells), 135)
        self.assertEqual(len(reuse), 45)
        self.assertEqual(len(anchors), 30)
        self.assertEqual(len(anchor_keys), 30)
        self.assertEqual(len(roles), 5)
        self.assertEqual(len(protocol['arms']), 9)

    def test_duplicate_keys_are_not_silently_overwritten(self):
        with self.assertRaisesRegex(AssertionError, 'Duplicate aggregate keys'):
            checker.keyed([{'arm': 'FULL'}, {'arm': 'FULL'}], ['arm'])

    def test_changed_source_digest_is_rejected(self):
        original = checker.read_json

        def changed(path):
            value = original(path)
            if path.name == 'ADMISSION.json':
                value = copy.deepcopy(value)
                value['source']['model.py'] = '0' * 64
            return value

        with mock.patch.object(checker, 'read_json', side_effect=changed):
            with self.assertRaises(AssertionError):
                checker.check_admission(HERE / 'results')

    def test_admission_cannot_claim_formal_fits(self):
        original = checker.read_json

        def changed(path):
            value = original(path)
            if path.name == 'ADMISSION.json':
                value = dict(value, new_formal_fits=1)
            return value

        with mock.patch.object(checker, 'read_json', side_effect=changed):
            with self.assertRaises(AssertionError):
                checker.check_admission(HERE / 'results')

    def test_full_verification_requires_final_manifest(self):
        original = checker.read_json

        def without_final_manifest(path):
            if path.name == 'PACKAGE_MANIFEST.json':
                raise FileNotFoundError('Missing actual final package')
            return original(path)

        with mock.patch.object(checker, 'read_json', side_effect=without_final_manifest):
            with self.assertRaisesRegex(FileNotFoundError, 'Missing actual final package'):
                checker.check_full(HERE / 'results')

    def test_full_export_allowlist_is_explicit(self):
        tree = ast.parse((HERE / 'package_public.py').read_text(encoding='utf-8'))
        allowed = next(ast.literal_eval(node.value) for node in tree.body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'ALLOW'
                               for target in node.targets))
        self.assertEqual(len(allowed), 33)
        self.assertEqual(len(set(allowed)), 33)
        self.assertIn('NULL_CONTROL_RETRAIN_PARITY.csv', allowed)
        self.assertIn('PREPROCESSING_REPLAY.csv', allowed)
        self.assertNotIn('PROGRESS.json', allowed)
        self.assertNotIn('MODEL_FREEZE_PRIVATE.json', allowed)


if __name__ == '__main__':
    unittest.main()
