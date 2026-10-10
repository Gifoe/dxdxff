"""Local aggregate-only delivery checks, without any private prediction access."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    verification = json.loads((root / 'audit/COMPLETION_VERIFICATION.json').read_text())
    assert verification['status'] == 'PASS'
    verified = 0
    for name, digest in verification['aggregate_artifacts_sha256'].items():
        choices = [root / name, root / 'audit' / name, root / 'results' / name]
        matches = [p for p in choices if p.is_file()]
        assert len(matches) == 1, name
        assert hashlib.sha256(matches[0].read_bytes()).hexdigest() == digest, name
        verified += 1
    banned_columns = {'patient', 'patient_id', 'channel', 'channel_name', 'y',
                      'y_true', 'label', 'score', 'prediction', 'logit'}
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        assert p.suffix.lower() in {'.py', '.md', '.json', '.csv', ''}, p
        assert '__pycache__' not in p.parts and '_PRIVATE' not in p.name, p
        if p.suffix == '.csv':
            with p.open(encoding='utf-8', newline='') as handle:
                headers = next(csv.reader(handle))
            assert banned_columns.isdisjoint(headers), p
    report = (root / 'FINAL_REPORT.md').read_text(encoding='utf-8')
    assert 'RANK4_REMAINS_BEST' in report
    for question in range(1, 25):
        assert f'\n{question}. ' in report, question
    print(f'PUBLIC_AGGREGATE_HASHES_AND_PRIVACY_PASS {verified}')


if __name__ == '__main__':
    main()
