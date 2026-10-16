"""NumPy-first synthetic tests, with anonymous measured collection/outcome counts."""
import numpy
import pytest
from common import *


class Counts:
    def __init__(self):self.passed=self.failed=self.skipped=0;self.collected=0
    def pytest_collection_finish(self,session):self.collected=len(session.items)
    def pytest_runtest_logreport(self,report):
        if report.when=='call':
            self.passed+=int(report.passed);self.failed+=int(report.failed);self.skipped+=int(report.skipped)


if __name__=='__main__':
    names=['test_pool.py','test_probes.py','test_model.py'];counts=Counts()
    exitcode=pytest.main([*[str(HERE/n) for n in names],'-q'],plugins=[counts])
    assert exitcode==0 and counts.collected==counts.passed==10 and counts.failed==0
    write(PUB/'UNIT_TEST_STATUS.json',dict(status='PASS',collected=counts.collected,passed=counts.passed,
        failed=counts.failed,skipped=counts.skipped,source={n:sha(HERE/n) for n in names},
        role='Synthetic end-of-run regression checks; chronological initial-test incident disclosed separately.'))
