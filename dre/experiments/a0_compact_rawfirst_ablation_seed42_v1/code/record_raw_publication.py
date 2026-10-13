"""Record externally verified raw-result publication; not a scientific gate tuning."""
import json
import re
import sys
from audit_sources import arguments,context


def main():
 commit=sys.argv.pop(1);assert re.fullmatch('[0-9a-f]{40}',commit)
 a=arguments();o=context(a);c=o[2];pub=a.runtime/'public'
 assert json.loads((pub/'RAW_INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
 assert json.loads((pub/'RAW_OUTER_STATUS.json').read_text())['status']=='ONE_FROZEN_RAW_OUTER_PASS_COMPLETE'
 report=pub/'RAW_ONLY_REPORT.md';assert report.exists()
 target=a.runtime/'RAW_PUBLICATION_PRIVATE.json';assert not target.exists()
 c.write_json(target,dict(github_commit=commit,verified_raw_outer_status_sha256=c.sha(pub/'RAW_OUTER_STATUS.json'),
  verified_raw_independent_sha256=c.sha(pub/'RAW_INDEPENDENT_VERIFICATION.json'),raw_report_sha256=c.sha(report),
  note='Caller must verify remote commit/tree/branch before invoking; exact frozen feature matrix unchanged'))
 print('RAW_PUBLICATION_RECORDED_FEATURE_STAGE_NOW_ALLOWED',flush=True)


if __name__=='__main__':main()
