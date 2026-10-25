"""Engineering-only adapter for three duplicate-key audit constructors.

The admitted run.py bytes, cached first candidate and every statistical expression
remain unchanged. The only substitution merges already identical C fields when
serializing fit metadata. No fitting, label, probability or selection code changes.
"""
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
RUN_SHA='096e4ef66484c58ceef8e800369a94845a661a6d21dd3f041e67362b35af5fc5'
OLD="dict(**meta,**pack['fit'],seconds=pack['seconds'],warnings=';'.join(pack['fit']['warnings']))"
NEW="{**meta,**pack['fit'],'seconds':pack['seconds'],'warnings':';'.join(pack['fit']['warnings'])}"


def adapt(source):
    assert source.count(OLD)==3,'Only the three audited constructor sites are permitted'
    result=source.replace(OLD,NEW)
    assert result.count(NEW)==3 and result.replace(NEW,OLD)==source
    compile(result,'run.py','exec')
    return result


if __name__=='__main__':
    assert len(sys.argv)>1 and sys.argv[1]=='train','This adapter only wraps the training audit constructors'
    path=HERE/'run.py';assert hashlib.sha256(path.read_bytes()).hexdigest()==RUN_SHA
    import core as c
    import os
    root=Path(sys.argv[sys.argv.index('--runtime')+1])
    binding=c.jr(root/'private/INPUT_FREEZE.json');assert binding['source']==c.sources()
    proof=dict(status='APPLIED',type='AUDIT_SERIALIZATION_ONLY',original_run_sha256=RUN_SHA,
        adapter_sha256=c.sha(Path(__file__)),replacement_sites=3,
        unchanged_statistical_source_hashes=c.sources(),first_candidate_preserved=True,
        formal_fits_before_failure=1,Outer_scores_before_failure=0,
        engineering_commit=os.environ['PREDICTION_ENGINEERING_COMMIT'])
    c.write(root/'public/ENGINEERING_AUDIT.json',proof)
    print('AUDIT_SERIALIZATION_ADAPTER_APPLIED 3constructors;original_sources_unchanged;first_candidate_preserved',flush=True)
    source=adapt(path.read_text(encoding='utf-8'))
    sys.argv[0]=str(path)
    exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
