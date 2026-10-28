from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent.parent
PUB=ROOT/'results/public_abcd'
def jr(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=jr(PUB/'manifest.json');provenance=jr(PUB/'PROVENANCE_SUPPLEMENT.json')
manifest=dict(schema='DRECHAN_PR_RANKING_ROBUSTNESS_RESULTS_V1',status='ABCD_COMPLETE_VERIFIED',
              scientific_source_commits=dict(ABC='3eac5b44073a9c18de54d74f7e2f32b1f08f2a0f',D='4d45a5b02524e4c0eb2d01c45454e780e261bfb9',normalization='eede217c80985bbb4f91f8df5580d4fcfbef0f6b'),
              anonymous_archive_sha256=dict(ABC='04aad3aa1a95afd1d1dcee4ec8f33f784e00bc6eb7e1a6a34ded4eb4a5f1af36',ABCD='539dd61e587a845b33754f3a83bf6bcac5d975f5464ad110340b0db7cfb7d2fa',normalization='b2831887e688df71c0d5395da2feea3f45c488371756e510c61ff9f2a253aa6e'),
              ABC_verification=jr(PUB/'ABC_VERIFICATION.json'),D_admission=jr(PUB/'D_ADMISSION.json'),D_verification=jr(PUB/'D_VERIFICATION.json'),checkpoint_audit=base['checkpoint_audit'],score_provenance=provenance,CPU_GPU_endpoint_parity=jr(ROOT/'logs/CPU_GPU_PARITY.json'),
              bootstrap=dict(draws=10000,seed=42,unit='patient',paired=True,stratification='within_source',family_sizes=[6,12,9,36]),individual_outputs='TRUSTED_SERVER_PRIVATE_ONLY',original_JBHI_tables_figures_unchanged=True)
manifest['artifact_files_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p!=ROOT/'manifest.json' and p.suffix in ['.py','.md','.tex','.json','.csv','.pdf','.svg','.png']}
(ROOT/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2),encoding='utf-8',newline='\n')
print('RESULT_MANIFEST_BUILT',len(manifest['artifact_files_sha256']))
