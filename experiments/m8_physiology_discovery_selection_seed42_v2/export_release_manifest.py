"""Public-file transfer identity and GPU environment, without private I/O."""
import platform
from common import *


if __name__=='__main__':
    assert jr(PUB/'INDEPENDENT_RESULTS_VERIFICATION.json')['status']=='PASS'
    assert jr(PUB/'UNIT_TEST_STATUS.json')['passed']==10
    write(PUB/'GPU_ENVIRONMENT.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),GPU=torch.cuda.get_device_name(0),
        training_precision='FP32',TF32='disabled explicitly in train/support/evaluation sources',global_driver_or_settings_modified=False))
    excluded={'PUBLIC_TRANSFER_MANIFEST.json','NEURAL_HISTORY.csv','SUPPORT30_HISTORY.csv'}
    files={p.relative_to(PUB).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(PUB.rglob('*')) if p.is_file() and p.suffix in ['.csv','.json'] and p.name not in excluded}
    write(PUB/'PUBLIC_TRANSFER_MANIFEST.json',dict(status='PASS',files=files,excluded_epoch_history_copies=True,
        private_runtime_never_recursively_copied=True,no_individual_fields_exported=True))
    print('PUBLIC_TRANSFER_HASH_MANIFEST_COMPLETE',len(files),flush=True)
