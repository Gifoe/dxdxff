param([string]$RunTag='02')
$ErrorActionPreference='Stop'
$taskCode='D:\m8_physiology_feature_discovery_seed42_v1'
$taskRuntime='D:\m8_physiology_feature_discovery_seed42_v1_runtime'
$stdout=Join-Path $taskRuntime ('integrated_'+$RunTag+'.log')
$stderr=Join-Path $taskRuntime ('integrated_'+$RunTag+'.err')
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'New logs required' }
$workers=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train_old7_m8|train_integrated' }
if ($workers) { throw 'Existing training worker; no duplicate allowed' }
$state=Get-Content -Raw -LiteralPath ($taskRuntime+'\old7_matched_M8\public\STATUS.json') | ConvertFrom-Json
if ($state.status -ne 'ALL80_MATCHED_OLD7_D1_AND_M8_DEVELOPMENT_CELLS_COMPLETE') { throw 'Old7 matrix incomplete' }
$oldFormal=Get-ChildItem -LiteralPath ($taskRuntime+'\integrated_M8') -Filter 'LAST_PRIVATE.pt' -Recurse -File
if ($oldFormal) { throw 'Prior actual training exists; inspect before switching' }
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
& 'E:\Anaconda\python.exe' ($taskCode+'\test_pairing_index.py')
if ($LASTEXITCODE -ne 0) { throw 'Axis regression failed' }
$worker=Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList @('-u',($taskCode+'\train_integrated_v2.py')) -WorkingDirectory $taskCode -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Output ('INTEGRATED_V2_STARTED PID='+$worker.Id)
while (-not $worker.HasExited) {
    Write-Output ('INTEGRATED_V2_VERIFIED_LIVE PID='+$worker.Id+' '+(Get-Date -Format o))
    Start-Sleep -Seconds 20
    $worker.Refresh()
}
$worker.WaitForExit()
Write-Output ('INTEGRATED_V2_TERMINAL exit='+$worker.ExitCode)
if ($worker.ExitCode -ne 0) { throw 'Inspect preserved logs; no automatic retry' }
