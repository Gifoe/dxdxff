# Reproduction driver, added after execution. Never deletes completed artifacts.
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$taskCode='D:\m8_physiology_discovery_selection_seed42_v2'
$taskRoot='D:\m8_physiology_discovery_selection_seed42_v2_runtime'
$taskGPU='E:\Anaconda\python.exe'
$taskCPU='E:\Anaconda\envs\persist_stable_251\python.exe'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1'
New-Item -ItemType Directory -Path $taskRoot -Force | Out-Null
$taskLock=[System.IO.File]::Open((Join-Path $taskRoot 'remaining_coordinator.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
function Read-Stage([string]$Name,[string]$Required){
    $taskFile=Join-Path "$taskRoot\public" $Name
    if(-not (Test-Path -LiteralPath $taskFile)){return $false}
    return ((Get-Content -LiteralPath $taskFile -Raw | ConvertFrom-Json).status -eq $Required)
}
function Run-Stage([string]$Script,[bool]$CPU=$false){
    $taskExe=if($CPU){$taskCPU}else{$taskGPU}
    $taskProgram="import sys,psutil;psutil.Process().cpu_affinity([16]);import numpy,runpy;sys.path.insert(0,r'$taskCode');runpy.run_path(r'$taskCode\$Script',run_name='__main__')"
    & $taskExe -u -X faulthandler -c $taskProgram
    if($LASTEXITCODE -ne 0){throw ('Diagnose failed stage, do not blindly retry: '+$Script)}
}
try {
    $taskDuplicate=Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match 'm8_physiology_discovery_selection_seed42_v2'}
    if($taskDuplicate){throw 'Another experiment worker exists'}
    if(Read-Stage 'INDEPENDENT_RESULTS_VERIFICATION.json' 'PASS'){
        Write-Output 'ALREADY_COMPLETE: no repeat training or scientific test. Read report and audit.'
        return
    }
    Run-Stage 'test_entry.py'
    if(-not (Read-Stage 'POOL_STATUS.json' 'COMPLETE_UNLABELED32_POOL')){Run-Stage 'build_pool.py'}
    if(-not (Read-Stage 'SCREEN_STATUS.json' 'ALL5_FIT_ONLY_SELECTIONS_FROZEN')){Run-Stage 'select_features.py' $true}
    if(-not (Read-Stage 'INDEPENDENT_SELECTION_VERIFICATION.json' 'PASS')){Run-Stage 'verify_selection.py' $true}
    if(-not (Read-Stage 'BIOLOGY_STATUS.json' 'COMPLETE_FIT_ONLY_PHYSIOLOGICAL_ASSOCIATIONS')){Run-Stage 'biology.py' $true}
    if(-not (Read-Stage 'REDUNDANCY_STATUS.json' 'PASS')){Run-Stage 'redundancy.py' $true}
    if(-not (Test-Path -LiteralPath "$taskRoot\MODEL_FREEZE_PRIVATE.pkl")){Run-Stage 'train.py'}
    if(-not (Read-Stage 'SUPPORT30_STATUS.json' 'ALL5_SUPPORT_MATCHED_RAW_MODELS_FROZEN')){Run-Stage 'support_control.py'}
    Run-Stage 'restore_status.py' $true
    if(-not (Read-Stage 'OUTER_STATUS.json' 'ALL105_FROZEN_SCORE_CELLS_SEALED')){Run-Stage 'evaluate.py'}
    if(-not (Read-Stage 'REPORT_VERIFICATION.json' 'PASS')){Run-Stage 'report.py' $true}
    Run-Stage 'verify_results.py' $true
    Run-Stage 'source_complementarity.py' $true
    Run-Stage 'export_audit.py' $true
    Write-Output 'COMPLETE: fetch only whitelisted anonymous public files.'
} finally {$taskLock.Dispose()}
