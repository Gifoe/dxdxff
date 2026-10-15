param(
    [ValidateSet('Cutover','Coordinator','LaneA','LaneB')][string]$Mode='Cutover',
    [int]$OriginalParentPID=19212,
    [int]$OriginalChildPID=12720,
    [string]$RunTag='accelerated_01'
)
$ErrorActionPreference='Stop'
$taskCode='D:\m8_raw_backbones_pr_seed42_v1'
$taskRuntime='D:\m8_raw_backbones_pr_seed42_v1_runtime'
$taskExecution='D:\m8_raw_backbones_pr_seed42_v1_execution_v1'
$taskPython='E:\Anaconda\python.exe'
$taskPS=(Get-Command powershell.exe).Source
$taskSelf=Join-Path $taskExecution 'parallel_resume.ps1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'

function Hash([string]$p){(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
function Status([string]$name,[hashtable]$data){
    $data['updated_utc']=[DateTime]::UtcNow.ToString('o')
    $path=Join-Path $taskExecution ($name+'.json')
    $temporary=$path+'.partial'
    [IO.File]::WriteAllText($temporary,($data | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $path -Force
}
function LatestCheckpoint {
    Get-ChildItem -LiteralPath (Join-Path $taskRuntime 'TimeConvCNN\PR') -Filter LAST_PRIVATE.pt -Recurse -File | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
}
function LaunchLane([string]$lane){
    $log=Join-Path $taskExecution ($lane+'_'+$RunTag+'.log')
    $err=Join-Path $taskExecution ($lane+'_'+$RunTag+'.err')
    if((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $err)){throw ('Do not overwrite previous lane logs: '+$lane)}
    Start-Process -FilePath $taskPS -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$taskSelf,'-Mode',$lane,'-RunTag',$RunTag) -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
}
try {
    if($Mode -eq 'Cutover'){
        if(Test-Path -LiteralPath (Join-Path $taskExecution 'CUTOVER_AUDIT.json')){throw 'Cutover already performed; inspect state before any new operation'}
        $smoke=Get-Content -LiteralPath D:\m8_raw_backbones_pr_seed42_v1_accelerated_smoke_runtime\public\SMOKE_TEST_AUDIT.json -Raw | ConvertFrom-Json
        if($smoke.status -ne 'PASS'){throw 'Accelerated four-backbone smoke not PASS'}
        $parent=Get-CimInstance Win32_Process -Filter "ProcessId=$OriginalParentPID"
        $child=Get-CimInstance Win32_Process -Filter "ProcessId=$OriginalChildPID"
        if($parent.Name -ne 'python.exe' -or $parent.CommandLine -notlike ('*'+$taskCode+'\run_all.py*')){throw 'Original parent identity differs'}
        if($child.Name -ne 'python.exe' -or $child.ParentProcessId -ne $OriginalParentPID -or $child.CommandLine -notlike ('*'+$taskCode+'\train.py*') -or $child.CommandLine -notlike '*TimeConvCNN*'){throw 'Original child identity differs'}
        $inputHash=Hash (Join-Path $taskRuntime 'INPUT_FREEZE_PRIVATE.json')
        $protocolHash=Hash (Join-Path $taskCode 'configs\PROTOCOL_LOCK.json')
        $executionFiles=@('fast_engine.py','admit_fast.py','worker.py','parallel_resume.ps1')
        $executionHashes=@{}
        foreach($f in $executionFiles){$executionHashes[$f]=Hash (Join-Path $taskExecution $f)}
        $old=LatestCheckpoint
        Status 'CUTOVER_STATUS' @{status='WAITING_FOR_NEXT_ATOMIC_COMPLETE_EPOCH';initial_last_utc=$old.LastWriteTimeUtc.ToString('o');original_parent_pid=$OriginalParentPID;original_child_pid=$OriginalChildPID}
        $deadline=[DateTime]::UtcNow.AddMinutes(10)
        do {
            Start-Sleep -Milliseconds 100
            $current=LatestCheckpoint
            $live=Get-CimInstance Win32_Process -Filter "ProcessId=$OriginalChildPID"
            if(-not $live){throw 'Original child exited before planned cutover; inspect normal phase transition/failure, do not assume checkpoint cutover'}
            if([DateTime]::UtcNow -gt $deadline){throw 'No fresh completed epoch within bounded cutover wait'}
        } while($current.FullName -eq $old.FullName -and $current.LastWriteTimeUtc -eq $old.LastWriteTimeUtc)
        # LAST_PRIVATE is atomically replaced only after a whole completed epoch.
        # Stop only the positively identified task parent and trainer; no tree kill.
        Stop-Process -Id $OriginalParentPID -Force
        if(Get-Process -Id $OriginalChildPID -ErrorAction SilentlyContinue){Stop-Process -Id $OriginalChildPID -Force}
        Start-Sleep -Milliseconds 500
        $remaining=Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -like ('*'+$taskCode+'*') -and $_.CommandLine -match '(train.py|outer.py|run_all.py)'}
        if($remaining){throw 'A scientific task process remains; do not start duplicates'}
        $backupRoot=Join-Path $taskRuntime 'ACCELERATION_CUTOVER_BACKUP'
        New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
        $checkpointHash=Hash $current.FullName
        $backup=Join-Path $backupRoot 'LAST_AT_CUTOVER_PRIVATE.pt'
        if(Test-Path -LiteralPath $backup){throw 'Existing cutover backup must not be overwritten'}
        Copy-Item -LiteralPath $current.FullName -Destination $backup
        if((Hash $backup) -ne $checkpointHash){throw 'Cutover backup whole-file hash mismatch'}
        if((Hash (Join-Path $taskRuntime 'INPUT_FREEZE_PRIVATE.json')) -ne $inputHash -or (Hash (Join-Path $taskCode 'configs\PROTOCOL_LOCK.json')) -ne $protocolHash){throw 'Frozen input/protocol changed'}
        Status 'EXECUTION_MANIFEST' @{status='LOCKED';execution_code_sha256=$executionHashes;original_input_freeze_sha256=$inputHash;original_protocol_sha256=$protocolHash;scientific_sources_modified=$false}
        Status 'CUTOVER_AUDIT' @{status='PASS';checkpoint_sha256=$checkpointHash;backup_sha256=(Hash $backup);complete_epochs_preserved=$true;original_input_freeze_sha256=$inputHash;original_protocol_sha256=$protocolHash;scientific_sources_modified=$false;new_hyperparameters=$false;new_data_or_predictions=$false;two_execution_lanes=@('TimeConvCNN','EEGNet then SEEGNet then SEEGformer')}
        Status 'CUTOVER_STATUS' @{status='PARALLEL_COORDINATOR_STARTED';coordinator_pid=$PID;run_tag=$RunTag}
        # Keep the owning SSH/PowerShell execution alive. OpenSSH on this server
        # reaps detached Start-Process children when the owning SSH job exits.
        & $taskSelf -Mode Coordinator -RunTag $RunTag
    } elseif($Mode -eq 'Coordinator'){
        $laneA=LaunchLane 'LaneA';$laneB=LaunchLane 'LaneB'
        Status 'COORDINATOR_STATUS' @{status='RUNNING';coordinator_pid=$PID;lane_a_pid=$laneA.Id;lane_b_pid=$laneB.Id;run_tag=$RunTag}
        Wait-Process -Id @($laneA.Id,$laneB.Id)
        $a=Get-Content (Join-Path $taskExecution 'LaneA_STATUS.json') -Raw | ConvertFrom-Json
        $b=Get-Content (Join-Path $taskExecution 'LaneB_STATUS.json') -Raw | ConvertFrom-Json
        if($a.status -ne 'COMPLETE' -or $b.status -ne 'COMPLETE'){throw 'One or more independent lanes failed; preserve checkpoints and inspect lane status/logs'}
        # Aggregate once after both lanes; prevent a simultaneous partial-table race.
        & $taskPython -u (Join-Path $taskExecution 'worker.py') aggregate
        if($LASTEXITCODE -ne 0){throw 'Final full-table aggregation failed'}
        & $taskPython -u (Join-Path $taskExecution 'worker.py') verify
        if($LASTEXITCODE -ne 0){throw 'Final independent verification failed; do not rerun scientific tests'}
        Status 'COORDINATOR_STATUS' @{status='ALL_EIGHT_OUTER_AND_VERIFICATION_COMPLETE'}
    } else {
        $models=if($Mode -eq 'LaneA'){@('TimeConvCNN')}else{@('EEGNet','SEEGNet','SEEGformer')}
        foreach($model in $models){
            foreach($phase in @('train','outer')){
                Status ($Mode+'_STATUS') @{status='RUNNING';model=$model;phase=$phase;worker_sources_unchanged=$true}
                & $taskPython -u (Join-Path $taskExecution 'worker.py') $phase --model $model
                if($LASTEXITCODE -ne 0){throw ('Frozen worker failed: '+$model+' '+$phase+' exit '+$LASTEXITCODE)}
            }
        }
        Status ($Mode+'_STATUS') @{status='COMPLETE';models=$models}
    }
} catch {
    Status ($Mode+'_STATUS') @{status='ENGINEERING_FAILURE';exception=$_.Exception.Message;preserve_all_logs_and_checkpoints=$true}
    throw
}
