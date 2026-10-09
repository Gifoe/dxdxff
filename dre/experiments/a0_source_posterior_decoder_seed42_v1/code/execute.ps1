param([string]$Runtime='C:\a0_source_posterior_decoder_seed42_runtime',
      [string]$Experiment='E:\DRE-nips\new-pipeline\7-11\a0_source_posterior_decoder_seed42_v1',
      [string]$Source='E:\DRE-nips\new-pipeline\7-11',
      [string]$Python='C:\pr_uncertainty_aware_supervision_seed42_runtime\runtime312\Scripts\python.exe',
      [string]$Attempt='02')
$ErrorActionPreference='Stop'
$Code=Join-Path $Experiment 'code'
$Protocol=Join-Path $Experiment 'PROTOCOL_LOCK.json'
function Run-Step([string]$Name,[string]$Script,[string[]]$Arguments) {
    $Out=Join-Path $Runtime ($Name+'.log')
    $Err=Join-Path $Runtime ($Name+'.err')
    if ((Test-Path -LiteralPath $Out) -or (Test-Path -LiteralPath $Err)) { throw "Preserve old logs; choose a new orchestration attempt name before resume: $Name" }
    $SavedPreference=$ErrorActionPreference
    $ErrorActionPreference='Continue'
    & $Python -u $Script @Arguments 1> $Out 2> $Err
    $StepCode=$LASTEXITCODE
    $ErrorActionPreference=$SavedPreference
    if ($StepCode -ne 0) { throw "Step $Name failed with exit $StepCode; inspect preserved private logs" }
}
if (-not (Test-Path -LiteralPath (Join-Path $Runtime 'public\A0_REPRODUCTION.json'))) {
    Run-Step ('prepare_'+$Attempt) (Join-Path $Code 'prepare.py') @('--runtime',$Runtime,'--protocol',$Protocol,'--source',$Source)
}
if (-not (Test-Path -LiteralPath (Join-Path $Runtime 'public\DECODER_UNIT_TEST_AUDIT.json'))) {
    Run-Step ('tests_'+$Attempt) (Join-Path $Experiment 'tests\test_contract.py') @('--runtime',$Runtime,'--protocol',$Protocol,'--source',$Source)
}
foreach ($Phase in @('formal','teachers','posterior')) {
    foreach ($Fold in 1..5) {
        $Marker=@{formal='FORMAL_COMPLETE.json';teachers='TEACHERS_COMPLETE.json';posterior='POSTERIOR_COMPLETE.json'}[$Phase]
        if (Test-Path -LiteralPath (Join-Path $Runtime ('fold'+$Fold+'\'+$Marker))) { continue }
        Run-Step ($Phase+'_'+$Fold+'_'+$Attempt) (Join-Path $Code 'run_fold.py') @('--runtime',$Runtime,'--protocol',$Protocol,'--source',$Source,'--phase',$Phase,'--fold',[string]$Fold,'--device','cuda')
    }
}
foreach ($Fold in 1..5) {
    if (Test-Path -LiteralPath (Join-Path $Runtime ('fold'+$Fold+'\INDEPENDENT_REPLAY.json'))) { continue }
    Run-Step ('verify_'+$Fold+'_'+$Attempt) (Join-Path $Code 'verify_fold.py') @('--runtime',$Runtime,'--fold',[string]$Fold)
}
Run-Step ('finalize_'+$Attempt) (Join-Path $Code 'finalize.py') @('--runtime',$Runtime,'--protocol',$Protocol)
