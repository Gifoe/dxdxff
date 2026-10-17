param(
 [Parameter(Mandatory=$true)][int]$InternalPid,
 [Parameter(Mandatory=$true)][string]$ExpectedInternalStartUtc,
 [string]$FinalRunTag='01'
)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$code='D:\raw_joint_moment_macro_seed42_v1';$runtime='D:\raw_joint_moment_macro_seed42_v1_runtime'
$expected=[ordered]@{
 'FINAL_PROTOCOL_LOCK.json'='11840236a07b1e65296f02f5c5cd4e498036b9d2cdc64a15e56af5ef78d75bd5'
 'final_stage.py'='95963ffb8c8606a98301ad6c8ca1f5998bd5dd2009de0047da47435e49fa22ef'
 'verify_final.py'='8eca96a40fa8fff0dd475c39e8583a0daaee9023d068740c7d1be310ff13884a'
 'audit_raw_support.py'='61710585be9521591c6f94e28c474caec275f143a17602f10cfa0be835ea8cd5'
 'run_final.ps1'='5bcedd6bdfc446a29645b8022e5bd4530f5d2eef6b95f10760a0b02c646ad5cd'
 'check_final_gate.py'='791fa95ebff2800367fd890181a04510f8ea94584ec2f9050c915022fbf1831c'
}
$guard=[System.IO.File]::Open((Join-Path $runtime 'handoff_watch.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
$history=[System.Collections.Generic.List[object]]::new()
function State([string]$status) {
 $history.Add([ordered]@{status=$status;time_local=(Get-Date -Format o)})
 $doc=[ordered]@{status=$status;expected_internal_pid=$InternalPid;expected_internal_start_utc=$ExpectedInternalStartUtc;final_run_tag=$FinalRunTag;final_source_sha256=$expected;history=@($history.ToArray())}
 [System.IO.File]::WriteAllText((Join-Path $runtime 'HANDOFF_STATUS_PRIVATE.json'),($doc|ConvertTo-Json -Depth 8),[System.Text.UTF8Encoding]::new($false))
}
function VerifySources {
 foreach($name in $expected.Keys) {
  if ((Get-FileHash -LiteralPath (Join-Path $code $name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected[$name]) {throw 'Frozen final source changed; do not advance.'}
 }
}
try {
 VerifySources
 $start=[DateTime]::Parse($ExpectedInternalStartUtc).ToUniversalTime()
 $p=Get-CimInstance Win32_Process -Filter "ProcessId=$InternalPid"
 while($null -ne $p) {
  if($p.Name -ne 'powershell.exe' -or $p.CreationDate.ToUniversalTime().Ticks -ne $start.Ticks) {throw 'Internal PID identity changed; no guessed handoff.'}
  State 'WAITING_CONFIRMED_LIVE_INTERNAL_PARENT'
  Write-Output "JOINT_RAW_HANDOFF_WAIT live_internal_parent=$InternalPid $(Get-Date -Format o)"
  Start-Sleep -Seconds 20
  $p=Get-CimInstance Win32_Process -Filter "ProcessId=$InternalPid"
 }
 State 'INTERNAL_PARENT_ENDED_CHECKING_COMPLETE_GATES'
 $terminal=Get-Content -LiteralPath "$runtime\public\INTERNAL_TERMINAL.json" -Raw|ConvertFrom-Json
 $verified=Get-Content -LiteralPath "$runtime\public\INDEPENDENT_INTERNAL_VERIFICATION.json" -Raw|ConvertFrom-Json
 if($terminal.status -ne 'COMPLETE_VERIFIED_JOINT_RAW_INTERNAL' -or -not $terminal.final_stage_warranted -or $verified.status -ne 'PASS' -or $verified.independent_CI_metric_checks -ne 126) {throw 'Full independent internal gate missing; do not advance.'}
 if((Get-FileHash -LiteralPath "$runtime\OWN_FIT_SELECTION_PRIVATE.json" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $verified.ownFIT_selection_sha256) {throw 'Frozen ownFIT choices changed.'}
 if(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'raw_joint_moment_macro_seed42_v1\\(admit|train_internal|summarize|final_stage|verify_final|audit_raw_support)\.py'}) {throw 'Owned worker still active; no duplicate/overlap.'}
 VerifySources
 State 'STARTING_ALREADY_FROZEN_FINAL_PIPELINE'
 & "$code\run_final.ps1" -RunTag $FinalRunTag
 $final=Get-Content -LiteralPath "$runtime\final_stage_v1\public\INDEPENDENT_FINAL_VERIFICATION.json" -Raw|ConvertFrom-Json
 $support=Get-Content -LiteralPath "$runtime\final_stage_v1\public\RAW_SUPPORT_AUDIT.json" -Raw|ConvertFrom-Json
 if($final.status -ne 'PASS' -or $support.status -ne 'PASS') {throw 'Final independent/census gate missing.'}
 State 'FINAL_PIPELINE_COMPLETE_VERIFIED'
 Write-Output 'JOINT_RAW_HANDOFF_AND_FINAL_COMPLETE_VERIFIED'
} catch {
 State 'STOPPED_ON_GUARD_OR_PIPELINE_FAILURE_NO_RETRY'
 throw
} finally {$guard.Dispose()}
