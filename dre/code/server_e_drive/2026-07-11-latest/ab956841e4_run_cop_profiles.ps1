param([string[]]$Profiles=@("O0_FEATURE_PHENOTYPE","O1_FEATURE_RAW_PROPAGATION","O2_COP_FULL","O3_COP_GROUP_GATED"),[int[]]$Seeds=@(42))
$ErrorActionPreference="Stop"
$Project=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. "$Project\scripts\task2\ensure_npam_protocol.ps1"
$PythonExe=if($env:DRE_PYTHON_EXE){$env:DRE_PYTHON_EXE}else{"python"};$FeatureCache=if($env:DRE_TASK2_COP_FEATURE_CACHE){$env:DRE_TASK2_COP_FEATURE_CACHE}else{$env:DRE_TASK1_FEATURE_CACHE_PATH}
foreach($Profile in $Profiles){foreach($Seed in $Seeds){
  $Output="$env:DRE_TASK2_COP_OUTPUT_DIR\$Profile\seed_$Seed";$Audit="$Output\cop_protocol_audit.json"
  if(Test-Path $Audit){$State=Get-Content $Audit -Raw|ConvertFrom-Json;if($State.completed_outer_folds -eq 5 -and $State.paper_valid -eq $true){Write-Host "Skipping completed $Profile seed $Seed" -ForegroundColor Yellow;continue}}
  Write-Host "`n========== Starting COP $Profile seed $Seed ==========" -ForegroundColor Cyan
  $Arguments=@("$Project\scripts\task2\run_cop_outcome.py","--profile",$Profile,"--feature_cache",$FeatureCache,"--raw_cache",$env:DRE_TASK1_RAW_CACHE_PATH,"--outcome_table",$env:DRE_TASK2_OUTCOME_TABLE,"--fold_manifest",$env:DRE_TASK2_FOLD_MANIFEST,"--exclusion_manifest",$env:DRE_TASK2_EXCLUSION_MANIFEST,"--p2_checkpoint_root",$env:DRE_TASK1_P2_CHECKPOINT_ROOT,"--p2_runtime_root",$env:DRE_TASK1_P2_RUNTIME_ROOT,"--p2_training_manifest",$env:DRE_TASK1_P2_TRAINING_MANIFEST,"--protocol","outer_cv","--outer_folds","5","--inner_folds","3","--seed",$Seed,"--bootstrap_repeats","2000","--output_dir",$Output,"--phenotype_cache_dir",$env:DRE_TASK2_COP_PHENOTYPE_CACHE_DIR,"--p2_evidence_cache_dir","$env:DRE_TASK2_COP_OUTPUT_DIR\_p2_evidence\seed_$Seed","--resume","--strict")
  & $PythonExe @Arguments;if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
}}
