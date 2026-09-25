param([string]$Profile="O0_FEATURE_PHENOTYPE",[int]$Seed=42)
$ErrorActionPreference="Stop"
$Project=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. "$Project\scripts\task2\ensure_npam_protocol.ps1"
$PythonExe=if($env:DRE_PYTHON_EXE){$env:DRE_PYTHON_EXE}else{"python"};$FeatureCache=if($env:DRE_TASK2_COP_FEATURE_CACHE){$env:DRE_TASK2_COP_FEATURE_CACHE}else{$env:DRE_TASK1_FEATURE_CACHE_PATH}
$Arguments=@("$Project\scripts\task2\run_cop_outcome.py","--profile",$Profile,"--feature_cache",$FeatureCache,"--raw_cache",$env:DRE_TASK1_RAW_CACHE_PATH,"--outcome_table",$env:DRE_TASK2_OUTCOME_TABLE,"--fold_manifest",$env:DRE_TASK2_FOLD_MANIFEST,"--exclusion_manifest",$env:DRE_TASK2_EXCLUSION_MANIFEST,"--protocol","quick","--max_outer_folds","1","--max_patients","30","--inner_folds","3","--seed",$Seed,"--bootstrap_repeats","200","--output_dir","$env:DRE_TASK2_COP_OUTPUT_DIR\quick\$Profile\seed_$Seed","--phenotype_cache_dir",$env:DRE_TASK2_COP_PHENOTYPE_CACHE_DIR,"--p2_evidence_cache_dir","$env:DRE_TASK2_COP_OUTPUT_DIR\_quick_p2_evidence\seed_$Seed","--resume")
if($Profile -in @("O2_COP_FULL","O3_COP_GROUP_GATED")){$Arguments+=@("--p2_checkpoint_root",$env:DRE_TASK1_P2_CHECKPOINT_ROOT,"--p2_runtime_root",$env:DRE_TASK1_P2_RUNTIME_ROOT)}
& $PythonExe @Arguments
exit $LASTEXITCODE
