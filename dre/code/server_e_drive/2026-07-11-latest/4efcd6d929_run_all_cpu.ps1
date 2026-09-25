$env:CUDA_VISIBLE_DEVICES=""
$env:OMP_NUM_THREADS="8"
$env:MKL_NUM_THREADS="8"
python -m scripts.task1_aaai_cpu.run_all_cpu @args
