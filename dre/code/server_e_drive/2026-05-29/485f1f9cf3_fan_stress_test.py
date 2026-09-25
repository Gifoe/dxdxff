from __future__ import annotations

import argparse
import ctypes
import math
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import get_context
from typing import Optional


GB = 1024**3
MB = 1024**2


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _total_system_memory_gb() -> Optional[float]:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullTotalPhys / GB
        return None

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / GB
    except (AttributeError, ValueError, OSError):
        return None


def _default_ram_target_gb() -> float:
    total = _total_system_memory_gb()
    if total is None:
        return 48.0
    return max(1.0, min(total * 0.75, total - 8.0))


def _query_nvidia_smi(gpu_index: int) -> Optional[dict[str, str]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    query = "temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw"
    cmd = [
        nvidia_smi,
        f"--id={gpu_index}",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    values = [part.strip() for part in result.stdout.strip().split(",")]
    if len(values) != 5:
        return None
    return {
        "temp_c": values[0],
        "util_pct": values[1],
        "mem_used_mb": values[2],
        "mem_total_mb": values[3],
        "power_w": values[4],
    }


def cpu_worker(stop_event, worker_id: int, load_percent: float) -> None:
    load = _clamp(load_percent / 100.0, 0.0, 1.0)
    period = 0.10
    x = 1.0 + worker_id
    while not stop_event.is_set():
        active_until = time.perf_counter() + period * load
        while time.perf_counter() < active_until and not stop_event.is_set():
            x = math.sin(x) * math.cos(x + 0.1) + math.sqrt(abs(x) + 1.0)
            if x > 10_000.0:
                x = 1.0
        rest = period * (1.0 - load)
        if rest > 0:
            time.sleep(rest)


def ram_worker(stop_event, target_gb: float, chunk_mb: int, touch_stride_pages: int) -> None:
    target_bytes = int(max(0.0, target_gb) * GB)
    chunk_bytes = max(1, int(chunk_mb)) * MB
    page = 4096
    chunks: list[bytearray] = []
    allocated = 0

    try:
        while allocated < target_bytes and not stop_event.is_set():
            size = min(chunk_bytes, target_bytes - allocated)
            block = bytearray(size)
            for offset in range(0, size, page):
                block[offset] = (offset // page) % 251
            chunks.append(block)
            allocated += size
            print(f"[ram] allocated {allocated / GB:.1f} / {target_bytes / GB:.1f} GB", flush=True)
    except MemoryError:
        print(f"[ram] MemoryError after allocating {allocated / GB:.1f} GB", flush=True)

    stride = max(1, int(touch_stride_pages)) * page
    while not stop_event.is_set():
        for block in chunks:
            for offset in range(0, len(block), stride):
                block[offset] = (block[offset] + 1) % 251
                if stop_event.is_set():
                    break
        time.sleep(0.25)


def gpu_worker(
    stop_event,
    gpu_index: int,
    target_memory_gb: Optional[float],
    gpu_headroom_gb: float,
    block_mb: int,
    matrix_size: int,
) -> None:
    try:
        import torch
    except ImportError:
        print("[gpu] PyTorch is not installed. Install a CUDA-enabled torch build for GPU load.", flush=True)
        return

    if not torch.cuda.is_available():
        print("[gpu] torch.cuda.is_available() is false. GPU load was not started.", flush=True)
        return

    torch.cuda.set_device(gpu_index)
    device = torch.device(f"cuda:{gpu_index}")
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / GB
    if target_memory_gb is None:
        target_memory_gb = max(0.0, total_gb - max(0.0, gpu_headroom_gb))
    target_memory_gb = min(max(0.0, target_memory_gb), max(0.0, total_gb - 0.5))

    dtype = torch.float16
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    block_elems = max(1, int(block_mb * MB // bytes_per_elem))
    reserve = []
    allocated = 0
    target_bytes = int(target_memory_gb * GB)

    print(
        f"[gpu] device={props.name}, total={total_gb:.1f} GB, reserve_target={target_memory_gb:.1f} GB",
        flush=True,
    )

    try:
        while allocated < target_bytes and not stop_event.is_set():
            elems = min(block_elems, max(1, (target_bytes - allocated) // bytes_per_elem))
            tensor = torch.empty(elems, dtype=dtype, device=device)
            tensor.fill_(1.0)
            reserve.append(tensor)
            allocated += elems * bytes_per_elem
            print(f"[gpu] reserved {allocated / GB:.1f} / {target_bytes / GB:.1f} GB", flush=True)
    except RuntimeError as exc:
        print(f"[gpu] stopped reserving memory at {allocated / GB:.1f} GB: {exc}", flush=True)

    size = max(1024, int(matrix_size))
    try:
        a = torch.randn((size, size), dtype=dtype, device=device)
        b = torch.randn((size, size), dtype=dtype, device=device)
    except RuntimeError as exc:
        print(f"[gpu] could not allocate compute matrices size={size}: {exc}", flush=True)
        return

    iterations = 0
    while not stop_event.is_set():
        c = a @ b
        a, b = b, c
        iterations += 1
        if iterations % 8 == 0:
            torch.cuda.synchronize(device)
            a = a / a.abs().mean().clamp_min(1e-3)
            b = b / b.abs().mean().clamp_min(1e-3)
            print(f"[gpu] matmul iterations={iterations}", flush=True)

    torch.cuda.synchronize(device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concurrent CPU, RAM, and CUDA GPU stress load for fan testing.",
    )
    parser.add_argument("--duration-sec", type=int, default=300, help="Run time before automatic stop.")
    parser.add_argument("--monitor-interval-sec", type=float, default=5.0)

    parser.add_argument("--no-cpu", action="store_true", help="Disable CPU load.")
    parser.add_argument("--cpu-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--cpu-load-percent", type=float, default=100.0)

    parser.add_argument("--no-ram", action="store_true", help="Disable RAM allocation and touch loop.")
    parser.add_argument("--system-ram-gb", type=float, default=None)
    parser.add_argument("--ram-chunk-mb", type=int, default=512)
    parser.add_argument("--ram-touch-stride-pages", type=int, default=64)

    parser.add_argument("--no-gpu", action="store_true", help="Disable CUDA GPU load.")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-memory-gb", type=float, default=None)
    parser.add_argument("--gpu-headroom-gb", type=float, default=3.0)
    parser.add_argument("--gpu-memory-block-mb", type=int, default=512)
    parser.add_argument("--gpu-matrix-size", type=int, default=12288)
    parser.add_argument("--gpu-temp-limit-c", type=float, default=88.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration_sec <= 0:
        raise SystemExit("--duration-sec must be positive")

    ram_target = args.system_ram_gb if args.system_ram_gb is not None else _default_ram_target_gb()
    cpu_workers = max(0, int(args.cpu_workers))

    print("Starting stress load. Press Ctrl+C to stop.", flush=True)
    print(
        "Config: "
        f"duration={args.duration_sec}s, "
        f"cpu={'off' if args.no_cpu else str(cpu_workers) + ' workers'}, "
        f"ram={'off' if args.no_ram else f'{ram_target:.1f} GB'}, "
        f"gpu={'off' if args.no_gpu else 'cuda:' + str(args.gpu_index)}",
        flush=True,
    )

    ctx = get_context("spawn")
    stop_event = ctx.Event()
    processes = []

    if not args.no_cpu:
        for idx in range(cpu_workers):
            process = ctx.Process(target=cpu_worker, args=(stop_event, idx, args.cpu_load_percent))
            process.start()
            processes.append(process)

    if not args.no_ram and ram_target > 0:
        process = ctx.Process(
            target=ram_worker,
            args=(stop_event, ram_target, args.ram_chunk_mb, args.ram_touch_stride_pages),
        )
        process.start()
        processes.append(process)

    if not args.no_gpu:
        process = ctx.Process(
            target=gpu_worker,
            args=(
                stop_event,
                args.gpu_index,
                args.gpu_memory_gb,
                args.gpu_headroom_gb,
                args.gpu_memory_block_mb,
                args.gpu_matrix_size,
            ),
        )
        process.start()
        processes.append(process)

    started = time.time()
    exit_code = 0
    try:
        while True:
            remaining = args.duration_sec - (time.time() - started)
            if remaining <= 0:
                break
            time.sleep(max(0.1, min(args.monitor_interval_sec, remaining)))
            elapsed = time.time() - started
            stats = _query_nvidia_smi(args.gpu_index) if not args.no_gpu else None
            if stats:
                print(
                    "[monitor] "
                    f"elapsed={elapsed:.0f}s "
                    f"gpu_temp={stats['temp_c']}C "
                    f"gpu_util={stats['util_pct']}% "
                    f"gpu_mem={stats['mem_used_mb']}/{stats['mem_total_mb']} MB "
                    f"gpu_power={stats['power_w']}W",
                    flush=True,
                )
                try:
                    if float(stats["temp_c"]) >= args.gpu_temp_limit_c:
                        print(
                            f"[monitor] GPU temperature limit {args.gpu_temp_limit_c:.1f}C reached; stopping.",
                            flush=True,
                        )
                        break
                except ValueError:
                    pass
            else:
                print(f"[monitor] elapsed={elapsed:.0f}s", flush=True)
    except KeyboardInterrupt:
        print("\nStopping after Ctrl+C.", flush=True)
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for process in processes:
            if process.exitcode not in (0, None):
                exit_code = max(exit_code, 1)
    print("Stress load finished.", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
