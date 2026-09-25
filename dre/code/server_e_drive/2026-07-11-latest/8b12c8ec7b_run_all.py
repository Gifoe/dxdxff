"""Single strict entry point for the frozen cross-seizure sensitivity study."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

from .audit_cross_seizure_inputs import audit
from .run_cross_seizure_inference import run
from .summarize_results import summarize
from .validate_all_seizure_reproduction import validate


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Inference-only Task 1 cross-seizure sensitivity")
    for name in ("prq_root", "bcr_root", "cache_path", "protocol_root", "reference_root", "output_root"):
        value.add_argument("--" + name.replace("_", "-"), dest=name, required=True)
    value.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62]); value.add_argument("--repeats", type=int, default=10)
    value.add_argument("--device", default="cuda"); value.add_argument("--amp", action="store_true"); value.add_argument("--strict", action="store_true"); value.add_argument("--resume", action="store_true")
    value.add_argument("--audit-only", action="store_true"); value.add_argument("--validate-all-only", action="store_true"); value.add_argument("--run-one", action="store_true"); value.add_argument("--run-two", action="store_true"); value.add_argument("--run-all-seizures", action="store_true")
    value.add_argument("--summarize-only", action="store_true"); value.add_argument("--build-figures-only", action="store_true")
    value.add_argument("--bootstrap-repeats", type=int, default=10_000)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.strict and args.amp:
        raise ValueError(
            "Strict cross-seizure reproduction requires FP32 inference; "
            "AMP changes saved-model numerics and overflows the frozen -1e9 mask."
        )
    audit(args)
    if args.audit_only:
        return
    if args.summarize_only or args.build_figures_only:
        summarize(args.output_root, bootstrap_repeats=args.bootstrap_repeats); return
    requested = any((args.run_one, args.run_two, args.run_all_seizures, args.validate_all_only))
    # Default formal sequence runs all -> validation gate -> one/two -> summary.
    if args.run_all_seizures or not requested:
        run(args, mode="all")
    validate(args)
    if args.validate_all_only:
        return
    if args.run_one or not requested:
        run(args, mode="one")
    if args.run_two or not requested:
        run(args, mode="two")
    summarize(args.output_root, bootstrap_repeats=args.bootstrap_repeats)


if __name__ == "__main__":
    main()
