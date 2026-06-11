from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.ai.evaluation.config import load_eval_config
from services.ai.evaluation.runner import run_dataset_evaluation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FakeAVCeleb dataset evaluation")
    parser.add_argument("--config", required=True, help="Evaluation INI config path")
    parser.add_argument("--output-root", default=None, help="Override output root")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--run-dir", default=None, help="Existing external run directory for resume")
    parser.add_argument("--type", dest="sample_type", default=None, help="Optional FakeAVCeleb type filter")
    parser.add_argument("--results-root", default=None, help="Optional final metrics export root")
    parser.add_argument("--shard-index", type=int, default=None, help="Zero-based shard index")
    parser.add_argument("--num-shards", type=int, default=None, help="Total shard count")
    parser.add_argument(
        "--overfit-thresholds",
        action="store_true",
        help="Opt in to diagnostic in-sample threshold calibration; not valid for benchmark reporting",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    output_root = Path(args.output_root) if args.output_root else None
    run_dir = Path(args.run_dir) if args.run_dir else None
    results_root = Path(args.results_root) if args.results_root else None
    config = load_eval_config(config_path, output_root_override=output_root)
    result = run_dataset_evaluation(
        config,
        limit=args.limit,
        run_dir=run_dir,
        sample_type=args.sample_type,
        results_root=results_root,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        overfit_thresholds=args.overfit_thresholds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
