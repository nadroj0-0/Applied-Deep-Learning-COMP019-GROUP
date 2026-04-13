
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from m5_tft_common_v5 import (
    DEFAULT_DATA_ROOT,
    FEATURE_SETS,
    TrainConfig,
    run_single_experiment,
)


DEFAULT_SCREENING_SETS: List[str] = [
    "full_safe",
    "full_safe_plus_compact_fe",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fast representative feature screen for TFT on the M5 export."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("compact_fe_compare_runs"))
    parser.add_argument("--feature-sets", type=str, default=",".join(DEFAULT_SCREENING_SETS))
    parser.add_argument("--fold", type=str, default="fold_2")
    parser.add_argument("--max-series", type=int, default=4096)
    parser.add_argument("--series-ids-path", type=Path, default=None)
    parser.add_argument("--sampling-strategy", type=str, default="stratified", choices=["stratified", "random"])
    parser.add_argument("--train-window-count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--attention-head-size", type=int, default=4)
    parser.add_argument("--hidden-continuous-size", type=int, default=8)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--extra-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    feature_sets = [s.strip() for s in args.feature_sets.split(",") if s.strip()]
    unknown = [s for s in feature_sets if s not in FEATURE_SETS]
    if unknown:
        raise KeyError(f"Unknown feature set(s): {unknown}")

    rows = []
    for feature_set in feature_sets:
        config = TrainConfig(
            data_root=args.data_root,
            output_dir=args.output_dir,
            fold_name=args.fold,
            feature_set_name=feature_set,
            seed=args.seed,
            train_window_count=args.train_window_count,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            hidden_size=args.hidden_size,
            attention_head_size=args.attention_head_size,
            hidden_continuous_size=args.hidden_continuous_size,
            lstm_layers=args.lstm_layers,
            learning_rate=args.learning_rate,
            max_series=args.max_series,
            series_ids_path=args.series_ids_path,
            sampling_strategy=args.sampling_strategy,
            extra_workers=args.extra_workers,
        )
        print(json.dumps({"running": feature_set}, indent=2))
        metrics = run_single_experiment(config)
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df = df.sort_values(["mean_pinball", "best_val_loss"], ascending=[True, True]).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "screen_summary.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
