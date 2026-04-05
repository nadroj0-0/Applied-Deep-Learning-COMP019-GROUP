# =============================================================================
# train_gru_models.py — M5 Sales Forecasting — V3 (BaseModel interface)
# COMP0197 Applied Deep Learning
# GenAI Note: Scaffolded with Claude (Anthropic). Verified by authors.
#
# Same structure as train.py but runs through the BaseModel interface.
# Data loading, feature engineering, and windowing happen inside each model
# class via preprocess() → Yen's load_and_split_data() + your windowing.
# Training runs through Experiment.train() exactly as before.
#
# Usage:
#   python train_gru_models.py                          — uses configs/experiment.yml
#   python train_gru_models.py --experiment configs/experiment.yml
#   python train_gru_models.py --run_name my_run        — override run name
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0]))

from utils.config_loader import (
    load_experiment,
    create_run_dir,
    snapshot_configs,
)
from models import (
    BaselineGRUDet,
    BaselineGRUProb,
    BaselineGRUNB,
    BaselineQuantileGRU,
    BaselineWQuantileGRU,
    HierarchicalGRUDet,
    HierarchicalGRUProb,
    HierarchicalGRUNB,
    HierarchicalQuantileGRU,
    HierarchicalWQuantileGRU,
)

PROJECT_DIR     = Path(__file__).resolve().parent
EXPERIMENT_PATH = PROJECT_DIR / "configs" / "experiment.yml"
MODELS_CFG_DIR  = PROJECT_DIR / "configs" / "models"

# =============================================================================
# THE ONLY LINE YOU EDIT — set False before submission
# True  → runs staged_search before training (stratified 3k subset, fast)
# False → trains directly using configs already in yml files
# =============================================================================
SEARCH = False

# =============================================================================
# MODEL REGISTRY — maps experiment.yml model names to their classes
# =============================================================================
MODEL_REGISTRY = {
    "baseline_gru_det":           BaselineGRUDet,
    "baseline_gru_prob":          BaselineGRUProb,
    "baseline_gru_nb":            BaselineGRUNB,
    "baseline_quantile_gru":      BaselineQuantileGRU,
    "baseline_wquantile_gru":     BaselineWQuantileGRU,
    "hierarchical_gru_det":       HierarchicalGRUDet,
    "hierarchical_gru_prob":      HierarchicalGRUProb,
    "hierarchical_gru_nb":        HierarchicalGRUNB,
    "hierarchical_quantile_gru":  HierarchicalQuantileGRU,
    "hierarchical_wquantile_gru": HierarchicalWQuantileGRU,
}


# =============================================================================
# CLI — mirrors train.py exactly, minus GPU data-loading overrides
# (data loading now happens inside each model class, not upfront)
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="M5 Train GRU Models (BaseModel interface)")
    p.add_argument("--experiment", type=str, default=str(EXPERIMENT_PATH),
                   help="Path to experiment.yml (default: configs/experiment.yml)")
    p.add_argument("--run_name",   type=str, default=None,
                   help="Override run_name from experiment.yml")
    p.add_argument("--num_workers", type=int, default=None,
                   help="Override num_workers from experiment.yml")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("main started")
    args = parse_args()
    print("args parsed")

    # ------------------------------------------------------------------
    # 1. Load experiment config
    # ------------------------------------------------------------------
    exp_cfg  = load_experiment(args.experiment)
    run_name = args.run_name or exp_cfg.get("run_name")
    if not run_name:
        raise ValueError("experiment.yml must have a 'run_name' field.")
    models = exp_cfg.get("models", [])
    if not models:
        raise ValueError("experiment.yml 'models' list is empty.")

    do_search = SEARCH or bool(exp_cfg.get("search", {}).get("enabled", False))

    print(f"\n{'='*60}")
    print(f"  M5 TRAINING (BaseModel) — {run_name}")
    print(f"  Models : {models}")
    print(f"  Search : {do_search}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 2. Create run directory and snapshot all configs into it
    # Same as train.py — must happen before any model runs so artefacts
    # go to the right place and configs are frozen at start time.
    # ------------------------------------------------------------------
    run_dir = create_run_dir(PROJECT_DIR, run_name)
    snapshot_configs(
        run_dir        = run_dir,
        experiment_yml = args.experiment,
        model_names    = models,
        models_cfg_dir = MODELS_CFG_DIR,
    )

    # ------------------------------------------------------------------
    # 3. Train each model
    # Each model class handles its own data loading (Yen's pipeline),
    # feature engineering, windowing, and training internally.
    # Search (if enabled) runs on stratified 3k subset via build_dataloaders.
    # Full training runs on Yen's full dataset via load_and_split_data().
    # ------------------------------------------------------------------
    data_dir = exp_cfg.get("train", {}).get("data_dir", "./data")

    results = []
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  TRAINING: {model_name}")
        print(f"{'='*60}")

        if model_name not in MODEL_REGISTRY:
            print(f"  [SKIP] '{model_name}' not in MODEL_REGISTRY — "
                  f"available: {sorted(MODEL_REGISTRY.keys())}")
            continue

        cls = MODEL_REGISTRY[model_name]

        try:
            m = cls(
                data_dir   = data_dir,
                output_dir = "./outputs",
                run_name   = run_name,
                do_search  = do_search,
                num_workers=args.num_workers,
            )
            # run_training_pipeline = load_and_split_data → preprocess → train
            # matches Yen's BaseModel interface exactly
            m.run_training_pipeline()
            results.append(model_name)
            print(f"\n  [DONE] {model_name}")

        except Exception as e:
            print(f"\n  [ERROR] {model_name} failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ALL TRAINING COMPLETE")
    print(f"  Trained  : {len(results)}/{len(models)} models")
    print(f"  Run folder: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
