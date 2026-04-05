# =============================================================================
# models.py — GRU/Hierarchical model subclasses of BaseModel
# COMP0197 Applied Deep Learning
# GenAI Note: Scaffolded with Claude (Anthropic). Verified by authors.
#
# This file satisfies the group's BaseModel interface requirement while
# preserving the complete underlying pipeline: YAML configs, Experiment class,
# build_dataloaders windowing, staged_search, full_train, early stopping,
# run directory management, and all evaluation infrastructure.
#
# BaseModel is a mask. Everything underneath is unchanged.
#
# Usage (interactive / notebook):
#   from models import BaselineGRUDet
#   m = BaselineGRUDet()
#   m.run()                        # full pipeline end-to-end
#
# Usage (existing runner — unchanged):
#   python train.py                # still works exactly as before
#   python search.py               # still works exactly as before
#   python test.py                 # still works exactly as before
# =============================================================================

import sys
import math
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))

from base_model import BaseModel

# ---------------------------------------------------------------------------
# Full pipeline imports — every one of these is non-negotiable.
# The BaseModel mask touches none of this.
# ---------------------------------------------------------------------------
from utils.data import (
    build_dataloaders, get_feature_cols, get_vocab_sizes,
    encode_hierarchy, WindowedM5Dataset, set_seed,
)
from utils.network import (
    build_baseline_gru,
    build_baseline_prob_gru,
    build_baseline_prob_gru_nb,
    build_baseline_quantile_gru,
    build_baseline_wquantile_gru,
    build_hierarchical_gru,
    build_hierarchical_prob_gru,
    build_hierarchical_prob_gru_nb,
    build_hierarchical_quantile_gru,
    build_hierarchical_wquantile_gru,
)
from utils.training_strategies import (
    gru_step,
    prob_gru_step,
    prob_nb_step,
    quantile_gru_step,
    wquantile_gru_step,
)
from utils.experiment import Experiment
from utils.config_loader import (
    load_model_config,
    load_registry,
    load_search_space,
    resolve_registry_entry,
    create_run_dir,
    snapshot_configs,
    write_best_config,
    get_model_run_dir,
    load_experiment,
)
from utils.common import full_train, save_json
from utils.hyperparameter import staged_search

# ---------------------------------------------------------------------------
# Paths — mirror train.py exactly
# ---------------------------------------------------------------------------
PROJECT_DIR    = Path(__file__).resolve().parent
REGISTRY_PATH  = PROJECT_DIR / "configs" / "registry.yml"
MODELS_CFG_DIR = PROJECT_DIR / "configs" / "models"

# ---------------------------------------------------------------------------
# Quantiles — M5 WSPL-aligned 9-value set
# ---------------------------------------------------------------------------
QUANTILES = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]


# =============================================================================
# SHARED PREPROCESSING — called once from every model's preprocess()
#
# Uses Yen's load_and_split_data() for the raw split, then applies your
# feature encoding and windowing on top. Config is read from the model yml
# so it matches exactly what train.py would use.
#
# Sets on the model instance:
#   train_loader, val_loader, test_loader
#   vocab_sizes, feature_index, stats
# =============================================================================

def _preprocess_from_base_model(self_model, model_name: str,
                                 run_name: str, include_weights: bool = False):
    """
    Shared preprocess implementation for all 10 model classes.

    Step 1 — Yen's load_and_split_data() (download, melt, calendar, prices,
              is_available, ffill, 1773/112/56 split, revenue weights, cache).
    Step 2 — Add your feature columns on top of her long-format output:
              hierarchy int encoding, has_event flag.
    Step 3 — Derive split boundaries from her raw splits (no recomputation).
    Step 4 — WindowedM5Dataset + DataLoaders using config from model yml.

    train.py is completely unaffected — it calls build_dataloaders directly.
    """
    # --- read config from yml so we use the same values as train.py ---
    run_dir       = PROJECT_DIR / "runs" / run_name
    run_model_yml = run_dir / "configs" / "models" / f"{model_name}.yml"

    # fall back to project configs if run snapshot doesn't exist yet
    if not run_model_yml.exists():
        run_model_yml = MODELS_CFG_DIR / f"{model_name}.yml"

    model_cfg  = load_model_config(run_model_yml)
    train_cfg  = model_cfg.get("train_config", {})

    feature_set    = str(train_cfg.get("feature_set",    "sales_yen_hierarchy"))
    seq_len        = int(train_cfg.get("seq_len",         28))
    horizon        = int(train_cfg.get("horizon",         28))
    batch_size     = int(train_cfg.get("batch_size",    1024))
    autoregressive = bool(train_cfg.get("autoregressive", True))
    num_workers    = int(train_cfg.get("num_workers",      0))
    seed           = int(train_cfg.get("seed",            42))

    generator, _ = set_seed(seed)

    # --- Step 1: Yen's preprocessing ---
    # Populates self_model.train_raw, val_raw, test_raw, item_weights
    self_model.load_and_split_data()

    # --- Step 2: feature encoding on top of her output ---
    featured = pd.concat([
        self_model.train_raw,
        self_model.val_raw,
        self_model.test_raw,
    ]).reset_index(drop=True)

    # Yen's output has: id, item_id, dept_id, cat_id, store_id, state_id,
    # d, sales, d_num, + full calendar cols + sell_price + is_available
    # We add: state_id_int, store_id_int, cat_id_int, dept_id_int, has_event
    # (these are needed by hierarchy models and the sales_yen_hierarchy feature set)
    include_dow = False   # Yen's pipeline doesn't have dow — add if needed
    featured = encode_hierarchy(featured, include_dow=include_dow)
    featured["has_event"] = (
        (~featured["event_name_1"].astype(str).isin(["none", "nan", "None"]))
        .astype(np.float32)
    )

    # ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(featured["date"]):
        featured["date"] = pd.to_datetime(featured["date"])

    # --- Step 3: split boundaries from her splits directly ---
    val_start_date  = pd.Timestamp(self_model.val_raw["date"].min())
    test_start_date = pd.Timestamp(self_model.test_raw["date"].min())

    # --- Step 4: windowing + DataLoaders ---
    feature_cols  = get_feature_cols(feature_set)
    feature_index = {col: i for i, col in enumerate(feature_cols)}
    series_ids    = featured["id"].drop_duplicates().tolist()

    # revenue weights from Yen's item_weights (already correct — daily rev, last 28 days)
    train_item_weights = None
    if include_weights:
        train_item_weights = (
            self_model.item_weights
            .reindex(series_ids)
            .fillna(0.0)
            .values
            .astype(np.float32)
        )
        total = train_item_weights.sum()
        if total > 0:
            train_item_weights /= total

    shared = dict(
        feature_cols    = feature_cols,
        seq_len         = seq_len,
        horizon         = horizon,
        val_start_date  = val_start_date.to_datetime64(),
        test_start_date = test_start_date.to_datetime64(),
        autoregressive  = autoregressive,
    )

    train_ds = WindowedM5Dataset(
        featured, split="train",
        item_weights=train_item_weights, series_ids=series_ids, **shared)
    val_ds   = WindowedM5Dataset(featured, split="val",  series_ids=series_ids, **shared)
    test_ds  = WindowedM5Dataset(featured, split="test", series_ids=series_ids, **shared)

    _pin = torch.cuda.is_available()
    self_model.train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=generator, pin_memory=_pin,
    )
    self_model.val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin,
    )
    self_model.test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin,
    )

    # set vocab_sizes and feature_index so builders get them
    hierarchy_cols = ["state_id_int", "store_id_int", "cat_id_int", "dept_id_int"]
    if any(c in feature_cols for c in hierarchy_cols):
        self_model.vocab_sizes = get_vocab_sizes(featured)
    else:
        self_model.vocab_sizes = {}

    self_model.feature_index = feature_index
    self_model.stats = None   # Yen's pipeline uses raw counts, no normalisation

    # store train_cfg on instance so train() can read it
    train_cfg["n_features"]    = len(feature_cols)
    train_cfg["vocab_sizes"]   = self_model.vocab_sizes
    train_cfg["feature_index"] = self_model.feature_index
    self_model._train_cfg = train_cfg

    print(f"[preprocess] {model_name} — "
          f"{len(train_ds):,} train / {len(val_ds):,} val / {len(test_ds):,} test windows")


# =============================================================================
# SHARED TRAINING — called from every model's train()
#
# Reads loaders set by preprocess(), creates Experiment, calls exp.train().
# Identical to what _run_full_pipeline does after data loading.
# =============================================================================

def _train_from_preprocess(self_model, model_name: str, run_name: str,
                            builder, step):
    """
    Shared train() implementation. Expects preprocess() to have already run.
    Creates Experiment, injects the loaders preprocess() built, trains.
    """
    run_dir   = PROJECT_DIR / "runs" / run_name
    model_dir = get_model_run_dir(run_dir, model_name)

    train_cfg = getattr(self_model, "_train_cfg", {})

    exp              = Experiment(model_name, train_cfg, model_dir=model_dir)
    exp.train_loader = self_model.train_loader
    exp.val_loader   = self_model.val_loader
    exp.test_dataset = self_model.test_loader
    exp.stats        = self_model.stats
    exp.preloaded    = True

    exp.train(builder, step)

    if exp.stats is not None:
        save_json(exp.stats, model_dir / "normalisation_stats.json")

    self_model.model   = exp.model
    self_model.history = exp.history
    print(f"\n  [DONE] {model_name} — artefacts in {model_dir}")
    return exp


# =============================================================================
# INTERNAL HELPER — used by run() to do the full pipeline in one shot
# (search + preprocess + train), mirrors train.py exactly.
# =============================================================================

def _run_full_pipeline(self_model, model_name: str,
                       model_type: str, is_nb: bool, is_prob: bool,
                       run_name: str = None, do_search: bool = False,
                       include_weights: bool = False):
    if run_name is None:
        run_name = model_name + "_run"

    # --- 1. Run dir + config snapshot ---
    run_dir = create_run_dir(PROJECT_DIR, run_name)
    snapshot_configs(
        run_dir        = run_dir,
        experiment_yml = str(PROJECT_DIR / "configs" / "experiment.yml"),
        model_names    = [model_name],
        models_cfg_dir = MODELS_CFG_DIR,
    )
    run_model_yml = run_dir / "configs" / "models" / f"{model_name}.yml"

    # --- 2. Optional search (stratified subset, your pipeline — fast) ---
    if do_search:
        registry     = load_registry(REGISTRY_PATH)
        resolved     = resolve_registry_entry(registry[model_name])
        builder      = resolved["builder"]
        step         = resolved["training_step"]
        search_space = load_search_space(run_model_yml)

        if search_space:
            model_cfg = load_model_config(run_model_yml)
            train_cfg = model_cfg.get("train_config", {})
            train_cfg["n_features"] = len(get_feature_cols(
                str(train_cfg.get("feature_set", "sales_yen_hierarchy"))))

            search_kwargs = dict(
                data_dir        = self_model.data_dir,
                seq_len         = int(train_cfg.get("seq_len",    28)),
                horizon         = int(train_cfg.get("horizon",    28)),
                batch_size      = int(train_cfg.get("batch_size", 1024)),
                top_k_series    = 1000,
                feature_set     = str(train_cfg.get("feature_set", "sales_yen_hierarchy")),
                autoregressive  = bool(train_cfg.get("autoregressive", True)),
                use_normalise   = bool(train_cfg.get("use_normalise", False)),
                sampling        = "stratified",
                num_workers     = int(train_cfg.get("num_workers", 0)),
                seed            = int(train_cfg.get("seed", 42)),
                split_protocol  = "yen_v1",
                weight_protocol = "yen_v1",
            )
            stl, svl, _, s_stats, s_vocab, s_fidx = build_dataloaders(**search_kwargs)
            train_cfg["vocab_sizes"]   = s_vocab
            train_cfg["feature_index"] = s_fidx

            model_dir        = get_model_run_dir(run_dir, model_name)
            exp              = Experiment(model_name, train_cfg, model_dir=model_dir)
            exp.train_loader = stl
            exp.val_loader   = svl
            exp.stats        = s_stats
            exp.preloaded    = True

            schedule = [
                {"epochs": 10, "keep": 5},
                {"epochs": 10, "keep": 3},
                {"epochs": 20, "keep": 1},
            ]
            exp.search(search_space=search_space, builder=builder,
                       training_step=step, schedule=schedule, initial_models=10)
            write_best_config(run_model_yml, exp.cfg)
            print(f"\n[models] Search complete for {model_name}")

    # --- 3. Preprocess using Yen's pipeline + your windowing ---
    _preprocess_from_base_model(
        self_model, model_name, run_name,
        include_weights=include_weights,
    )

    # --- 4. Resolve builder + step, inject train_cfg extras ---
    registry = load_registry(REGISTRY_PATH)
    resolved = resolve_registry_entry(registry[model_name])
    builder  = resolved["builder"]
    step     = resolved["training_step"]

    # --- 5. Train ---
    return _train_from_preprocess(
        self_model, model_name, run_name, builder, step)


# =============================================================================
# BASELINE MODELS
# =============================================================================

class BaselineGRUDet(BaseModel):
    """Vanilla GRU — MSE loss (deterministic)."""
    model_name = "baseline_gru_det"

    def __init__(self, data_dir="./data", output_dir="./outputs",
                 run_name=None, do_search=False):
        super().__init__(data_dir=data_dir, output_dir=output_dir)
        self._run_name  = run_name or self.model_name + "_run"
        self._do_search = do_search
        self._exp       = None

    def preprocess(self):
        """Yen's preprocessing → your feature encoding → windowed DataLoaders."""
        _preprocess_from_base_model(self, self.model_name, self._run_name,
                                    include_weights=False)

    def train(self):
        """Create Experiment from preprocess() outputs and train."""
        registry = load_registry(REGISTRY_PATH)
        resolved = resolve_registry_entry(registry[self.model_name])
        _train_from_preprocess(self, self.model_name, self._run_name,
                               resolved["builder"], resolved["training_step"])

    def predict(self):
        raise NotImplementedError(
            f"{self.model_name}.predict() not yet implemented. "
            "Use test.py for evaluation."
        )

    def run(self):
        """Full pipeline: search (optional) → preprocess → train."""
        self._exp = _run_full_pipeline(
            self, model_name=self.model_name,
            model_type="baseline_gru", is_nb=False, is_prob=False,
            run_name=self._run_name, do_search=self._do_search,
            include_weights=False,
        )


class BaselineGRUProb(BaseModel):
    """Vanilla GRU — Gaussian NLL loss (probabilistic)."""
    model_name = "baseline_gru_prob"

    def __init__(self, data_dir="./data", output_dir="./outputs",
                 run_name=None, do_search=False):
        super().__init__(data_dir=data_dir, output_dir=output_dir)
        self._run_name  = run_name or self.model_name + "_run"
        self._do_search = do_search

    def preprocess(self):
        _preprocess_from_base_model(self, self.model_name, self._run_name,
                                    include_weights=False)

    def train(self):
        registry = load_registry(REGISTRY_PATH)
        resolved = resolve_registry_entry(registry[self.model_name])
        _train_from_preprocess(self, self.model_name, self._run_name,
                               resolved["builder"], resolved["training_step"])

    def predict(self):
        raise NotImplementedError(
            f"{self.model_name}.predict() not yet implemented. "
            "Use test.py for evaluation."
        )

    def run(self):
        _run_full_pipeline(
            self, model_name=self.model_name,
            model_type="baseline_gru_prob", is_nb=False, is_prob=True,
            run_name=self._run_name, do_search=self._do_search,
            include_weights=False,
        )


class BaselineGRUNB(BaseModel):
    """Vanilla GRU — Negative Binomial NLL loss."""
    model_name = "baseline_gru_nb"

    def __init__(self, data_dir="./data", output_dir="./outputs",
                 run_name=None, do_search=False):
        super().__init__(data_dir=data_dir, output_dir=output_dir)
        self._run_name  = run_name or self.model_name + "_run"
        self._do_search = do_search

    def preprocess(self):
        _preprocess_from_base_model(self, self.model_name, self._run_name,
                                    include_weights=False)

    def train(self):
        registry = load_registry(REGISTRY_PATH)
        resolved = resolve_registry_entry(registry[self.model_name])
        _train_from_preprocess(self, self.model_name, self._run_name,
                               resolved["builder"], resolved["training_step"])

    def predict(self):
        raise NotImplementedError(
            f"{self.model_name}.predict() not yet implemented. "
            "Use test.py for evaluation."
        )

    def run(self):
        _run_full_pipeline(
            self, model_name=self.model_name,
            model_type="baseline_gru_nb", is_nb=True, is_prob=True,
            run_name=self._run_name, do_search=self._do_search,
            include_weights=False,
        )


class BaselineQuantileGRU(BaseModel):
    """Vanilla GRU — unweighted pinball loss, 9 quantiles."""
    model_name = "baseline_quantile_gru"

    def __init__(self, data_dir="./data", output_dir="./outputs",
                 run_name=None, do_search=False):
        super().__init__(data_dir=data_dir, output_dir=output_dir)
        self._run_name  = run_name or self.model_name + "_run"
        self._do_search = do_search

    def preprocess(self):
        _preprocess_from_base_model(self, self.model_name, self._run_name,
                                    include_weights=False)

    def train(self):
        registry = load_registry(REGISTRY_PATH)
        resolved = resolve_registry_entry(registry[self.model_name])
        _train_from_preprocess(self, self.model_name, self._run_name,
                               resolved["builder"], resolved["training_step"])

    def predict(self):
        raise NotImplementedError(
            f"{self.model_name}.predict() not yet implemented. "
            "Use test.py for evaluation."
        )

    def run(self):
        _run_full_pipeline(
            self, model_name=self.model_name,
            model_type="baseline_quantile_gru", is_nb=False, is_prob=False,
            run_name=self._run_name, do_search=self._do_search,
            include_weights=False,
        )


class BaselineWQuantileGRU(BaseModel):
    """Vanilla GRU — revenue-weighted pinball loss, 9 quantiles."""
    model_name = "baseline_wquantile_gru"

    def __init__(self, data_dir="./data", output_dir="./outputs",
                 run_name=None, do_search=False):
        super().__init__(data_dir=data_dir, output_dir=output_dir)
        self._run_name  = run_name or self.model_name + "_run"
        self._do_search = do_search

    def preprocess(self):
        # include_weights=True — train loader returns (x, y, weight) triples
        _preprocess_from_base_model(self, self.model_name, self._run_name,
                                    include_weights=True)

    def train(self):
        registry = load_registry(REGISTRY_PATH)
        resolved = resolve_registry_entry(registry[self.model_name])
        _train_from_preprocess(self, self.model_name, self._run_name,
                               resolved["builder"], resolved["training_step"])

    def predict(self):
        raise NotImplementedError(
            f"{self.model_name}.predict() not yet implemented. "
            "Use test.py for evaluation."
        )

    def run(self):
        _run_full_pipeline(
            self, model_name=self.model_name,
            model_type="baseline_wquantile_gru", is_nb=False, is_prob=False,
            run_name=self._run_name, do_search=self._do_search,
            include_weights=True,
        )


# =============================================================================
# HIERARCHICAL MODELS
# =============================================================================

class HierarchicalGRUDet(BaseModel):
    """GRU with learned hierarchy embeddings — MSE loss (deterministic)."""
    model_name = "hierarchical_gru_det"

    def __init__(self, data_dir="./data", output_dir="./outputs",
                 run_name=None, do_search=False):
        super().__init__(data_dir=data_dir, output_dir=output_dir)
        self._run_name  = run_name or self.model_name + "_run"
        self._do_search = do_search

    def preprocess(self):
        _preprocess_from_base_model(self, self.model_name, self._run_name,
                                    include_weights=False)

    def train(self):
        registry = load_registry(REGISTRY_PATH)
        resolved = resolve_registry_entry(registry[self.model_name])
        _train_from_preprocess(self, self.model_name, self._run_name,
                               resolved["builder"], resolved["training_step"])

    def predict(self):
        raise NotImplementedError(
            f"{self.model_name}.predict() not yet implemented. "
            "Use test.py for evaluation."
        )

    def run(self):
        _run_full_pipeline(
            self, model_name=self.model_name,
            model_type="hierarchical_gru", is_nb=False, is_prob=False,
            run_name=self._run_name, do_search=self._do_search,
            include_weights=False,
        )


class HierarchicalGRUProb(BaseModel):
    """GRU with learned hierarchy embeddings — Gaussian NLL loss."""
    model_name = "hierarchical_gru_prob"

    def __init__(self, data_dir="./data", output_dir="./outputs",
                 run_name=None, do_search=False):
        super().__init__(data_dir=data_dir, output_dir=output_dir)
        self._run_name  = run_name or self.model_name + "_run"
        self._do_search = do_search

    def preprocess(self):
        _preprocess_from_base_model(self, self.model_name, self._run_name,
                                    include_weights=False)

    def train(self):
        registry = load_registry(REGISTRY_PATH)
        resolved = resolve_registry_entry(registry[self.model_name])
        _train_from_preprocess(self, self.model_name, self._run_name,
                               resolved["builder"], resolved["training_step"])

    def predict(self):
        raise NotImplementedError(
            f"{self.model_name}.predict() not yet implemented. "
            "Use test.py for evaluation."
        )

    def run(self):
        _run_full_pipeline(
            self, model_name=self.model_name,
            model_type="hierarchical_gru_prob", is_nb=False, is_prob=True,
            run_name=self._run_name, do_search=self._do_search,
            include_weights=False,
        )


class HierarchicalGRUNB(BaseModel):
    """GRU with learned hierarchy embeddings — Negative Binomial NLL loss."""
    model_name = "hierarchical_gru_nb"

    def __init__(self, data_dir="./data", output_dir="./outputs",
                 run_name=None, do_search=False):
        super().__init__(data_dir=data_dir, output_dir=output_dir)
        self._run_name  = run_name or self.model_name + "_run"
        self._do_search = do_search

    def preprocess(self):
        _preprocess_from_base_model(self, self.model_name, self._run_name,
                                    include_weights=False)

    def train(self):
        registry = load_registry(REGISTRY_PATH)
        resolved = resolve_registry_entry(registry[self.model_name])
        _train_from_preprocess(self, self.model_name, self._run_name,
                               resolved["builder"], resolved["training_step"])

    def predict(self):
        raise NotImplementedError(
            f"{self.model_name}.predict() not yet implemented. "
            "Use test.py for evaluation."
        )

    def run(self):
        _run_full_pipeline(
            self, model_name=self.model_name,
            model_type="hierarchical_gru_nb", is_nb=True, is_prob=True,
            run_name=self._run_name, do_search=self._do_search,
            include_weights=False,
        )


class HierarchicalQuantileGRU(BaseModel):
    """GRU with learned hierarchy embeddings — unweighted pinball loss."""
    model_name = "hierarchical_quantile_gru"

    def __init__(self, data_dir="./data", output_dir="./outputs",
                 run_name=None, do_search=False):
        super().__init__(data_dir=data_dir, output_dir=output_dir)
        self._run_name  = run_name or self.model_name + "_run"
        self._do_search = do_search

    def preprocess(self):
        _preprocess_from_base_model(self, self.model_name, self._run_name,
                                    include_weights=False)

    def train(self):
        registry = load_registry(REGISTRY_PATH)
        resolved = resolve_registry_entry(registry[self.model_name])
        _train_from_preprocess(self, self.model_name, self._run_name,
                               resolved["builder"], resolved["training_step"])

    def predict(self):
        raise NotImplementedError(
            f"{self.model_name}.predict() not yet implemented. "
            "Use test.py for evaluation."
        )

    def run(self):
        _run_full_pipeline(
            self, model_name=self.model_name,
            model_type="hierarchical_quantile_gru", is_nb=False, is_prob=False,
            run_name=self._run_name, do_search=self._do_search,
            include_weights=False,
        )


class HierarchicalWQuantileGRU(BaseModel):
    """GRU with learned hierarchy embeddings — revenue-weighted pinball loss."""
    model_name = "hierarchical_wquantile_gru"

    def __init__(self, data_dir="./data", output_dir="./outputs",
                 run_name=None, do_search=False):
        super().__init__(data_dir=data_dir, output_dir=output_dir)
        self._run_name  = run_name or self.model_name + "_run"
        self._do_search = do_search

    def preprocess(self):
        _preprocess_from_base_model(self, self.model_name, self._run_name,
                                    include_weights=True)

    def train(self):
        registry = load_registry(REGISTRY_PATH)
        resolved = resolve_registry_entry(registry[self.model_name])
        _train_from_preprocess(self, self.model_name, self._run_name,
                               resolved["builder"], resolved["training_step"])

    def predict(self):
        raise NotImplementedError(
            f"{self.model_name}.predict() not yet implemented. "
            "Use test.py for evaluation."
        )

    def run(self):
        _run_full_pipeline(
            self, model_name=self.model_name,
            model_type="hierarchical_wquantile_gru", is_nb=False, is_prob=False,
            run_name=self._run_name, do_search=self._do_search,
            include_weights=True,
        )
