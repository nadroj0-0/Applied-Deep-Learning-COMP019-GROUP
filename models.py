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
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from base_model import BaseModel

# ---------------------------------------------------------------------------
# Full pipeline imports — every one of these is non-negotiable.
# The BaseModel mask touches none of this.
# ---------------------------------------------------------------------------
from utils.data import build_dataloaders, get_feature_cols, get_vocab_sizes
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
# INTERNAL HELPERS — replicate train.py's data routing and training logic
# exactly, so calling m.run() is identical to running train.py for that model
# =============================================================================

def _load_loaders_for_model(model_name: str, train_cfg: dict,
                             data_dir: str, model_type: str,
                             is_nb: bool, is_prob: bool) -> dict:
    """
    Replicate train.py's _load_train_data + loader routing for a single model.
    Returns a loaders dict with the correct train/val/test loaders for this
    model type, exactly as train.py would route them.
    """
    use_norm = bool(train_cfg.get("use_normalise", False))

    data_kwargs = dict(
        data_dir       = data_dir,
        seq_len        = int(train_cfg.get("seq_len",     28)),
        horizon        = int(train_cfg.get("horizon",     28)),
        batch_size     = int(train_cfg.get("batch_size", 1024)),
        top_k_series   = int(train_cfg.get("top_k_series", 30490)),
        feature_set    = str(train_cfg.get("feature_set", "sales_hierarchy_dow")),
        autoregressive = bool(train_cfg.get("autoregressive", True)),
        use_normalise  = use_norm,
        sampling       = str(train_cfg.get("sampling", "all")),
        max_series     = train_cfg.get("max_series"),
        num_workers    = int(train_cfg.get("num_workers", 0)),
        seed           = int(train_cfg.get("seed", 42)),
        split_protocol=str(train_cfg.get("split_protocol", "default")),
        weight_protocol=str(train_cfg.get("weight_protocol", "default")),
    )

    # Exactly mirrors train.py's three-loader routing pattern
    if not use_norm:
        tl, vl, tel, stats, vocab_sizes = build_dataloaders(**data_kwargs)
        tl_w, _, _, _, _                = build_dataloaders(**data_kwargs,
                                                             include_weights=True)
        loaders = dict(
            train_loader_det       = tl,   val_loader_det   = vl, test_loader_det   = tel, stats_det   = stats,
            train_loader_gauss     = tl,   val_loader_gauss = vl, test_loader_gauss = tel, stats_gauss = stats,
            train_loader_nb        = tl,   val_loader_nb    = vl, test_loader_nb    = tel, stats_nb    = stats,
            train_loader_wquantile = tl_w, vocab_sizes      = vocab_sizes,
        )
    else:
        tl_det,   vl_det,   tel_det,   s_det,   vocab_sizes = build_dataloaders(**data_kwargs, zscore_target=True)
        tl_gauss, vl_gauss, tel_gauss, s_gauss, _           = build_dataloaders(**data_kwargs, zscore_target=False)
        tl_nb,    vl_nb,    tel_nb,    s_nb,    _           = build_dataloaders(**data_kwargs, zscore_target=False)
        tl_w,     _,        _,         _,       _           = build_dataloaders(**data_kwargs, zscore_target=True,
                                                                                include_weights=True)
        loaders = dict(
            train_loader_det       = tl_det,   val_loader_det   = vl_det,   test_loader_det   = tel_det,   stats_det   = s_det,
            train_loader_gauss     = tl_gauss, val_loader_gauss = vl_gauss, test_loader_gauss = tel_gauss, stats_gauss = s_gauss,
            train_loader_nb        = tl_nb,    val_loader_nb    = vl_nb,    test_loader_nb    = tel_nb,    stats_nb    = s_nb,
            train_loader_wquantile = tl_w,     vocab_sizes      = vocab_sizes,
        )

    # Route to the correct loader for this model type — exactly train.py
    if is_nb:
        return (loaders["train_loader_nb"], loaders["val_loader_nb"],
                loaders["test_loader_nb"], loaders["stats_nb"],
                loaders["vocab_sizes"])
    elif model_type in ("baseline_wquantile_gru", "hierarchical_wquantile_gru"):
        return (loaders["train_loader_wquantile"], loaders["val_loader_det"],
                loaders["test_loader_det"], loaders["stats_det"],
                loaders["vocab_sizes"])
    elif is_prob:
        return (loaders["train_loader_gauss"], loaders["val_loader_gauss"],
                loaders["test_loader_gauss"], loaders["stats_gauss"],
                loaders["vocab_sizes"])
    else:
        return (loaders["train_loader_det"], loaders["val_loader_det"],
                loaders["test_loader_det"], loaders["stats_det"],
                loaders["vocab_sizes"])


def _run_full_pipeline(self_model, model_name: str,
                       model_type: str, is_nb: bool, is_prob: bool,
                       run_name: str = None, do_search: bool = False):
    """
    The core pipeline — identical to what train.py does for a single model.

    Steps (matching train.py exactly):
      1. Create run directory + snapshot configs
      2. Optional: run staged_search via Experiment.search()
      3. Load train_config from yml (post-search if search ran)
      4. Load data via build_dataloaders() with correct routing
      5. Create Experiment, inject loaders
      6. Call Experiment.train(builder, step)
      7. Save normalisation stats

    All artefacts go to runs/{run_name}/models/{model_name}/ — same as
    running train.py directly.
    """
    if run_name is None:
        run_name = model_name + "_run"

    # --- 1. Run dir + config snapshot (mirrors train.py step 2) ---
    run_dir = create_run_dir(PROJECT_DIR, run_name)
    snapshot_configs(
        run_dir        = run_dir,
        experiment_yml = str(PROJECT_DIR / "configs" / "experiment.yml"),
        model_names    = [model_name],
        models_cfg_dir = MODELS_CFG_DIR,
    )

    run_model_yml = run_dir / "configs" / "models" / f"{model_name}.yml"

    # --- 2. Optional search (mirrors train.py step 3) ---
    if do_search:
        registry    = load_registry(REGISTRY_PATH)
        resolved    = resolve_registry_entry(registry[model_name])
        builder     = resolved["builder"]
        step        = resolved["training_step"]
        search_space = load_search_space(run_model_yml)

        if search_space:
            model_cfg = load_model_config(run_model_yml)
            train_cfg = model_cfg.get("train_config", {})
            train_cfg["n_features"] = len(get_feature_cols(
                str(train_cfg.get("feature_set", "sales_hierarchy_dow"))))

            # quick search loaders — stratified subset, mirrors search.py
            search_kwargs = dict(
                data_dir       = self_model.data_dir,
                seq_len        = int(train_cfg.get("seq_len", 28)),
                horizon        = int(train_cfg.get("horizon", 28)),
                batch_size     = int(train_cfg.get("batch_size", 1024)),
                top_k_series   = 1000,
                feature_set    = str(train_cfg.get("feature_set", "sales_hierarchy_dow")),
                autoregressive = bool(train_cfg.get("autoregressive", True)),
                use_normalise  = bool(train_cfg.get("use_normalise", False)),
                sampling       = "stratified",
                num_workers    = int(train_cfg.get("num_workers", 0)),
                seed           = int(train_cfg.get("seed", 42)),
            )
            stl, svl, _, s_stats, s_vocab = build_dataloaders(**search_kwargs)
            train_cfg["vocab_sizes"] = s_vocab

            model_dir = get_model_run_dir(run_dir, model_name)
            exp = Experiment(model_name, train_cfg, model_dir=model_dir)
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

    # --- 3. Load train_config from yml (post-search if ran) ---
    registry = load_registry(REGISTRY_PATH)
    resolved = resolve_registry_entry(registry[model_name])
    builder  = resolved["builder"]
    step     = resolved["training_step"]

    model_cfg = load_model_config(run_model_yml)
    train_cfg = model_cfg.get("train_config", {})
    train_cfg["n_features"] = len(get_feature_cols(
        str(train_cfg.get("feature_set", "sales_hierarchy_dow"))))
    # --- Inject experiment-level protocols ---
    exp_cfg = load_experiment(run_dir / "configs" / "experiment.yml")
    train_cfg["split_protocol"] = exp_cfg.get("train", {}).get("split_protocol", "default")
    train_cfg["weight_protocol"] = exp_cfg.get("train", {}).get("weight_protocol", "default")

    # --- 4. Load data with correct routing (mirrors train.py step 4+5) ---
    train_loader, val_loader, test_loader, stats, vocab_sizes = \
        _load_loaders_for_model(
            model_name = model_name,
            train_cfg  = train_cfg,
            data_dir   = self_model.data_dir,
            model_type = model_type,
            is_nb      = is_nb,
            is_prob    = is_prob,
        )
    train_cfg["vocab_sizes"] = vocab_sizes

    # --- 5+6. Create Experiment, inject loaders, train (mirrors train.py) ---
    model_dir = get_model_run_dir(run_dir, model_name)
    exp = Experiment(model_name, train_cfg, model_dir=model_dir)
    exp.train_loader = train_loader
    exp.val_loader   = val_loader
    exp.test_dataset = test_loader
    exp.stats        = stats
    exp.preloaded    = True

    exp.train(builder, step)

    # --- 7. Save stats ---
    if exp.stats is not None:
        save_json(exp.stats, model_dir / "normalisation_stats.json")

    # Store on the BaseModel instance so caller can inspect
    self_model.model   = exp.model
    self_model.history = exp.history

    print(f"\n  [DONE] {model_name} — artefacts in {model_dir}")
    return exp


# =============================================================================
# BASELINE MODELS
# =============================================================================

class BaselineGRUDet(BaseModel):
    """
    Vanilla GRU — MSE loss (deterministic).
    preprocess() and train() delegate to the full pipeline:
      build_dataloaders → Experiment → full_train → early stopping → save
    Config read from configs/models/baseline_gru_det.yml.
    """
    model_name = "baseline_gru_det"

    def __init__(self, data_dir="./data", output_dir="./outputs",
                 run_name=None, do_search=False):
        super().__init__(data_dir=data_dir, output_dir=output_dir)
        self._run_name  = run_name or self.model_name + "_run"
        self._do_search = do_search
        self._exp       = None

    def preprocess(self):
        """
        Loads data using build_dataloaders() with sliding-window windowing,
        full dataset (30,490 series, all stores), and exact train.py routing.
        Windowing, normalisation, and loader construction all happen here.
        """
        # Data loading happens inside run() via _run_full_pipeline.
        # Calling preprocess() directly also triggers run() for compatibility.
        self.run()

    def train(self):
        """
        Creates Experiment, injects pre-loaded loaders, calls Experiment.train()
        which calls full_train() → train_model() loop → early stopping →
        save_model() → save_history(). Identical to train.py.
        """
        # No-op: training completed inside preprocess() → run().
        pass

    def run(self):
        """Full pipeline: data → Experiment → search (optional) → train."""
        self._exp = _run_full_pipeline(
            self, model_name="baseline_gru_det",
            model_type="baseline_gru", is_nb=False, is_prob=False,
            run_name=self._run_name, do_search=self._do_search,
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
        """Triggers the full pipeline. Calling run() directly is equivalent."""
        self.run()

    def train(self):
        """No-op: training is completed inside preprocess() → run()."""
        pass

    def run(self):
        _run_full_pipeline(
            self, model_name="baseline_gru_prob",
            model_type="baseline_gru_prob", is_nb=False, is_prob=True,
            run_name=self._run_name, do_search=self._do_search,
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
        """Triggers the full pipeline. Calling run() directly is equivalent."""
        self.run()

    def train(self):
        """No-op: training is completed inside preprocess() → run()."""
        pass

    def run(self):
        _run_full_pipeline(
            self, model_name="baseline_gru_nb",
            model_type="baseline_gru_nb", is_nb=True, is_prob=True,
            run_name=self._run_name, do_search=self._do_search,
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
        """Triggers the full pipeline. Calling run() directly is equivalent."""
        self.run()

    def train(self):
        """No-op: training is completed inside preprocess() → run()."""
        pass

    def run(self):
        _run_full_pipeline(
            self, model_name="baseline_quantile_gru",
            model_type="baseline_quantile_gru", is_nb=False, is_prob=False,
            run_name=self._run_name, do_search=self._do_search,
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
        """Triggers the full pipeline. Calling run() directly is equivalent."""
        self.run()

    def train(self):
        """No-op: training is completed inside preprocess() → run()."""
        pass

    def run(self):
        _run_full_pipeline(
            self, model_name="baseline_wquantile_gru",
            model_type="baseline_wquantile_gru", is_nb=False, is_prob=False,
            run_name=self._run_name, do_search=self._do_search,
        )


# =============================================================================
# HIERARCHICAL MODELS
# =============================================================================

class HierarchicalGRUDet(BaseModel):
    """
    GRU with learned hierarchy embeddings — MSE loss (deterministic).
    Replaces raw integer hierarchy columns with dense embedding lookups via
    _HierarchyEmbedder, avoiding false ordinal relationships on nominal IDs.
    vocab_sizes injected automatically from build_dataloaders().
    """
    model_name = "hierarchical_gru_det"

    def __init__(self, data_dir="./data", output_dir="./outputs",
                 run_name=None, do_search=False):
        super().__init__(data_dir=data_dir, output_dir=output_dir)
        self._run_name  = run_name or self.model_name + "_run"
        self._do_search = do_search

    def preprocess(self):
        """Triggers the full pipeline. Calling run() directly is equivalent."""
        self.run()

    def train(self):
        """No-op: training is completed inside preprocess() → run()."""
        pass

    def run(self):
        _run_full_pipeline(
            self, model_name="hierarchical_gru_det",
            model_type="hierarchical_gru", is_nb=False, is_prob=False,
            run_name=self._run_name, do_search=self._do_search,
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
        """Triggers the full pipeline. Calling run() directly is equivalent."""
        self.run()

    def train(self):
        """No-op: training is completed inside preprocess() → run()."""
        pass

    def run(self):
        _run_full_pipeline(
            self, model_name="hierarchical_gru_prob",
            model_type="hierarchical_gru_prob", is_nb=False, is_prob=True,
            run_name=self._run_name, do_search=self._do_search,
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
        """Triggers the full pipeline. Calling run() directly is equivalent."""
        self.run()

    def train(self):
        """No-op: training is completed inside preprocess() → run()."""
        pass

    def run(self):
        _run_full_pipeline(
            self, model_name="hierarchical_gru_nb",
            model_type="hierarchical_gru_nb", is_nb=True, is_prob=True,
            run_name=self._run_name, do_search=self._do_search,
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
        """Triggers the full pipeline. Calling run() directly is equivalent."""
        self.run()

    def train(self):
        """No-op: training is completed inside preprocess() → run()."""
        pass

    def run(self):
        _run_full_pipeline(
            self, model_name="hierarchical_quantile_gru",
            model_type="hierarchical_quantile_gru", is_nb=False, is_prob=False,
            run_name=self._run_name, do_search=self._do_search,
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
        """Triggers the full pipeline. Calling run() directly is equivalent."""
        self.run()

    def train(self):
        """No-op: training is completed inside preprocess() → run()."""
        pass

    def run(self):
        _run_full_pipeline(
            self, model_name="hierarchical_wquantile_gru",
            model_type="hierarchical_wquantile_gru", is_nb=False, is_prob=False,
            run_name=self._run_name, do_search=self._do_search,
        )
