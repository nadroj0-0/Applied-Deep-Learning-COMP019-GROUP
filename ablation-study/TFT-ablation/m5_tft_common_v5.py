
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss


DEFAULT_DATA_ROOT = Path(
    r"D:\game theory\m5-forecasting-uncertainty\Applied-Deep-Learning-COMP019-GROUP\m5_processed_validation"
)

QUANTILES: List[float] = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

# ---------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------

FEATURE_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "time_index": {
        "known_reals": ["d_num"],
    },
    "static_core": {
        "static_categoricals": ["item_idx", "dept_idx", "cat_idx", "store_idx", "state_idx"],
    },
    "id_embedding": {
        "static_categoricals": ["id_idx"],
    },
    "calendar_categoricals": {
        "known_categoricals": [
            "wday",
            "month",
            "week_of_year",
            "quarter",
            "event_name_1_code",
            "event_type_1_code",
            "event_name_2_code",
            "event_type_2_code",
        ],
    },
    "calendar_reals": {
        "known_reals": [
            "snap",
            "day_of_month",
            "day_of_year",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_year_start",
            "is_year_end",
        ],
    },
    "calendar_cyclical": {
        "known_reals": ["month_sin", "month_cos", "wday_sin", "wday_cos"],
    },
    "lifecycle_safe": {
        "known_reals": ["days_since_release"],
    },
    "price_core": {
        "known_reals": ["sell_price"],
    },
    "price_dynamics": {
        "known_reals": [
            "price_lag_1w",
            "price_change_1w",
            "price_pct_change_1w",
            "price_roll_mean_4w",
            "price_roll_mean_13w",
            "price_roll_mean_52w",
            "price_rel_4w",
            "price_rel_13w",
            "price_rel_52w",
            "price_rel_cat_store",
            "price_rel_dept_store",
            "price_rank_dept_store",
            "price_change_flag_1w",
        ],
    },
    "own_history": {
        "unknown_reals": [
            "sales_lag_1",
            "sales_lag_7",
            "sales_lag_14",
            "sales_lag_28",
            "sales_lag_56",
            "sales_roll_mean_7",
            "sales_roll_mean_28",
            "sales_roll_mean_56",
            "sales_roll_std_28",
            "sales_roll_nonzero_rate_28",
            "sale_occurrence",
        ],
    },
    "hierarchy_history": {
        "unknown_reals": [
            "store_sales_lag_7",
            "store_sales_lag_28",
            "store_sales_roll_mean_7",
            "store_sales_roll_mean_28",
            "state_sales_lag_7",
            "state_sales_lag_28",
            "state_sales_roll_mean_7",
            "state_sales_roll_mean_28",
            "cat_store_sales_lag_7",
            "cat_store_sales_lag_28",
            "cat_store_sales_roll_mean_7",
            "cat_store_sales_roll_mean_28",
            "dept_store_sales_lag_7",
            "dept_store_sales_lag_28",
            "dept_store_sales_roll_mean_7",
            "dept_store_sales_roll_mean_28",
        ],
    },
    "engineered_known": {
        "known_reals": [
            "event_lead_1",
            "event_lead_2",
            "event_lag_1",
            "event_lag_2",
            "promo_depth_4w",
            "promo_depth_13w",
        ],
    },
    "engineered_unknown": {
        "unknown_reals": [
            "trend_ratio_7_28",
            "trend_ratio_28_56",
            "item_share_cat_28",
            "item_share_dept_28",
        ],
    },
    "compact_fe_known": {
        "known_reals": [
            "event_lead_1",
            "event_lead_2",
            "event_lag_1",
            "event_lag_2",
            "days_to_next_event",
            "days_since_prev_event",
            "promo_depth_4w",
            "promo_depth_13w",
            "promo_depth_52w",
            "is_discount_13w",
            "days_in_price_regime",
        ],
    },
    "compact_fe_unknown": {
        "unknown_reals": [
            "trend_ratio_7_28",
            "trend_ratio_28_56",
            "volatility_ratio_28",
            "item_share_cat_28",
            "item_share_dept_28",
            "item_share_store_28",
            "days_since_last_nonzero_sale",
            "zero_streak_len",
        ],
    },
    # IMPORTANT: do not use the raw static table directly for fold CV because it was
    # computed on the full d_1..d_1913 history. We recompute these fold-safely.
    "fold_safe_stats": {
        "static_reals": [
            "series_mean_sales",
            "series_std_sales",
            "series_zero_rate",
            "series_nonzero_count",
            "series_mean_nonzero_sales",
        ],
    },
}

FEATURE_SETS: Dict[str, List[str]] = {
    "baseline": [
        "time_index",
        "static_core",
        "calendar_categoricals",
        "calendar_reals",
        "price_core",
        "lifecycle_safe",
    ],
    "baseline_plus_price": [
        "time_index",
        "static_core",
        "calendar_categoricals",
        "calendar_reals",
        "price_core",
        "price_dynamics",
        "lifecycle_safe",
    ],
    "baseline_plus_history": [
        "time_index",
        "static_core",
        "calendar_categoricals",
        "calendar_reals",
        "price_core",
        "price_dynamics",
        "own_history",
        "lifecycle_safe",
    ],
    "baseline_plus_hierarchy": [
        "time_index",
        "static_core",
        "calendar_categoricals",
        "calendar_reals",
        "price_core",
        "price_dynamics",
        "own_history",
        "hierarchy_history",
        "lifecycle_safe",
    ],
    "full_safe": [
        "time_index",
        "static_core",
        "calendar_categoricals",
        "calendar_reals",
        "calendar_cyclical",
        "price_core",
        "price_dynamics",
        "own_history",
        "hierarchy_history",
        "lifecycle_safe",
    ],
    "full_safe_plus_compact_fe": [
        "time_index",
        "static_core",
        "calendar_categoricals",
        "calendar_reals",
        "calendar_cyclical",
        "price_core",
        "price_dynamics",
        "own_history",
        "hierarchy_history",
        "lifecycle_safe",
        "compact_fe_known",
        "compact_fe_unknown",
    ],
    "full_safe_plus_engineered": [
        "time_index",
        "static_core",
        "calendar_categoricals",
        "calendar_reals",
        "calendar_cyclical",
        "price_core",
        "price_dynamics",
        "own_history",
        "hierarchy_history",
        "lifecycle_safe",
        "engineered_known",
        "engineered_unknown",
    ],
    "full_safe_plus_stats": [
        "time_index",
        "static_core",
        "calendar_categoricals",
        "calendar_reals",
        "calendar_cyclical",
        "price_core",
        "price_dynamics",
        "own_history",
        "hierarchy_history",
        "lifecycle_safe",
        "fold_safe_stats",
    ],
    "full_safe_plus_id": [
        "time_index",
        "static_core",
        "id_embedding",
        "calendar_categoricals",
        "calendar_reals",
        "calendar_cyclical",
        "price_core",
        "price_dynamics",
        "own_history",
        "hierarchy_history",
        "lifecycle_safe",
    ],
}


@dataclass
class FoldWindow:
    name: str
    train_start: int
    train_end: int
    validation_start: Optional[int] = None
    validation_end: Optional[int] = None
    predict_start: Optional[int] = None
    predict_end: Optional[int] = None

    @property
    def is_holdout(self) -> bool:
        return self.validation_start is None and self.predict_start is not None

    @property
    def eval_start(self) -> int:
        return self.validation_start if self.validation_start is not None else int(self.predict_start)

    @property
    def eval_end(self) -> int:
        return self.validation_end if self.validation_end is not None else int(self.predict_end)

    @property
    def horizon(self) -> int:
        return self.eval_end - self.eval_start + 1


@dataclass
class HardwareConfig:
    accelerator: str
    devices: int
    precision: str
    pin_memory: bool
    num_workers: int
    persistent_workers: bool
    prefetch_factor: Optional[int]


@dataclass
class TrainConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_dir: Path = Path("runs")
    fold_name: str = "fold_2"
    feature_set_name: str = "baseline_plus_hierarchy"
    seed: int = 42
    encoder_length: int = 56
    prediction_length: int = 28
    train_window_count: int = 64
    batch_size: int = 128
    max_epochs: int = 12
    learning_rate: float = 1e-3
    hidden_size: int = 32
    attention_head_size: int = 4
    hidden_continuous_size: int = 16
    lstm_layers: int = 2
    dropout: float = 0.10
    gradient_clip_val: float = 0.10
    max_series: Optional[int] = None
    series_ids_path: Optional[Path] = None
    sampling_strategy: str = "stratified"
    checkpoint_dirname: str = "checkpoints"
    log_dirname: str = "logs"
    extra_workers: Optional[int] = None
    enforce_monotonic_quantiles: bool = True


# ---------------------------------------------------------------------
# Split handling
# ---------------------------------------------------------------------

_DEFAULT_SPLITS: Dict[str, FoldWindow] = {
    "fold_1": FoldWindow(name="fold_1", train_start=1, train_end=1857, validation_start=1858, validation_end=1885),
    "fold_2": FoldWindow(name="fold_2", train_start=1, train_end=1885, validation_start=1886, validation_end=1913),
    "holdout": FoldWindow(name="holdout", train_start=1, train_end=1913, predict_start=1914, predict_end=1941),
}


def _parse_d_range(text: str) -> Tuple[int, int]:
    nums = list(map(int, re.findall(r"d_(\d+)", str(text))))
    if len(nums) != 2:
        raise ValueError(f"Could not parse d-range from: {text!r}")
    return nums[0], nums[1]


def _parse_window_spec(value: Any, *, field_name: str) -> Tuple[int, int]:
    if value is None or value == "":
        raise ValueError(f"Split field {field_name!r} is empty")

    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])

    if isinstance(value, dict):
        if {"start", "end"} <= set(value):
            return int(value["start"]), int(value["end"])
        if {"from", "to"} <= set(value):
            return int(value["from"]), int(value["to"])

    return _parse_d_range(str(value))


def load_recommended_splits(data_root: Path) -> Dict[str, FoldWindow]:
    split_path = data_root / "metadata" / "validation_recommended_splits.json"
    if not split_path.exists():
        return _DEFAULT_SPLITS.copy()

    with split_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_items: List[Tuple[str, Dict[str, Any]]] = []

    if isinstance(payload, dict):
        if "splits" in payload and isinstance(payload["splits"], dict):
            raw_items = [(str(name), spec) for name, spec in payload["splits"].items() if isinstance(spec, dict)]
        else:
            raw_items = [(str(name), spec) for name, spec in payload.items() if isinstance(spec, dict)]
    elif isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            name = row.get("split") or row.get("name") or row.get("fold")
            if name is None:
                continue
            raw_items.append((str(name), row))
    else:
        return _DEFAULT_SPLITS.copy()

    splits: Dict[str, FoldWindow] = {}
    for name, spec in raw_items:
        train_value = spec.get("train_window", spec.get("train"))
        valid_value = spec.get("validation_window", spec.get("valid"))
        predict_value = spec.get("predict_window", spec.get("predict"))

        if train_value in (None, ""):
            continue

        train_start, train_end = _parse_window_spec(train_value, field_name=f"{name}.train")

        if valid_value not in (None, ""):
            val_start, val_end = _parse_window_spec(valid_value, field_name=f"{name}.valid")
            splits[name] = FoldWindow(
                name=name,
                train_start=train_start,
                train_end=train_end,
                validation_start=val_start,
                validation_end=val_end,
            )
        elif predict_value not in (None, ""):
            pred_start, pred_end = _parse_window_spec(predict_value, field_name=f"{name}.predict")
            splits[name] = FoldWindow(
                name=name,
                train_start=train_start,
                train_end=train_end,
                predict_start=pred_start,
                predict_end=pred_end,
            )

    return splits or _DEFAULT_SPLITS.copy()


# ---------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------

def resolve_num_workers(requested: Optional[int] = None) -> int:
    if requested is not None:
        return max(0, int(requested))
    cpu_count = max(1, (os.cpu_count() or 1))
    # On laptops and especially on Windows, too many workers often hurts more than it helps.
    return min(4, cpu_count)


def get_hardware_config(extra_workers: Optional[int] = None) -> HardwareConfig:
    num_workers = resolve_num_workers(extra_workers)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        return HardwareConfig(
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    mps_is_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if mps_is_available:
        return HardwareConfig(
            accelerator="mps",
            devices=1,
            precision="32-true",
            pin_memory=False,
            num_workers=0,
            persistent_workers=False,
            prefetch_factor=None,
        )

    return HardwareConfig(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        pin_memory=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

import os


def seed_everything(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)


def merge_feature_dicts(groups: Sequence[str]) -> Dict[str, List[str]]:
    merged = {
        "static_categoricals": [],
        "static_reals": [],
        "known_categoricals": [],
        "known_reals": [],
        "unknown_reals": [],
    }
    for group_name in groups:
        if group_name not in FEATURE_GROUPS:
            raise KeyError(f"Unknown feature group: {group_name}")
        group = FEATURE_GROUPS[group_name]
        for bucket, cols in group.items():
            for col in cols:
                if col not in merged[bucket]:
                    merged[bucket].append(col)
    return merged


def get_feature_spec(feature_set_name: str) -> Dict[str, List[str]]:
    if feature_set_name not in FEATURE_SETS:
        raise KeyError(
            f"Unknown feature set: {feature_set_name}. Available: {', '.join(sorted(FEATURE_SETS))}"
        )
    return merge_feature_dicts(FEATURE_SETS[feature_set_name])


def list_feature_set_names() -> List[str]:
    return sorted(FEATURE_SETS)


def _batch_paths(data_root: Path) -> List[Path]:
    batch_dir = data_root / "features"
    paths = sorted(batch_dir.glob("validation_features_batch_*.pkl.gz"))
    if not paths:
        raise FileNotFoundError(f"No feature batches found in {batch_dir}")
    return paths



def load_series_info(data_root: Path) -> pd.DataFrame:
    path = data_root / "static" / "validation_series_info.pkl.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_pickle(path, compression="gzip")


def load_series_stats(data_root: Path) -> pd.DataFrame:
    path = data_root / "static" / "validation_series_stats.pkl.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_pickle(path, compression="gzip")


def _read_series_id_file(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "ids" in payload:
                values = payload["ids"]
            else:
                values = list(payload.values())
        else:
            values = payload
        return {str(v) for v in values if str(v).strip()}

    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "	"
        frame = pd.read_csv(path, sep=sep)
        for candidate in ("id", "series_id"):
            if candidate in frame.columns:
                return set(frame[candidate].astype(str))
        if frame.shape[1] == 1:
            return set(frame.iloc[:, 0].astype(str))
        raise ValueError(f"Could not find an id column in {path}")

    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _build_sampling_metadata(data_root: Path) -> pd.DataFrame:
    series_info = load_series_info(data_root)
    series_stats = load_series_stats(data_root)[["id", "series_mean_sales", "series_zero_rate"]]
    meta = series_info.merge(series_stats, on="id", how="left")

    meta["series_mean_sales"] = pd.to_numeric(meta["series_mean_sales"], errors="coerce").fillna(0.0)
    meta["series_zero_rate"] = pd.to_numeric(meta["series_zero_rate"], errors="coerce").fillna(1.0)
    meta["zero_bin"] = pd.qcut(
        meta["series_zero_rate"].rank(method="first"),
        q=4,
        labels=["zr0", "zr1", "zr2", "zr3"],
    )
    return meta


def _allocate_group_counts(group_sizes: pd.Series, total: int) -> pd.Series:
    sizes = group_sizes.astype(int)
    nonempty = sizes[sizes > 0]

    if total >= int(nonempty.sum()):
        return sizes.copy()

    base = pd.Series(0, index=sizes.index, dtype="int64")

    if total < len(nonempty):
        keep = nonempty.sort_values(ascending=False).index[:total]
        base.loc[keep] = 1
        return base

    base.loc[nonempty.index] = 1
    remaining = total - len(nonempty)
    weights = nonempty / nonempty.sum()

    extras = np.floor(weights * remaining).astype("int64")
    allocation = base.copy()
    allocation.loc[nonempty.index] += extras

    used = int(allocation.sum())
    leftover = total - used
    if leftover > 0:
        fractional = (weights * remaining - extras).sort_values(ascending=False)
        ordered = list(fractional.index)
        i = 0
        while leftover > 0:
            allocation.loc[ordered[i % len(ordered)]] += 1
            leftover -= 1
            i += 1

    allocation = np.minimum(allocation, sizes).astype("int64")

    deficit = total - int(allocation.sum())
    if deficit > 0:
        spare = (sizes - allocation).sort_values(ascending=False)
        for idx, cap in spare.items():
            if deficit <= 0:
                break
            if cap <= 0:
                continue
            add = min(int(cap), deficit)
            allocation.loc[idx] += add
            deficit -= add

    return allocation


def _pick_evenly_spaced_ids(group: pd.DataFrame, n: int) -> List[str]:
    ordered = group.sort_values(["series_mean_sales", "series_zero_rate", "id"]).reset_index(drop=True)
    if n >= len(ordered):
        return ordered["id"].astype(str).tolist()

    positions = np.linspace(0, len(ordered) - 1, n)
    raw_idx = np.round(positions).astype(int)
    raw_idx = np.clip(raw_idx, 0, len(ordered) - 1)

    chosen: List[int] = []
    used: set[int] = set()
    for idx in raw_idx:
        if idx not in used:
            chosen.append(int(idx))
            used.add(int(idx))
            continue
        left = idx - 1
        right = idx + 1
        found = None
        while left >= 0 or right < len(ordered):
            if left >= 0 and left not in used:
                found = left
                break
            if right < len(ordered) and right not in used:
                found = right
                break
            left -= 1
            right += 1
        if found is None:
            break
        chosen.append(int(found))
        used.add(int(found))

    if len(chosen) < n:
        for idx in range(len(ordered)):
            if idx not in used:
                chosen.append(idx)
                used.add(idx)
            if len(chosen) >= n:
                break

    chosen = sorted(chosen[:n])
    return ordered.iloc[chosen]["id"].astype(str).tolist()


def sample_series_ids(
    data_root: Path,
    max_series: Optional[int],
    seed: int,
    series_ids_path: Optional[Path] = None,
    sampling_strategy: str = "stratified",
) -> Optional[set[str]]:
    if series_ids_path is not None:
        return _read_series_id_file(series_ids_path)

    if max_series is None:
        return None

    meta = _build_sampling_metadata(data_root)
    if max_series >= len(meta):
        return set(meta["id"].astype(str).tolist())

    sampling_strategy = str(sampling_strategy).lower().strip()
    if sampling_strategy == "random":
        sampled = meta.sample(n=max_series, random_state=seed)
        return set(sampled["id"].astype(str).tolist())

    if sampling_strategy != "stratified":
        raise ValueError(f"Unknown sampling_strategy={sampling_strategy!r}. Use 'stratified' or 'random'.")

    group_cols = ["cat_id", "store_id", "zero_bin"]
    group_sizes = meta.groupby(group_cols, observed=True).size()
    allocation = _allocate_group_counts(group_sizes, total=max_series)

    selected_ids: List[str] = []
    for key, n_keep in allocation.items():
        if int(n_keep) <= 0:
            continue
        cat_id, store_id, zero_bin = key
        group = meta[
            (meta["cat_id"] == cat_id)
            & (meta["store_id"] == store_id)
            & (meta["zero_bin"] == zero_bin)
        ]
        selected_ids.extend(_pick_evenly_spaced_ids(group, int(n_keep)))

    if len(selected_ids) > max_series:
        selected_ids = selected_ids[:max_series]

    return set(selected_ids)


def _required_static_columns(feature_spec: Dict[str, List[str]]) -> List[str]:
    base = ["id"]
    maybe_needed = {
        "id_idx",
        "item_idx",
        "dept_idx",
        "cat_idx",
        "store_idx",
        "state_idx",
    }
    needed = set(feature_spec["static_categoricals"]) | set(feature_spec["static_reals"])
    cols = base + [
        c
        for c in maybe_needed
        if c in needed or c in {"id_idx", "item_idx", "dept_idx", "cat_idx", "store_idx", "state_idx"}
    ]
    # always keep all numeric static ids because they are tiny and often useful later
    return list(dict.fromkeys(cols))


def _required_batch_feature_columns(
    feature_spec: Dict[str, List[str]],
    add_engineered: bool,
) -> List[str]:
    cols: List[str] = ["id", "d_num", "sales"]

    for bucket in ("known_categoricals", "known_reals", "unknown_reals"):
        for col in feature_spec[bucket]:
            if col not in cols:
                cols.append(col)

    if add_engineered:
        engineered_source_cols = [
            "is_event_day",
            "sell_price",
            "price_rel_4w",
            "price_rel_13w",
            "price_rel_52w",
            "sales_roll_mean_7",
            "sales_roll_mean_28",
            "sales_roll_mean_56",
            "sales_roll_std_28",
            "store_sales_roll_mean_28",
            "cat_store_sales_roll_mean_28",
            "dept_store_sales_roll_mean_28",
        ]
        for col in engineered_source_cols:
            if col not in cols:
                cols.append(col)

    return cols


def _compute_training_window_bounds(
    train_end: int,
    encoder_length: int,
    prediction_length: int,
    train_window_count: int,
) -> Tuple[int, int]:
    max_decoder_start = train_end - prediction_length + 1
    min_decoder_start = max(1 + encoder_length, max_decoder_start - train_window_count + 1)
    slice_start = max(1, min_decoder_start - encoder_length)
    return min_decoder_start, slice_start



def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact, leakage-safe feature engineering layered on top of the processed M5 export.

    Known-future additions:
      - event lead/lag flags
      - distance to next / previous event
      - promo depth relative to rolling price anchors
      - discount flag
      - length of current price regime

    Observed-past additions:
      - demand momentum / volatility ratios
      - hierarchy share ratios
      - intermittent-demand recency / zero-streak features

    Notes:
      - The frame is already trimmed to a relatively small rolling slice before this runs.
      - We only use columns that are already present in the processed export.
      - Group-wise sequence features are computed in a single pass over each series.
    """
    df = df.sort_values(["id", "d_num"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Calendar-based known-future event features
    # ------------------------------------------------------------------
    calendar = (
        df[["d_num", "is_event_day"]]
        .drop_duplicates("d_num")
        .sort_values("d_num")
        .reset_index(drop=True)
    )
    calendar["is_event_day"] = pd.to_numeric(calendar["is_event_day"], errors="coerce").fillna(0).astype("int8")
    d_vals = calendar["d_num"].to_numpy(dtype="int32")
    event_flag = calendar["is_event_day"].to_numpy(dtype="int8")
    event_days = d_vals[event_flag == 1]

    for k in (1, 2):
        calendar[f"event_lead_{k}"] = calendar["is_event_day"].shift(-k).fillna(0).astype("float32")
        calendar[f"event_lag_{k}"] = calendar["is_event_day"].shift(k).fillna(0).astype("float32")

    if len(event_days) == 0:
        calendar["days_to_next_event"] = np.float32(366.0)
        calendar["days_since_prev_event"] = np.float32(366.0)
    else:
        next_idx = np.searchsorted(event_days, d_vals, side="left")
        prev_idx = np.searchsorted(event_days, d_vals, side="right") - 1

        next_dist = np.full(len(d_vals), 366.0, dtype="float32")
        prev_dist = np.full(len(d_vals), 366.0, dtype="float32")

        valid_next = next_idx < len(event_days)
        valid_prev = prev_idx >= 0

        next_dist[valid_next] = (event_days[next_idx[valid_next]] - d_vals[valid_next]).astype("float32")
        prev_dist[valid_prev] = (d_vals[valid_prev] - event_days[prev_idx[valid_prev]]).astype("float32")

        calendar["days_to_next_event"] = next_dist
        calendar["days_since_prev_event"] = prev_dist

    df = df.merge(
        calendar[
            [
                "d_num",
                "event_lead_1",
                "event_lead_2",
                "event_lag_1",
                "event_lag_2",
                "days_to_next_event",
                "days_since_prev_event",
            ]
        ],
        on="d_num",
        how="left",
    )

    # ------------------------------------------------------------------
    # Row-wise ratios and relative features
    # ------------------------------------------------------------------
    eps = np.float32(1e-3)

    if "price_rel_4w" in df:
        df["promo_depth_4w"] = np.clip(1.0 - df["price_rel_4w"].astype("float32"), -3.0, 3.0)
    if "price_rel_13w" in df:
        df["promo_depth_13w"] = np.clip(1.0 - df["price_rel_13w"].astype("float32"), -3.0, 3.0)
    if "price_rel_52w" in df:
        df["promo_depth_52w"] = np.clip(1.0 - df["price_rel_52w"].astype("float32"), -3.0, 3.0)

    if {"sell_price", "price_roll_mean_13w"} <= set(df.columns):
        sell_price = pd.to_numeric(df["sell_price"], errors="coerce").astype("float32")
        price_roll_13 = pd.to_numeric(df["price_roll_mean_13w"], errors="coerce").astype("float32")
        df["is_discount_13w"] = (sell_price < price_roll_13).astype("float32")

    if {"sales_roll_mean_7", "sales_roll_mean_28"} <= set(df.columns):
        df["trend_ratio_7_28"] = (
            df["sales_roll_mean_7"].astype("float32") / (df["sales_roll_mean_28"].astype("float32") + eps)
        )
    if {"sales_roll_mean_28", "sales_roll_mean_56"} <= set(df.columns):
        df["trend_ratio_28_56"] = (
            df["sales_roll_mean_28"].astype("float32") / (df["sales_roll_mean_56"].astype("float32") + eps)
        )
    if {"sales_roll_std_28", "sales_roll_mean_28"} <= set(df.columns):
        df["volatility_ratio_28"] = (
            df["sales_roll_std_28"].astype("float32") / (df["sales_roll_mean_28"].astype("float32") + eps)
        )
    if {"sales_roll_mean_28", "cat_store_sales_roll_mean_28"} <= set(df.columns):
        df["item_share_cat_28"] = (
            df["sales_roll_mean_28"].astype("float32")
            / (df["cat_store_sales_roll_mean_28"].astype("float32") + eps)
        )
    if {"sales_roll_mean_28", "dept_store_sales_roll_mean_28"} <= set(df.columns):
        df["item_share_dept_28"] = (
            df["sales_roll_mean_28"].astype("float32")
            / (df["dept_store_sales_roll_mean_28"].astype("float32") + eps)
        )
    if {"sales_roll_mean_28", "store_sales_roll_mean_28"} <= set(df.columns):
        df["item_share_store_28"] = (
            df["sales_roll_mean_28"].astype("float32")
            / (df["store_sales_roll_mean_28"].astype("float32") + eps)
        )

    # ------------------------------------------------------------------
    # Per-series sequence features
    # ------------------------------------------------------------------
    n = len(df)
    d_num = pd.to_numeric(df["d_num"], errors="coerce").to_numpy(dtype="int32", copy=False)
    sales = pd.to_numeric(df["sales"], errors="coerce").to_numpy(dtype="float32", copy=False)

    has_price = "sell_price" in df.columns
    if has_price:
        prices = pd.to_numeric(df["sell_price"], errors="coerce").to_numpy(dtype="float32", copy=False)
        days_in_price_regime = np.ones(n, dtype="float32")
    else:
        prices = None
        days_in_price_regime = np.zeros(n, dtype="float32")

    days_since_last_nonzero_sale = np.zeros(n, dtype="float32")
    zero_streak_len = np.zeros(n, dtype="float32")

    ids = df["id"].astype(str).to_numpy()
    start = 0
    while start < n:
        end = start + 1
        current_id = ids[start]
        while end < n and ids[end] == current_id:
            end += 1

        last_nonzero_day = None
        in_zero_regime = False
        current_zero_streak = 0.0
        current_price_regime = 1.0

        for pos in range(start, end):
            day = int(d_num[pos])
            prev_day = int(d_num[pos - 1]) if pos > start else day
            day_gap = float(max(day - prev_day, 1))
            sale = sales[pos]
            observed_sale = not np.isnan(sale)

            if observed_sale:
                if sale > 0:
                    last_nonzero_day = day
                    in_zero_regime = False
                    current_zero_streak = 0.0
                else:
                    if in_zero_regime:
                        current_zero_streak += day_gap
                    else:
                        current_zero_streak = float(day - last_nonzero_day) if last_nonzero_day is not None else day_gap
                    in_zero_regime = True
            else:
                if in_zero_regime:
                    current_zero_streak += day_gap

            days_since_last_nonzero_sale[pos] = (
                float(day - last_nonzero_day) if last_nonzero_day is not None else 0.0
            )
            zero_streak_len[pos] = current_zero_streak if in_zero_regime else 0.0

            if has_price:
                if pos == start:
                    current_price_regime = 1.0
                else:
                    prev_price = prices[pos - 1]
                    cur_price = prices[pos]
                    if np.isfinite(prev_price) and np.isfinite(cur_price) and np.isclose(prev_price, cur_price, atol=1e-6):
                        current_price_regime += day_gap
                    else:
                        current_price_regime = 1.0
                days_in_price_regime[pos] = current_price_regime

        start = end

    df["days_since_last_nonzero_sale"] = days_since_last_nonzero_sale
    df["zero_streak_len"] = zero_streak_len
    if has_price:
        df["days_in_price_regime"] = days_in_price_regime

    return df


def compute_fold_safe_series_stats_streaming(
    data_root: Path,
    train_end: int,
    allowed_ids: Optional[set[str]],
) -> pd.DataFrame:
    stats_accumulator: Dict[str, Dict[str, float]] = {}

    for batch_path in _batch_paths(data_root):
        chunk = pd.read_pickle(batch_path, compression="gzip")[["id", "d_num", "sales"]]
        chunk = chunk[(chunk["d_num"] <= train_end) & chunk["sales"].notna()]
        if allowed_ids is not None:
            chunk = chunk[chunk["id"].astype(str).isin(allowed_ids)]
        if chunk.empty:
            continue

        chunk["sales"] = chunk["sales"].astype("float64")
        grouped = chunk.groupby("id")["sales"]

        partial = pd.DataFrame(
            {
                "count": grouped.size(),
                "sum_sales": grouped.sum(),
                "sumsq_sales": grouped.apply(lambda s: np.square(s.to_numpy()).sum()),
                "zero_count": grouped.apply(lambda s: (s.to_numpy() == 0).sum()),
                "nonzero_sum": grouped.apply(lambda s: s[s > 0].sum()),
                "nonzero_count": grouped.apply(lambda s: (s.to_numpy() > 0).sum()),
            }
        ).reset_index()

        for row in partial.itertuples(index=False):
            key = str(row.id)
            acc = stats_accumulator.setdefault(
                key,
                {
                    "count": 0.0,
                    "sum_sales": 0.0,
                    "sumsq_sales": 0.0,
                    "zero_count": 0.0,
                    "nonzero_sum": 0.0,
                    "nonzero_count": 0.0,
                },
            )
            acc["count"] += float(row.count)
            acc["sum_sales"] += float(row.sum_sales)
            acc["sumsq_sales"] += float(row.sumsq_sales)
            acc["zero_count"] += float(row.zero_count)
            acc["nonzero_sum"] += float(row.nonzero_sum)
            acc["nonzero_count"] += float(row.nonzero_count)

    rows: List[Dict[str, Any]] = []
    for series_id, acc in stats_accumulator.items():
        count = max(acc["count"], 1.0)
        mean_sales = acc["sum_sales"] / count
        mean_sq = acc["sumsq_sales"] / count
        var_sales = max(mean_sq - mean_sales**2, 0.0)
        nonzero_count = acc["nonzero_count"]
        mean_nonzero = acc["nonzero_sum"] / nonzero_count if nonzero_count > 0 else 0.0

        rows.append(
            {
                "id": series_id,
                "series_mean_sales": np.float32(mean_sales),
                "series_std_sales": np.float32(math.sqrt(var_sales)),
                "series_zero_rate": np.float32(acc["zero_count"] / count),
                "series_nonzero_count": np.float32(nonzero_count),
                "series_mean_nonzero_sales": np.float32(mean_nonzero),
            }
        )

    return pd.DataFrame(rows)


def load_fold_frame(
    data_root: Path,
    fold: FoldWindow,
    feature_spec: Dict[str, List[str]],
    encoder_length: int,
    prediction_length: int,
    train_window_count: int,
    max_series: Optional[int] = None,
    seed: int = 42,
    add_engineered: bool = True,
    series_ids_path: Optional[Path] = None,
    sampling_strategy: str = "stratified",
) -> Tuple[pd.DataFrame, int]:
    allowed_ids = sample_series_ids(
        data_root,
        max_series=max_series,
        seed=seed,
        series_ids_path=series_ids_path,
        sampling_strategy=sampling_strategy,
    )
    min_train_prediction_idx, slice_start = _compute_training_window_bounds(
        train_end=fold.train_end,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        train_window_count=train_window_count,
    )

    eval_end = fold.eval_end

    series_info = load_series_info(data_root)
    if allowed_ids is not None:
        series_info = series_info[series_info["id"].astype(str).isin(allowed_ids)]

    static_cols = _required_static_columns(feature_spec)
    static_info = series_info[static_cols].copy()
    required_batch_cols = _required_batch_feature_columns(feature_spec, add_engineered=add_engineered)

    frames: List[pd.DataFrame] = []
    for batch_path in _batch_paths(data_root):
        chunk = pd.read_pickle(batch_path, compression="gzip")
        chunk = chunk[(chunk["d_num"] >= slice_start) & (chunk["d_num"] <= eval_end)]

        if allowed_ids is not None:
            chunk = chunk[chunk["id"].astype(str).isin(allowed_ids)]

        if chunk.empty:
            continue

        keep_cols = [c for c in required_batch_cols if c in chunk.columns]
        chunk = chunk[keep_cols].copy()
        chunk = chunk.merge(static_info, on="id", how="left")

        missing_static = [
            col for col in static_cols if col != "id" and col in chunk.columns and chunk[col].isna().any()
        ]
        if missing_static:
            raise ValueError(
                f"Static join failed for columns {missing_static}. Check validation_series_info.pkl.gz and id alignment."
            )

        frames.append(chunk)

    if not frames:
        raise RuntimeError("No rows were loaded for the requested fold / feature configuration.")

    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.sort_values(["id", "d_num"]).reset_index(drop=True)

    if add_engineered:
        df = add_engineered_features(df)

    if (
        "static_reals" in feature_spec
        and any(c in feature_spec["static_reals"] for c in FEATURE_GROUPS["fold_safe_stats"]["static_reals"])
    ):
        fold_safe_stats = compute_fold_safe_series_stats_streaming(
            data_root=data_root,
            train_end=fold.train_end,
            allowed_ids=allowed_ids,
        )
        if not fold_safe_stats.empty:
            df = df.merge(fold_safe_stats, on="id", how="left")

    return df, min_train_prediction_idx


def _fill_real_column(df: pd.DataFrame, col: str) -> pd.Series:
    series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    if col == "price_lag_1w":
        fallback = pd.to_numeric(df.get("sell_price", 0.0), errors="coerce")
        series = series.fillna(fallback)
    elif col in {"price_roll_mean_4w", "price_roll_mean_13w", "price_roll_mean_52w"}:
        fallback = pd.to_numeric(df.get("sell_price", 0.0), errors="coerce")
        series = series.fillna(fallback)
    elif col in {"price_rel_4w", "price_rel_13w", "price_rel_52w", "price_rel_cat_store", "price_rel_dept_store"}:
        series = series.fillna(1.0)
    elif col == "price_rank_dept_store":
        series = series.fillna(0.5)
    else:
        series = series.fillna(0.0)

    return series.astype("float32")


def sanitize_frame_for_tft(
    df: pd.DataFrame,
    feature_spec: Dict[str, List[str]],
) -> pd.DataFrame:
    required_cols = list(
        dict.fromkeys(
            ["id", "d_num", "sales"]
            + feature_spec["static_categoricals"]
            + feature_spec["static_reals"]
            + feature_spec["known_categoricals"]
            + feature_spec["known_reals"]
            + feature_spec["unknown_reals"]
        )
    )

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Required feature columns missing from frame: {missing_cols}")

    df = df[required_cols].copy()

    duplicated_rows = int(df.duplicated(["id", "d_num"]).sum())
    if duplicated_rows > 0:
        raise ValueError(f"Found {duplicated_rows} duplicate (id, d_num) rows in the fold frame")

    categorical_cols = ["id"] + feature_spec["static_categoricals"] + feature_spec["known_categoricals"]
    real_cols = feature_spec["static_reals"] + feature_spec["known_reals"] + feature_spec["unknown_reals"]

    for col in categorical_cols:
        if col not in df.columns:
            raise KeyError(f"Categorical feature {col!r} is missing from dataframe")

        series = df[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype("string")
        else:
            series = series.astype("string")

        series = series.fillna("__MISSING__")
        df[col] = (col + "_" + series.astype(str)).astype("category")

    for col in real_cols:
        if col not in df.columns:
            raise KeyError(f"Real feature {col!r} is missing from dataframe")
        df[col] = _fill_real_column(df, col)

    df["sales"] = (
        pd.to_numeric(df["sales"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )
    df["d_num"] = pd.to_numeric(df["d_num"], errors="raise").astype("int32")

    return df


def build_datasets(
    df: pd.DataFrame,
    feature_spec: Dict[str, List[str]],
    fold: FoldWindow,
    min_train_prediction_idx: int,
    encoder_length: int,
    prediction_length: int,
    batch_size: int,
    hardware: HardwareConfig,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, Any, Any]:
    df = sanitize_frame_for_tft(df, feature_spec)

    train_df = df[df["d_num"] <= fold.train_end].copy()
    predict_df = df[df["d_num"] <= fold.eval_end].copy()

    categorical_cols = ["id"] + feature_spec["static_categoricals"] + feature_spec["known_categoricals"]
    categorical_encoders = {
        col: NaNLabelEncoder(add_nan=True).fit(df[col].astype(str))
        for col in categorical_cols
    }

    training = TimeSeriesDataSet(
        train_df,
        time_idx="d_num",
        target="sales",
        group_ids=["id"],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=prediction_length,
        max_prediction_length=prediction_length,
        min_prediction_idx=min_train_prediction_idx,
        static_categoricals=feature_spec["static_categoricals"],
        static_reals=feature_spec["static_reals"],
        time_varying_known_categoricals=feature_spec["known_categoricals"],
        time_varying_known_reals=feature_spec["known_reals"],
        time_varying_unknown_reals=["sales"] + feature_spec["unknown_reals"],
        target_normalizer=GroupNormalizer(groups=["id"], transformation="softplus"),
        categorical_encoders=categorical_encoders,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        predict_df,
        predict=True,
        stop_randomization=True,
    )

    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": hardware.num_workers,
        "pin_memory": hardware.pin_memory,
        "persistent_workers": hardware.persistent_workers,
    }
    if hardware.prefetch_factor is not None and hardware.num_workers > 0:
        loader_kwargs["prefetch_factor"] = hardware.prefetch_factor

    train_loader = training.to_dataloader(train=True, **loader_kwargs)
    val_loader = validation.to_dataloader(train=False, **loader_kwargs)

    return training, validation, train_loader, val_loader


def create_trainer_and_model(
    training: TimeSeriesDataSet,
    config: TrainConfig,
    hardware: HardwareConfig,
    run_dir: Path,
) -> Tuple[Any, TemporalFusionTransformer, ModelCheckpoint]:
    checkpoint_dir = run_dir / config.checkpoint_dirname
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = CSVLogger(save_dir=str(run_dir / config.log_dirname), name=config.feature_set_name, version=config.fold_name)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=3,
        verbose=True,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"{config.feature_set_name}-{config.fold_name}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=hardware.accelerator,
        devices=hardware.devices,
        precision=hardware.precision,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[lr_monitor, early_stopping, checkpoint_callback],
        logger=logger,
        log_every_n_steps=25,
        num_sanity_val_steps=0,
        enable_model_summary=True,
    )

    # PyTorch Forecasting's TFT uses a very large negative attention mask bias by default.
    # In mixed precision this can overflow float16; the library docs recommend using
    # -float("inf") for mixed precision training.
    mask_bias = -float("inf") if hardware.precision != "32-true" else -1e9

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        hidden_continuous_size=config.hidden_continuous_size,
        lstm_layers=config.lstm_layers,
        dropout=config.dropout,
        output_size=len(QUANTILES),
        loss=QuantileLoss(quantiles=QUANTILES),
        log_interval=-1,
        reduce_on_plateau_patience=2,
        optimizer="AdamW",
        mask_bias=mask_bias,
    )

    return trainer, model, checkpoint_callback


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_actual_array(y: Any) -> Optional[np.ndarray]:
    if y is None:
        return None
    if isinstance(y, tuple):
        if len(y) == 0:
            return None
        return _to_numpy(y[0])
    return _to_numpy(y)


def _extract_prediction_tensor(prediction_output: Any) -> Any:
    if hasattr(prediction_output, "output") and getattr(prediction_output, "output") is not None:
        return prediction_output.output
    if hasattr(prediction_output, "prediction") and getattr(prediction_output, "prediction") is not None:
        return prediction_output.prediction
    return prediction_output


def build_prediction_frame(
    prediction_output: Any,
    quantiles: Sequence[float] = QUANTILES,
    enforce_monotonic_quantiles: bool = True,
) -> pd.DataFrame:
    pred = _to_numpy(_extract_prediction_tensor(prediction_output))
    if pred.ndim != 3:
        raise ValueError(f"Expected [n_series, horizon, n_quantiles] prediction tensor, got shape {pred.shape}")

    if enforce_monotonic_quantiles:
        pred = np.sort(pred, axis=-1)

    actual = extract_actual_array(getattr(prediction_output, "y", None))
    index_df = prediction_output.index.reset_index(drop=True).copy()

    group_col = "id" if "id" in index_df.columns else index_df.columns[0]
    time_col = "d_num" if "d_num" in index_df.columns else index_df.columns[-1]

    n_series, horizon, n_quantiles = pred.shape
    if n_quantiles != len(quantiles):
        raise ValueError(f"Quantile dimension mismatch: got {n_quantiles}, expected {len(quantiles)}")

    rows: List[Dict[str, Any]] = []
    for i in range(n_series):
        start_time = int(index_df.loc[i, time_col])
        series_id = str(index_df.loc[i, group_col])

        for step in range(horizon):
            row = {
                "id": series_id,
                "d_num": start_time + step,
                "horizon_step": step + 1,
            }
            if actual is not None:
                row["actual"] = float(actual[i, step])

            for j, q in enumerate(quantiles):
                row[f"q_{q}"] = float(pred[i, step, j])

            rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_df["point_forecast"] = pred_df[f"q_{0.5}"]
    return pred_df


def pinball_loss_np(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> np.ndarray:
    diff = y_true - y_pred
    return np.where(diff >= 0, q * diff, (1.0 - q) * (-diff))


def evaluate_quantiles(pred_df: pd.DataFrame, quantiles: Sequence[float] = QUANTILES) -> Dict[str, float]:
    if "actual" not in pred_df.columns:
        return {}

    y_true = pred_df["actual"].to_numpy(dtype="float32")
    metrics: Dict[str, float] = {}

    per_q = []
    for q in quantiles:
        y_pred = pred_df[f"q_{q}"].to_numpy(dtype="float32")
        pb = pinball_loss_np(y_true, y_pred, q).mean()
        metrics[f"pinball_q_{q}"] = float(pb)
        per_q.append(pb)

    metrics["mean_pinball"] = float(np.mean(per_q))
    metrics["mae_p50"] = float(np.abs(y_true - pred_df["point_forecast"].to_numpy(dtype="float32")).mean())

    if {f"q_{0.025}", f"q_{0.975}"} <= set(pred_df.columns):
        lo = pred_df[f"q_{0.025}"].to_numpy(dtype="float32")
        hi = pred_df[f"q_{0.975}"].to_numpy(dtype="float32")
        metrics["coverage_95"] = float(((y_true >= lo) & (y_true <= hi)).mean())

    return metrics


def run_single_experiment(config: TrainConfig) -> Dict[str, Any]:
    seed_everything(config.seed)
    hardware = get_hardware_config(config.extra_workers)

    splits = load_recommended_splits(config.data_root)
    if config.fold_name not in splits:
        raise KeyError(f"Unknown fold {config.fold_name!r}. Available: {', '.join(sorted(splits))}")

    fold = splits[config.fold_name]
    if fold.horizon != config.prediction_length:
        raise ValueError(
            f"Fold {fold.name} has horizon {fold.horizon}, but config.prediction_length={config.prediction_length}"
        )

    feature_spec = get_feature_spec(config.feature_set_name)

    run_dir = config.output_dir / config.feature_set_name / config.fold_name
    run_dir.mkdir(parents=True, exist_ok=True)

    data_frame, min_train_prediction_idx = load_fold_frame(
        data_root=config.data_root,
        fold=fold,
        feature_spec=feature_spec,
        encoder_length=config.encoder_length,
        prediction_length=config.prediction_length,
        train_window_count=config.train_window_count,
        max_series=config.max_series,
        seed=config.seed,
        add_engineered=True,
        series_ids_path=config.series_ids_path,
        sampling_strategy=config.sampling_strategy,
    )

    training, validation, train_loader, val_loader = build_datasets(
        df=data_frame,
        feature_spec=feature_spec,
        fold=fold,
        min_train_prediction_idx=min_train_prediction_idx,
        encoder_length=config.encoder_length,
        prediction_length=config.prediction_length,
        batch_size=config.batch_size,
        hardware=hardware,
    )

    trainer, model, checkpoint_callback = create_trainer_and_model(
        training=training,
        config=config,
        hardware=hardware,
        run_dir=run_dir,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    if not best_path:
        raise RuntimeError("No checkpoint was saved.")

    best_model = TemporalFusionTransformer.load_from_checkpoint(best_path)

    predict_kwargs = {
        "mode": "quantiles",
        "return_index": True,
        "trainer_kwargs": {
            "accelerator": hardware.accelerator,
            "devices": hardware.devices,
            "precision": hardware.precision,
            "logger": False,
            "enable_progress_bar": False,
        },
    }

    if not fold.is_holdout:
        predict_kwargs["return_y"] = True

    prediction_output = best_model.predict(val_loader, **predict_kwargs)

    pred_df = build_prediction_frame(
        prediction_output=prediction_output,
        quantiles=QUANTILES,
        enforce_monotonic_quantiles=config.enforce_monotonic_quantiles,
    )

    pred_path = run_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics: Dict[str, Any] = {
        "fold_name": fold.name,
        "feature_set_name": config.feature_set_name,
        "best_checkpoint": str(best_path),
        "n_training_windows": len(training),
        "n_validation_windows": len(validation),
        "n_series_in_prediction": int(pred_df["id"].nunique()),
        "train_window_count": config.train_window_count,
        "max_series": config.max_series,
        "encoder_length": config.encoder_length,
        "prediction_length": config.prediction_length,
        "batch_size": config.batch_size,
        "hidden_size": config.hidden_size,
        "attention_head_size": config.attention_head_size,
        "hidden_continuous_size": config.hidden_continuous_size,
        "lstm_layers": config.lstm_layers,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "precision": hardware.precision,
        "accelerator": hardware.accelerator,
        "best_val_loss": float(checkpoint_callback.best_model_score.cpu().item())
        if checkpoint_callback.best_model_score is not None
        else float("nan"),
        "prediction_path": str(pred_path),
    }
    metrics.update(evaluate_quantiles(pred_df, quantiles=QUANTILES))

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def config_to_dict(config: TrainConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["data_root"] = str(payload["data_root"])
    payload["output_dir"] = str(payload["output_dir"])
    if payload.get("series_ids_path") is not None:
        payload["series_ids_path"] = str(payload["series_ids_path"])
    return payload
