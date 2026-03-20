from pathlib import Path
import pandas as pd
import json 

SPLIT_NAME = "fold_2" # choose: "fold_1", "fold_2", or "holdout"

base_dir = Path("m5_processed_validation_ca3")
metadata_path = Path("m5_processed_validation") / "metadata" / "validation_recommended_splits.json"
series_info_path = Path("m5_processed_validation") / "validation_series_info_ca3.pkl.gz"

# Load split metadata
with open(metadata_path, "r", encoding="utf-8") as f:
    split_registry = json.load(f)

if SPLIT_NAME not in split_registry:
    raise KeyError(f"Split '{SPLIT_NAME}' not found. Available: {list(split_registry.keys())}")

split_cfg = split_registry[SPLIT_NAME]
print("Using split:", SPLIT_NAME)
print(split_cfg)


#helper for split
def split_dataset(df, name, out_dir, split_cfg, split_name):
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    future_df = pd.DataFrame()

    if "train" in split_cfg:
        train_start, train_end = split_cfg["train"]
        train_df = df[
            (df["sales_observed"] == 1) &
            (df["d_num"] >= train_start) &
            (df["d_num"] <= train_end)
        ].copy()

    if "valid" in split_cfg:
        valid_start, valid_end = split_cfg["valid"]
        valid_df = df[
            (df["sales_observed"] == 1) &
            (df["d_num"] >= valid_start) &
            (df["d_num"] <= valid_end)
        ].copy()

    if "predict" in split_cfg:
        pred_start, pred_end = split_cfg["predict"]
        future_df = df[
            (df["is_future"] == 1) &
            (df["d_num"] >= pred_start) &
            (df["d_num"] <= pred_end)
        ].copy()
    else:
        # For fold_1 / fold_2, keep all future rows if present
        future_df = df[df["is_future"] == 1].copy()

    train_df.to_pickle(out_dir / f"{name}_{split_name}_train.pkl.gz", compression="gzip")

    if len(valid_df) > 0:
        valid_df.to_pickle(out_dir / f"{name}_{split_name}_valid.pkl.gz", compression="gzip")

    if len(future_df) > 0:
        future_df.to_pickle(out_dir / f"{name}_{split_name}_future.pkl.gz", compression="gzip")

    print(f"\n{name} [{split_name}]")
    print("  train :", train_df.shape)
    print("  valid :", valid_df.shape)
    print("  future:", future_df.shape)


series_info = pd.read_pickle(series_info_path, compression = "gzip")
print("Loaded series info:", series_info.shape)

# Load non empty ca3 batches
batch_files = sorted(base_dir.glob("validation_features_batch*.pkl.gz"))

dfs = []
for batch_path in batch_files:
    print(f"Loading {batch_path.name}...")
    batch_df = pd.read_pickle(batch_path, compression = "gzip")
    if len(batch_df) == 0:
        print(" Empty batch, skipping.")
        continue
    dfs.append(batch_df)
# Concatenate all batches
full_df = pd.concat(dfs, ignore_index=True)
print("Concatenated full dataframe shape:", full_df.shape)

# Merge with series info to get store_id and item_id
feat = full_df.merge(series_info[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "id_idx", "item_idx", 
                                  "dept_idx", "cat_idx", "store_idx", "state_idx"]], on="id", how="left")

print("Merged features with series info:", feat.shape)
#Sort data
feat = feat.sort_values(["id", "d_num"]).reset_index(drop=True)

#Save final dataset
feat.to_pickle(base_dir / "ca3_all_features.pkl.gz", compression = "gzip")
print("Saved final CA3 dataset:", feat.shape)

# define lstm columns
lstm_columns = ["id", "d_num", "date", "sales", "sales_observed", "is_future", 
"item_idx", "dept_idx", "cat_idx", "store_idx", "state_idx",
    "sell_price", "snap",
    "wday", "month", "week_of_year", "is_weekend", "is_event_day",
    "event_name_1_code", "event_type_1_code", "event_name_2_code", "event_type_2_code", "sales_lag_7", "sales_lag_28", 
    "sales_roll_mean_7", "sales_roll_mean_28",
     "sales_roll_std_28"
]
lstm_columns = [col for col in lstm_columns if col in feat.columns]
lstm_df = feat[lstm_columns].copy()
lstm_df.to_pickle(base_dir / f"ca3_lstm_{SPLIT_NAME}.pkl.gz", compression="gzip")
split_dataset(lstm_df, "ca3_lstm", base_dir, split_cfg, SPLIT_NAME)

#DeepAR columns
deepar_columns = ["id", "d_num", "date", "sales", "sales_observed", "is_future",
"item_idx", "dept_idx", "cat_idx", "store_idx", "state_idx",
    "sell_price", "snap", "wday", "month", "week_of_year", "quarter", "is_weekend", "is_event_day",
    "event_name_1_code", "event_type_1_code", "event_name_2_code", "event_type_2_code", "month_sin", "month_cos", "wday_sin"  , "wday_cos"]

deepar_columns = [col for col in deepar_columns if col in feat.columns]
deepar_df = feat[deepar_columns].copy()
deepar_df.to_pickle(base_dir / f"ca3_deepar_{SPLIT_NAME}.pkl.gz", compression="gzip")
split_dataset(deepar_df, "ca3_deepar", base_dir, split_cfg, SPLIT_NAME)

#TFT columns
tft_columns = ["id", "d_num", "date", "sales", "sales_observed", "is_future",
"item_idx", "dept_idx", "cat_idx", "store_idx", "state_idx", "days_since_release", "age_since_first_sale",
    "sell_price", "price_change_1w", "price_pct_change_1w", "price_rel_4w","price_rel_13w",
    "price_rel_52w", "price_rank_dept_store",
    "snap", "wday", "month", "year", "week_of_year", "quarter", "day_of_month", "day_of_year",
    "is_weekend", "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end", "is_year_start",
    "is_year_end",  "is_event_day",
    "event_name_1_code", "event_type_1_code", "event_name_2_code", "event_type_2_code", "month_sin", "month_cos", "wday_sin"  , "wday_cos",
    "sales_lag_7", "sales_lag_28", "sales_roll_mean_7", "sales_roll_mean_28", "sales_roll_std_28", "store_sales_lag_7", "store_sales_lag_28",
    "store_sales_roll_mean_7", "store_sales_roll_mean_28", "state_sales_lag_7", "state_sales_lag_28", "state_sales_roll_mean_7",
    "state_sales_roll_mean_28", "cat_store_sales_lag_7", "cat_store_sales_lag_28", "cat_store_sales_roll_mean_7","cat_store_sales_roll_mean_28",
    "dept_store_sales_lag_7", "dept_store_sales_lag_28", "dept_store_sales_roll_mean_7",  "dept_store_sales_roll_mean_28"
]
tft_columns = [col for col in tft_columns if col in feat.columns]
tft_df = feat[tft_columns].copy()
tft_df.to_pickle(base_dir / f"ca3_tft_{SPLIT_NAME}.pkl.gz", compression="gzip")
split_dataset(tft_df, "ca3_tft", base_dir, split_cfg, SPLIT_NAME)
print(f"\nDone building and splitting CA_3 model datasets using {SPLIT_NAME}.")



