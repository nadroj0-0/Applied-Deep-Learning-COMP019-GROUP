import os
import random
import pickle
import json
import torch

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseModel(ABC):

    QUANTILES   = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
    PRED_LENGTH = 28
    SEED        = 25

    def __init__(self, data_dir="data", output_dir="outputs"):
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir   = data_dir
        self.output_dir = output_dir
        os.makedirs(data_dir,   exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        self.train_raw = self.val_raw = self.test_raw = None
        self.calendar  = self.prices  = self.item_weights = None
        self.train_processed = self.val_processed = self.test_processed = None
        self.model = None

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def preprocess(self): ...
    # Input:  self.train_raw, self.val_raw, self.test_raw
    # Output: self.train_processed, self.val_processed, self.test_processed
    #         (format is model-specific: tensors, lgb.Dataset, DataFrames, etc.)

    @abstractmethod
    def train(self): ...
    # Input:  self.train_processed, self.val_processed
    # Must:
    #   1. Set self.model
    #   2. Save weights → output_dir/{model_name}.pth (or .pkl for LGBM)

    def load_and_split_data(self):
        """
        Downloads (or loads from cache) M5 data and processes it into long-format dataframes (1 row per item and day). 
        Split into train (d_1-d_1773), val (d_1774-d_1885), and test (d_1886-d_1941).
        Save raw splits to cache for loading when training models.

        Output: self.train_raw, self.val_raw, self.test_raw, self.item_weights
        """
        # Step 1: load data
        cache = os.path.join(self.data_dir, "raw_split.pkl")

        if os.path.exists(cache):
            with open(cache, "rb") as f:
                d = pickle.load(f)
            print("Loaded cached data splits.")

        else:
            base     = "https://huggingface.co/datasets/kashif/M5/resolve/main"
            sales    = pd.read_csv(f"{base}/sales_train_evaluation.csv")
            calendar = pd.read_csv(f"{base}/calendar.csv")
            prices   = pd.read_csv(f"{base}/sell_prices.csv")
            print("Downloaded M5 data.")

            # Step 2: melt sales to long format
            id_cols  = [c for c in sales.columns if not c.startswith("d_")]
            day_cols = [c for c in sales.columns if c.startswith("d_")]
            sales_long = sales[id_cols + day_cols].melt(
                id_vars=id_cols, var_name='d', value_name='sales'
            )
            sales_long['d_num'] = sales_long['d'].str[2:].astype(int)

            # Step 3: merge calendar data and set dtypes
            for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
                calendar[col] = calendar[col].fillna('none').astype('category')
            calendar['weekday'] = calendar['weekday'].astype('category')
            calendar['date']    = pd.to_datetime(calendar['date'])
            for col in ['wday', 'month', 'year', 'snap_CA', 'snap_TX', 'snap_WI']:
                calendar[col] = calendar[col].astype('int8')
            sales_long = sales_long.merge(calendar, on='d', how='left')

            # Step 4: merge daily prices
            sales_long = sales_long.merge(
                prices[['store_id', 'item_id', 'wm_yr_wk', 'sell_price']],
                on=['store_id', 'item_id', 'wm_yr_wk'], how='left'
            )

            # Step 5: add is_available flag and forward-fill missing prices
            sales_long['is_available'] = sales_long['sell_price'].notna().astype('int8')
            sales_long = sales_long.sort_values(['id', 'd_num']).reset_index(drop=True)
            sales_long['sell_price'] = (
                sales_long.groupby('id')['sell_price']
                .transform(lambda x: x.ffill().fillna(0.0))
            )

            # Step 6: check missing values
            missing = sales_long.isnull().sum()
            missing = missing[missing > 0]
            assert len(missing) == 0, f"Missing values found:\n{missing}"

            # Step 7: split into train/val/test
            total_days = len(day_cols)
            train_end  = total_days - 6 * self.PRED_LENGTH   # 1773
            val_end    = total_days - 2 * self.PRED_LENGTH   # 1885

            train_raw = sales_long[sales_long['d_num'] <= train_end].reset_index(drop=True)
            val_raw   = sales_long[(sales_long['d_num'] > train_end) & (sales_long['d_num'] <= val_end)].reset_index(drop=True)
            test_raw  = sales_long[sales_long['d_num'] > val_end].reset_index(drop=True)

            assert train_raw['d_num'].nunique() == 1773
            assert val_raw['d_num'].nunique()   == 112
            assert test_raw['d_num'].nunique()  == 56
            print(f"Train: d_1-d_{train_end} | Val: d_{train_end+1}-d_{val_end} | Test: d_{val_end+1}-d_{total_days}")

            # Step 8: calculate revenue weights on last 28 training days only
            last28 = train_raw[train_raw['d_num'] > train_end - 28]
            train_rev = (last28['sales'] * last28['sell_price']).groupby(last28['id']).sum()
            item_weights = (train_rev / train_rev.sum()).rename('weight')

            # Step 9: save data splits to cache
            d = dict(train_raw=train_raw, val_raw=val_raw, test_raw=test_raw,
                        item_weights=item_weights)

            # save all splits and weights into raw_split.pkl
            with open(cache, "wb") as f:
                pickle.dump(d, f)
            print(f"Cached to {cache}")

        self.train_raw    = d["train_raw"]
        self.val_raw      = d["val_raw"]
        self.test_raw     = d["test_raw"]
        self.item_weights = d["item_weights"]
        print("Finished data processing.")

        return self.train_raw, self.val_raw, self.test_raw, self.item_weights

    def evaluate(self):           pass

    def run_training_pipeline(self):
    # Combines loading and splitting data, preprocessing, and training into a single pipeline
        self.load_and_split_data()
        self.preprocess()
        self.train()

    def run_inference_pipeline(self):
    # Combines loading processed test data and trained model, inference and evaluation into a single pipeline
        self.evaluate()