import os
import json
import pickle
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from base_model import BaseModel


warnings.filterwarnings("ignore")


class TFTModel(BaseModel):
    """
    Temporal Fusion Transformer model compatible with BaseModel.

    Key design:
    - Uses BaseModel.load_and_split_data() for shared M5 processing.
    - Builds TFT datasets from train/val/test raw splits.
    - Trains on d_1-d_1773.
    - Validates on the final 28 days of the val block:
        encoder context: d_1830-d_1857
        decoder target:  d_1858-d_1885
    - Predicts on the final 28 days of the test block:
        encoder context: d_1886-d_1913
        decoder target:  d_1914-d_1941
    """

    MAX_ENCODER_LENGTH = 28
    MAX_PREDICTION_LENGTH = 28

    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "outputs",
        batch_size: int = 128,
        num_workers: int = 2,
        max_epochs: int = 10,
        learning_rate: float = 1e-3,
        hidden_size: int = 32,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 16,
        gradient_clip_val: float = 0.1,
    ):
        super().__init__(data_dir=data_dir, output_dir=output_dir)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.gradient_clip_val = gradient_clip_val

        self.dataset_objects: Dict[str, Any] = {}
        self.dataloaders: Dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        return "tft"

    def _checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.model_name}.ckpt")

    def _config_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.model_name}_config.json")

    def _preprocess_cache_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.model_name}_preprocess.pkl")

    def _metrics_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.model_name}_train_metrics.json")

    def _prepare_full_dataframe(self) -> pd.DataFrame:
        """
        Combine train/val/test raw splits into one dataframe for consistent TFT encoding.
        """
        full_df = pd.concat(
            [self.train_raw, self.val_raw, self.test_raw],
            axis=0,
            ignore_index=True
        ).copy()

        full_df["series_id"] = full_df["id"].astype(str)
        full_df["time_idx"] = full_df["d_num"].astype(int)
        full_df["sales"] = full_df["sales"].astype(float)
        full_df["sell_price"] = full_df["sell_price"].astype(float)
        full_df["is_available"] = full_df["is_available"].astype(float)

        # Keep raw string categoricals for TFT
        categorical_cols = [
            "series_id",
            "item_id", "dept_id", "cat_id", "store_id", "state_id",
            "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"
        ]
        for col in categorical_cols:
            full_df[col] = full_df[col].astype(str)

        # Known reals
        known_real_cols = [
            "time_idx", "sell_price", "is_available",
            "snap_CA", "snap_TX", "snap_WI",
            "wday", "month", "year"
        ]
        for col in known_real_cols:
            full_df[col] = full_df[col].astype(float)

        full_df = full_df.sort_values(["series_id", "time_idx"]).reset_index(drop=True)
        return full_df

    def preprocess(self):
        """
        Build TFT TimeSeriesDataSet objects and dataloaders.

        We create:
        - training dataset using only train period (<= 1773)
        - validation prediction dataset for decoder horizon d_1858-d_1885
        - test prediction dataset for decoder horizon d_1914-d_1941
        """
        full_df = self._prepare_full_dataframe()

        train_end = int(self.train_raw["d_num"].max())      # 1773
        val_start = int(self.val_raw["d_num"].min())        # 1774
        val_end = int(self.val_raw["d_num"].max())          # 1885
        test_start = int(self.test_raw["d_num"].min())      # 1886
        test_end = int(self.test_raw["d_num"].max())        # 1941

        # Final validation target horizon inside val split:
        # encoder: 1830-1857, decoder: 1858-1885
        val_prediction_start = val_end - self.PRED_LENGTH + 1  # 1858

        # Final test target horizon:
        # encoder: 1886-1913, decoder: 1914-1941
        test_prediction_start = self.TARGET_START  # 1914

        training_cut_df = full_df[full_df["time_idx"] <= train_end].copy()

        training = TimeSeriesDataSet(
            training_cut_df,
            time_idx="time_idx",
            target="sales",
            group_ids=["series_id"],

            min_encoder_length=self.MAX_ENCODER_LENGTH,
            max_encoder_length=self.MAX_ENCODER_LENGTH,
            min_prediction_length=self.MAX_PREDICTION_LENGTH,
            max_prediction_length=self.MAX_PREDICTION_LENGTH,

            static_categoricals=[
                "item_id", "dept_id", "cat_id", "store_id", "state_id"
            ],

            time_varying_known_categoricals=[
                "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"
            ],

            time_varying_known_reals=[
                "time_idx", "sell_price", "is_available",
                "snap_CA", "snap_TX", "snap_WI",
                "wday", "month", "year"
            ],

            time_varying_unknown_reals=[
                "sales"
            ],

            target_normalizer=GroupNormalizer(
                groups=["series_id"],
                transformation="softplus"
            ),

            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # Validation dataframe must include enough context before 1858
        val_context_start = val_prediction_start - self.MAX_ENCODER_LENGTH  # 1830
        val_df = full_df[
            (full_df["time_idx"] >= val_context_start) &
            (full_df["time_idx"] <= val_end)
        ].copy()

        validation = TimeSeriesDataSet.from_dataset(
            training,
            val_df,
            predict=True,
            stop_randomization=True
        )

        # Test dataframe must include enough context before 1914
        test_context_start = test_prediction_start - self.MAX_ENCODER_LENGTH  # 1886
        test_df = full_df[
            (full_df["time_idx"] >= test_context_start) &
            (full_df["time_idx"] <= test_end)
        ].copy()

        test_dataset = TimeSeriesDataSet.from_dataset(
            training,
            test_df,
            predict=True,
            stop_randomization=True
        )

        train_dataloader = training.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        val_dataloader = validation.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        test_dataloader = test_dataset.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        self.train_processed = training
        self.val_processed = validation
        self.test_processed = test_dataset

        self.dataset_objects = {
            "training": training,
            "validation": validation,
            "test": test_dataset,
            "full_df": full_df,
            "val_prediction_start": val_prediction_start,
            "test_prediction_start": test_prediction_start,
        }

        self.dataloaders = {
            "train": train_dataloader,
            "val": val_dataloader,
            "test": test_dataloader,
        }

        preprocess_info = {
            "max_encoder_length": self.MAX_ENCODER_LENGTH,
            "max_prediction_length": self.MAX_PREDICTION_LENGTH,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
            "val_prediction_start": val_prediction_start,
            "test_prediction_start": test_prediction_start,
        }

        with open(self._preprocess_cache_path(), "wb") as f:
            pickle.dump(preprocess_info, f)

        with open(self._config_path(), "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "quantiles": self.QUANTILES,
                    "batch_size": self.batch_size,
                    "num_workers": self.num_workers,
                    "max_epochs": self.max_epochs,
                    "learning_rate": self.learning_rate,
                    "hidden_size": self.hidden_size,
                    "attention_head_size": self.attention_head_size,
                    "dropout": self.dropout,
                    "hidden_continuous_size": self.hidden_continuous_size,
                    "max_encoder_length": self.MAX_ENCODER_LENGTH,
                    "max_prediction_length": self.MAX_PREDICTION_LENGTH,
                },
                f,
                indent=2
            )

        print(f"Training samples:   {len(training)}")
        print(f"Validation samples: {len(validation)}")
        print(f"Test samples:       {len(test_dataset)}")

    def train(self):
        """
        Train TFT and save best checkpoint to outputs/tft.ckpt
        """
        assert self.train_processed is not None, "Run preprocess() first."
        assert self.val_processed is not None, "Run preprocess() first."

        training = self.train_processed
        train_dataloader = self.dataloaders["train"]
        val_dataloader = self.dataloaders["val"]

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=len(self.QUANTILES),
            loss=QuantileLoss(quantiles=self.QUANTILES),
            log_interval=10,
            reduce_on_plateau_patience=3,
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=3,
            verbose=True,
            mode="min"
        )

        lr_logger = LearningRateMonitor()

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir,
            filename=f"{self.model_name}-best-temp",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            gradient_clip_val=self.gradient_clip_val,
            callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
            log_every_n_steps=10,
        )

        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        best_model_path = checkpoint_callback.best_model_path
        if not best_model_path:
            raise RuntimeError("No best checkpoint was saved during training.")

        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        self.model = best_tft

        final_ckpt_path = self._checkpoint_path()
        if os.path.abspath(best_model_path) != os.path.abspath(final_ckpt_path):
            import shutil
            shutil.copy(best_model_path, final_ckpt_path)

        metrics = {
            "best_model_path": final_ckpt_path,
            "best_val_loss": float(checkpoint_callback.best_model_score.cpu().item())
            if checkpoint_callback.best_model_score is not None else None,
        }

        with open(self._metrics_path(), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Best checkpoint saved to: {final_ckpt_path}")
        print(f"Best val_loss: {metrics['best_val_loss']}")

    def predict(self) -> pd.DataFrame:
        """
        Load saved TFT checkpoint and generate 9 quantile predictions for d_1914-d_1941.
        """
        checkpoint_path = self._checkpoint_path()
        assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

        # Rebuild processed datasets if this is a fresh inference run
        if self.test_processed is None or "test" not in self.dataloaders:
            self.preprocess()

        test_dataloader = self.dataloaders["test"]

        best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        best_tft.eval()
        self.model = best_tft

        predictions = best_tft.predict(
            test_dataloader,
            mode="raw",
            return_x=True,
            return_index=True
        )

        raw_predictions = predictions.output
        x = predictions.x
        index = predictions.index

        pred_tensor = raw_predictions["prediction"].detach().cpu().numpy()
        actuals = x["decoder_target"].detach().cpu().numpy()

        rows = []
        batch_size, horizon, num_q = pred_tensor.shape

        expected_horizon = self.PRED_LENGTH
        expected_num_q = len(self.QUANTILES)

        assert horizon == expected_horizon, f"Expected horizon {expected_horizon}, got {horizon}"
        assert num_q == expected_num_q, f"Expected {expected_num_q} quantiles, got {num_q}"

        for i in range(batch_size):
            series_id = index.iloc[i]["series_id"]

            for t in range(horizon):
                row = {
                    "id": series_id,
                    "day_ahead": t + 1,
                }

                for j, q in enumerate(self.QUANTILES):
                    row[f"q{q}"] = float(pred_tensor[i, t, j])

                rows.append(row)

        preds_df = pd.DataFrame(rows)

        q_cols = [f"q{q}" for q in self.QUANTILES]

        preds_df[q_cols] = (
            preds_df[q_cols]
            .clip(lower=0)
            .apply(
                lambda row: np.maximum.accumulate(row.values),
                axis=1,
                result_type="expand"
            )
            .set_axis(q_cols, axis=1)
        )

        preds_df = (
            preds_df[["id", "day_ahead"] + q_cols]
            .sort_values(["id", "day_ahead"])
            .reset_index(drop=True)
        )

        out_path = os.path.join(self.output_dir, f"{self.model_name}_predictions.csv")
        preds_df.to_csv(out_path, index=False)
        print(f"Predictions saved to {out_path}")

        return preds_df

    def predict_ood_unseen(self) -> pd.DataFrame:
        """
        Run inference on CA_3 series excluded during subsampling.
        Maps each unseen series_id to a seen series from the same dept_id.
        """
        checkpoint_path = self._checkpoint_path()
        assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    
        cache = os.path.join(self.data_dir, "raw_split.pkl")
        with open(cache, "rb") as f:
            d = pickle.load(f)
    
        all_ca3_test = d["test_raw"].copy()
        seen_ids = set(self.test_raw["id"].unique())
        unseen_test = all_ca3_test[~all_ca3_test["id"].isin(seen_ids)].copy()
    
        assert len(unseen_test) > 0, "No unseen series found — was subsampling applied?"
        print(f"Unseen series count: {unseen_test['id'].nunique()}")
    
        # Build dept_id -> list of seen series_ids mapping
        seen_meta = self.train_raw.drop_duplicates("id")[["id", "dept_id"]]
        dept_to_seen = seen_meta.groupby("dept_id")["id"].apply(list).to_dict()
    
        # Map each unseen id to a random seen id from same dept
        unseen_meta = unseen_test.drop_duplicates("id")[["id", "dept_id"]]
        id_map = {}
        for _, row in unseen_meta.iterrows():
            candidates = dept_to_seen.get(row["dept_id"], [])
            assert len(candidates) > 0, f"No seen series for dept {row['dept_id']}"
            id_map[row["id"]] = np.random.choice(candidates)
    
        unseen_test["original_id"] = unseen_test["id"]
        unseen_test["id"] = unseen_test["id"].map(id_map)
        unseen_test["series_id"] = unseen_test["id"].astype(str)
    
        # Temporarily swap test_raw
        original_test_raw = self.test_raw
        self.test_raw = unseen_test.drop(columns=["original_id"])
    
        full_df = self._prepare_full_dataframe()
        full_df["series_id"] = full_df["id"].map(id_map).fillna(full_df["id"]).astype(str)
    
        test_prediction_start = self.TARGET_START
        test_context_start = test_prediction_start - self.MAX_ENCODER_LENGTH
        test_df = full_df[
            (full_df["time_idx"] >= test_context_start) &
            (full_df["time_idx"] <= full_df["time_idx"].max())
        ].copy()
    
        test_dataset = TimeSeriesDataSet.from_dataset(
            self.train_processed, test_df, predict=True, stop_randomization=True
        )
        test_dataloader = test_dataset.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=self.num_workers
        )
    
        best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        best_tft.eval()
    
        predictions = best_tft.predict(
            test_dataloader, mode="raw", return_x=True, return_index=True
        )
    
        pred_tensor = predictions.output["prediction"].detach().cpu().numpy()
        index = predictions.index
        reverse_map = {v: k for k, v in id_map.items()}
        q_cols = [f"q{q}" for q in self.QUANTILES]
    
        rows = []
        for i in range(pred_tensor.shape[0]):
            masked_id = index.iloc[i]["series_id"]
            original_id = reverse_map.get(masked_id, masked_id)
            for t in range(self.PRED_LENGTH):
                row = {"id": original_id, "day_ahead": t + 1}
                for j, q in enumerate(self.QUANTILES):
                    row[f"q{q}"] = float(pred_tensor[i, t, j])
                rows.append(row)
    
        preds_df = pd.DataFrame(rows)
        preds_df[q_cols] = (
            preds_df[q_cols]
            .clip(lower=0)
            .apply(lambda row: np.maximum.accumulate(row.values), axis=1, result_type="expand")
            .set_axis(q_cols, axis=1)
        )
        preds_df = preds_df[["id", "day_ahead"] + q_cols].sort_values(["id", "day_ahead"]).reset_index(drop=True)
    
        self.test_raw = original_test_raw
    
        out_path = os.path.join(self.output_dir, f"{self.model_name}_ood_unseen_predictions.csv")
        preds_df.to_csv(out_path, index=False)
        print(f"OOD unseen predictions saved to {out_path}")
    
        return preds_df
    
    def predict_ood(self, target_store: str) -> pd.DataFrame:
        """
        Run inference on a different store using CA_3-trained model.
        Replaces unseen series_ids with CA_3 equivalents at inference time,
        then restores original IDs in the output.
        """
        checkpoint_path = self._checkpoint_path()
        assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

        # Load full test data from cache
        cache = os.path.join(self.data_dir, "raw_split.pkl")
        with open(cache, "rb") as f:
            d = pickle.load(f)

        full_test = d["test_raw"]
        store_test = full_test[full_test["store_id"] == target_store].copy()

        # Build id mapping: TX_1 series_id -> CA_3 equivalent
        store_test["original_id"] = store_test["id"]
        store_test["id"] = store_test["id"].str.replace(
            f"_{target_store}_", "_CA_3_", regex=False
        )
        store_test["series_id"] = store_test["id"].astype(str)

        # Temporarily replace self.test_raw to reuse preprocess machinery
        original_test_raw = self.test_raw
        self.test_processed = None
        self.test_raw = store_test.drop(columns=["original_id"])

        # Rebuild only the test dataloader
        full_df = self._prepare_full_dataframe()
        full_df = full_df[full_df["store_id"] == target_store].copy()
        full_df["series_id"] = full_df["id"].str.replace(
            f"_{target_store}_", "_CA_3_", regex=False
        )

        test_prediction_start = self.TARGET_START
        test_context_start = test_prediction_start - self.MAX_ENCODER_LENGTH
        test_df = full_df[
            (full_df["time_idx"] >= test_context_start) &
            (full_df["time_idx"] <= full_df["time_idx"].max())
        ].copy()

        test_dataset = TimeSeriesDataSet.from_dataset(
            self.train_processed,
            test_df,
            predict=True,
            stop_randomization=True
        )
        test_dataloader = test_dataset.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=self.num_workers
        )

        # Run inference
        best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        best_tft.eval()

        predictions = best_tft.predict(
            test_dataloader, mode="raw", return_x=True, return_index=True
        )

        pred_tensor = predictions.output["prediction"].detach().cpu().numpy()
        index = predictions.index
        q_cols = [f"q{q}" for q in self.QUANTILES]

        rows = []
        for i in range(pred_tensor.shape[0]):
            masked_id = index.iloc[i]["series_id"]
            # Restore original store ID
            original_id = masked_id.replace("_CA_3_", f"_{target_store}_")
            for t in range(self.PRED_LENGTH):
                row = {"id": original_id, "day_ahead": t + 1}
                for j, q in enumerate(self.QUANTILES):
                    row[f"q{q}"] = float(pred_tensor[i, t, j])
                rows.append(row)

        preds_df = pd.DataFrame(rows)
        preds_df[q_cols] = (
            preds_df[q_cols]
            .clip(lower=0)
            .apply(lambda row: np.maximum.accumulate(row.values), axis=1, result_type="expand")
            .set_axis(q_cols, axis=1)
        )
        preds_df = preds_df[["id", "day_ahead"] + q_cols].sort_values(["id", "day_ahead"]).reset_index(drop=True)

        # Restore original test_raw
        self.test_raw = original_test_raw

        out_path = os.path.join(self.output_dir, f"{self.model_name}_ood_{target_store}_predictions.csv")
        preds_df.to_csv(out_path, index=False)
        print(f"OOD predictions saved to {out_path}")

        return preds_df