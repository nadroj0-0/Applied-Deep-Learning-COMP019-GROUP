import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from base_model import BaseModel


class LightGBM(BaseModel):

    @property
    def model_name(self):
        return "lightgbm"

    def preprocess(self):
        """
        Builds tabular features from long-format raw splits.
        Lags (1,7,14), rolling means (7,28), cyclical dow, trend, and next-day target.
        Early-row NaNs filled with 0. Rows with no target (last day per item) dropped.
        Sets: self.train_processed, self.val_processed, self.test_processed.
        """
        def _build_features(df):
            df = df.sort_values(["id", "d_num"]).copy()
            g = df.groupby("id")["sales"]
            # past sales at fixed offsets. Removed lag 28 to avoid losing too many rows
            for lag in [1, 7, 14]:   
                df[f"lag_{lag}"] = g.shift(lag)
            # smooth out noise and capture the short-term trend
            for w in [7, 28]:
                df[f"roll_mean_{w}"] = g.shift(1).rolling(w).mean() 
            # encode Mon-Sun as a cycle so that day 7 is close to day 1 
            df["dow_sin"] = np.sin(2 * np.pi * df["wday"] / 7).astype(np.float32)   
            df["dow_cos"] = np.cos(2 * np.pi * df["wday"] / 7).astype(np.float32)
            # raw day number lets the model learn long-term trends
            df["trend"]   = df["d_num"].astype(np.float32)
            # drop any row with missing target (incomplete windows)
            df["target"]  = g.shift(-1)   
            feat_cols = ["lag_1","lag_7","lag_14","roll_mean_7","roll_mean_28"]
            # early rows where lags/rolling means are unavailable are filled wth 0 
            df[feat_cols] = df[feat_cols].fillna(0)
            return df.dropna(subset=["target"]).reset_index(drop=True)

        self.train_processed = _build_features(self.train_raw)
        self.val_processed   = _build_features(self.val_raw)
        self.test_processed  = _build_features(self.test_raw)

    def train(self):
        """
        Trains one LightGBM quantile model per quantile (9 total).
        Each model optimises pinball loss at its quantile level.
        Saves models as .txt files to output_dir.
        Sets: self.models.
        """
        FEAT_COLS = [
            "lag_1","lag_7","lag_14",
            "roll_mean_7","roll_mean_28",
            "dow_sin","dow_cos","trend",
            "wday","month","sell_price","is_available"
        ]
        X = self.train_processed[FEAT_COLS].values.astype(np.float32)
        y = self.train_processed["target"].values.astype(np.float32)

        self.models  = {}
        total        = len(self.QUANTILES)
        train_start  = time.time()

        for i, q in enumerate(self.QUANTILES):
            q_start = time.time()
            dtrain  = lgb.Dataset(X, y)
            params  = {
                "objective": "quantile", "alpha": q,
                "learning_rate": 0.05, "num_leaves": 31, "verbose": -1
            }
            self.models[q] = lgb.train(params, dtrain, num_boost_round=100)
            self.models[q].save_model(
                os.path.join(self.output_dir, f"{self.model_name}_q{q}.txt")
            )
            elapsed      = time.time() - q_start
            total_so_far = time.time() - train_start
            eta          = (total_so_far / (i + 1)) * (total - i - 1)
            print(f"[{i+1}/{total}] q={q} | {elapsed/60:.1f} min | ETA {eta/60:.1f} min")

        print(f"Training complete in {(time.time() - train_start)/60:.1f} min")

    def predict(self):
        """
        Loads saved models and autoregressively forecasts 28 days per item.
        Uses d_1886-d_1913 as context. Median prediction fed back into rolling buffer at each step.
        Post-processes: clips negatives, enforces monotone quantiles.
        Saves predictions to output_dir/{model_name}_predictions.csv.
        Returns: preds_df (30490*28 rows) with columns id | day_ahead | q0.025 ... q0.975.
        """
        FEAT_COLS = [
        "lag_1","lag_7","lag_14",
        "roll_mean_7","roll_mean_28",
        "dow_sin","dow_cos","trend",
        "wday","month","sell_price","is_available"
    ]
        # load 9 saved models from disk
        models = {
            q: lgb.Booster(model_file=os.path.join(self.output_dir, f"{self.model_name}_q{q}.txt"))
            for q in self.QUANTILES
        }

        ctx = self.test_processed[
            (self.test_processed["d_num"] >= 1886) &
            (self.test_processed["d_num"] <= 1913)
        ].copy()

        rows = []
        for item_id, item_ctx in ctx.groupby("id"):
            item_ctx = item_ctx.sort_values("d_num")
            buf      = item_ctx["sales"].values[-28:].tolist()
            last     = item_ctx.iloc[-1]

            # Autoregressively forecast 28 steps per item
            for h in range(1, 29):
                dow = int((last["wday"] + h - 1) % 7)
                # build a feature vector from a rolling sales buffer
                x = np.array([[
                    buf[-1], buf[-7], buf[-14],
                    np.mean(buf[-7:]), np.mean(buf),
                    np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
                    last["d_num"] + h,
                    dow, last["month"], last["sell_price"], last["is_available"]
                ]], dtype=np.float32)

                rec = {"id": item_id, "day_ahead": h}
                preds_step = {}
                # predict all 9 quantiles
                for q in self.QUANTILES:
                    p = max(0.0, float(models[q].predict(x)[0]))
                    preds_step[q] = p
                    rec[f"q{q}"] = p

                # push the median back into the buffer
                buf.append(preds_step[0.5])
                rows.append(rec)

        # Post-processing
        preds_df = pd.DataFrame(rows).sort_values(["id","day_ahead"]).reset_index(drop=True)
        q_cols = [f"q{q}" for q in self.QUANTILES]
        # clip negative predictions to 0
        preds_df[q_cols] = preds_df[q_cols].clip(lower=0)
        # enforce non-decreasing quantiles
        preds_df[q_cols] = np.maximum.accumulate(preds_df[q_cols].values, axis=1)

        out = os.path.join(self.output_dir, f"{self.model_name}_predictions.csv")
        preds_df.to_csv(out, index=False)
        print(f"Saved → {out}")
        return preds_df