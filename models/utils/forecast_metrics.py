import numpy as np


def finalise_quantiles(q_preds):
    q_preds = np.clip(np.asarray(q_preds, dtype=np.float32), 0.0, None)
    return np.maximum.accumulate(q_preds, axis=-1)


def crps_from_quantiles(y, q_preds, quantiles):
    y_exp = y[:, :, np.newaxis]
    q = np.array(quantiles, dtype=np.float32)[np.newaxis, np.newaxis, :]
    diff = y_exp - q_preds
    loss = np.maximum(q * diff, (q - 1) * diff)
    weights = np.diff(np.concatenate(([0.0], np.array(quantiles, dtype=np.float32))))
    return float((loss * weights[np.newaxis, np.newaxis, :]).sum(axis=2).mean())


def _looks_probabilistic(q_preds, quantiles):
    median_idx = quantiles.index(0.5)
    median = q_preds[:, :, [median_idx]]
    return not np.allclose(q_preds, median, atol=1e-6, rtol=1e-6)


def compute_extra_forecast_metrics(model, preds_df, probabilistic_override=None):
    quantiles = list(model.QUANTILES)
    q_cols = [f"q{q}" for q in quantiles]
    forecast_start = int(model.TARGET_START)
    forecast_end = forecast_start + int(model.PRED_LENGTH) - 1

    future_truth = (
        model.test_raw[
            (model.test_raw["d_num"] >= forecast_start) &
            (model.test_raw["d_num"] <= forecast_end)
        ][["id", "d_num", "sales"]]
        .copy()
    )
    future_truth["day_ahead"] = future_truth["d_num"] - forecast_start + 1

    merged = preds_df.merge(
        future_truth[["id", "day_ahead", "sales"]],
        on=["id", "day_ahead"],
        how="inner",
    ).sort_values(["id", "day_ahead"]).reset_index(drop=True)

    series_ids = merged["id"].drop_duplicates().tolist()
    pred_q = merged[["id", "day_ahead"] + q_cols].pivot(index="id", columns="day_ahead", values=q_cols)
    q_preds = np.stack([
        pred_q[f"q{q}"].reindex(series_ids).values.astype(np.float32)
        for q in quantiles
    ], axis=2)

    targets = (
        merged[["id", "day_ahead", "sales"]]
        .pivot(index="id", columns="day_ahead", values="sales")
        .reindex(series_ids)
        .values.astype(np.float32)
    )
    preds = (
        merged[["id", "day_ahead", "q0.5"]]
        .pivot(index="id", columns="day_ahead", values="q0.5")
        .reindex(series_ids)
        .values.astype(np.float32)
    )

    preds = np.clip(preds, 0, 1e6)
    targets = np.clip(targets, 0, 1e6)
    q_preds = finalise_quantiles(q_preds)

    mask = targets > 0
    ss_res = float(((targets - preds) ** 2).sum())
    ss_tot = float(((targets - targets.mean()) ** 2).sum())
    per_series_mse = ((preds - targets) ** 2).mean(axis=1)
    per_series_mae = np.abs(preds - targets).mean(axis=1)

    weights = model.item_weights.reindex(series_ids).fillna(0.0).values.astype(np.float32)
    if weights.sum() > 0:
        weights /= weights.sum()

    metrics = {
        "rmse": float(np.sqrt(((preds - targets) ** 2).mean())),
        "mae": float(np.abs(preds - targets).mean()),
        "mape": float(np.abs((preds[mask] - targets[mask]) / targets[mask]).mean() * 100) if mask.any() else float("nan"),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "w_rmse": float(np.sqrt((weights * per_series_mse).sum())),
        "w_mae": float((weights * per_series_mae).sum()),
    }

    probabilistic = (
        _looks_probabilistic(q_preds, quantiles)
        if probabilistic_override is None
        else bool(probabilistic_override)
    )
    if probabilistic:
        idx_025 = quantiles.index(0.025)
        idx_975 = quantiles.index(0.975)
        lower = q_preds[:, :, idx_025]
        upper = q_preds[:, :, idx_975]
        metrics["coverage_95"] = float(((targets >= lower) & (targets <= upper)).mean())
        metrics["interval_width"] = float((upper - lower).mean())
        metrics["quantile_crps"] = crps_from_quantiles(targets, q_preds, quantiles)

    return metrics
