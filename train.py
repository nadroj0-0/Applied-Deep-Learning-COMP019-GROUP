# =============================================================================
# train.py  —  Deterministic GRU / LSTM baselines for M5 sales forecasting
# COMP0197 Applied Deep Learning
# =============================================================================
# GenAI Note: Scaffolded with Claude (Anthropic). Verified by authors.
# =============================================================================

import sys
import json
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))
import torch
from utils.common  import device, train_model, full_train, init_optimiser, init_loss, save_model, save_json, rmse, mae
from utils.training_strategies import gru_step
import torch.nn as nn
from utils.network import SalesGRU, SalesLSTM
from utils.network import build_gru, build_lstm

DEFAULTS = dict(
    data_dir    = "./data",   # folder containing M5 CSVs or m5_processed_validation/
    model       = "gru",      # "gru" or "lstm"
    use_batches = True,       # True  -> load from pickle batches (recommended)
                              # False -> load from raw CSVs (legacy fallback)
    store_id    = "CA_3",     # store to train on (batched mode only)
    seq_len     = 28,         # lookback window (days)
    horizon     = 28,         # forecast horizon — matches M5 eval window
    batch_size  = 256,
    epochs      = 50,
    lr          = 1e-3,
    hidden      = 128,
    layers      = 2,
    dropout     = 0.2,
    max_series  = None,       # set to e.g. 50 for a quick debug run
    num_workers = 2,
    seed        = 42,
    out_dir     = "outputs",
    early_stopping_patience  = 10,
    early_stopping_min_delta = 0.001,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train deterministic GRU/LSTM on M5")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        else:
            p.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    return vars(p.parse_args())


def main():
    cfg = parse_args()
    OUT_DIR = Path(cfg["out_dir"])
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"  Model       : {cfg['model'].upper()}")
    print(f"  Data source : {'pickle batches' if cfg['use_batches'] else 'raw CSVs'}")
    if cfg["use_batches"]:
        print(f"  Store       : {cfg['store_id']}")
    print(f"  Seq len     : {cfg['seq_len']}  |  Horizon : {cfg['horizon']}")
    print(f"  Epochs      : {cfg['epochs']}   |  LR      : {cfg['lr']}")
    print(f"  Hidden      : {cfg['hidden']}   |  Layers  : {cfg['layers']}")
    print(f"  Device      : {device}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data — switch between batched and raw CSV pipeline
    # ------------------------------------------------------------------
    if cfg["use_batches"]:
        from utils.data import build_dataloaders_from_batches, N_BATCH_FEATURES
        train_loader, val_loader, test_loader, stats = build_dataloaders_from_batches(
            data_dir    = cfg["data_dir"],
            seq_len     = cfg["seq_len"],
            horizon     = cfg["horizon"],
            batch_size  = cfg["batch_size"],
            store_id    = cfg["store_id"],
            max_series  = cfg["max_series"],
            num_workers = cfg["num_workers"],
            seed        = cfg["seed"],
        )
        input_size = N_BATCH_FEATURES
    else:
        from utils.data import build_dataloaders, N_FEATURES
        train_loader, val_loader, test_loader, stats = build_dataloaders(
            data_dir    = cfg["data_dir"],
            seq_len     = cfg["seq_len"],
            horizon     = cfg["horizon"],
            batch_size  = cfg["batch_size"],
            max_series  = cfg["max_series"],
            num_workers = cfg["num_workers"],
            seed        = cfg["seed"],
        )
        input_size = N_FEATURES

    # # ------------------------------------------------------------------
    # # 2. Model
    # # ------------------------------------------------------------------
    # model_cls = SalesGRU if cfg["model"] == "gru" else SalesLSTM
    # model = model_cls(
    #     input_size  = input_size,
    #     hidden_size = cfg["hidden"],
    #     num_layers  = cfg["layers"],
    #     dropout     = cfg["dropout"],
    #     horizon     = cfg["horizon"],
    # ).to(device)
    #
    # print(model)
    # print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    #
    # # ------------------------------------------------------------------
    # # 3. Train
    # # ------------------------------------------------------------------
    # criterion, _ = init_loss(model, train_loader)
    # optimiser = init_optimiser(model, "Adam", lr=cfg["lr"])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimiser, patience=5, factor=0.5
    # )
    # history = train_model(
    #     cfg["epochs"], train_loader, val_loader,
    #     model, criterion, optimiser,
    #     training_step=gru_step,
    #     early_stopping_patience=cfg["early_stopping_patience"],
    #     early_stopping_min_delta=cfg["early_stopping_min_delta"],
    #     scheduler=scheduler,
    #     clip_grad_norm=1.0,
    #     extra_metrics={"val_rmse": rmse, "val_mae": mae},
    # )
    # ------------------------------------------------------------------
    # 2 + 3. Build and train
    # ------------------------------------------------------------------
    builder = build_gru if cfg["model"] == "gru" else build_lstm
    model, history, _, _ = full_train(
        name=cfg["model"],
        builder=builder,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["epochs"],
        model_dir=OUT_DIR,
        training_step=gru_step,
    )
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    # # ------------------------------------------------------------------
    # # 4. Save
    # # ------------------------------------------------------------------
    # save_model(model, cfg["model"], OUT_DIR)
    # save_json(history, OUT_DIR / f"{cfg['model']}_history.json")
    # save_json(stats,   OUT_DIR / "normalisation_stats.json")
    # save_json(cfg,     OUT_DIR / f"{cfg['model']}_config.json")
    #
    # print(f"\nAll outputs saved to: {OUT_DIR}/")
    # ------------------------------------------------------------------
    # 4. Save remaining outputs
    # ------------------------------------------------------------------
    save_json(stats, OUT_DIR / "normalisation_stats.json")
    save_json(cfg, OUT_DIR / f"{cfg['model']}_config.json")


if __name__ == "__main__":
    main()