# =============================================================================
# train.py  —  Deterministic GRU / LSTM baselines for M5 sales forecasting
# COMP0197 Applied Deep Learning
# =============================================================================
# GenAI Note: Scaffolded with Claude (Anthropic). Verified by authors.
# =============================================================================

import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.experiment import *
from utils.network import build_gru, build_lstm
from utils.training_strategies import gru_step
from utils.data import build_dataloaders_from_batches

import argparse #for debuggin remove later before submission

PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_CONFIG = {
    "seed":                     42,
    "epochs":                   50,
    "lr":                       1e-3,
    "hidden":                   128,
    "layers":                   2,
    "dropout":                  0.2,
    "batch_size":               256,
    "seq_len":                  28,
    "horizon":                  28,
    "store_id":                 "CA_3",
    "data_dir":                 "./data",
    "max_series":               None,
    "num_workers":              2,
    "early_stopping_patience":  10,
    "early_stopping_min_delta": 0.001,
}

SEARCH = True

GRU_SEARCH_SPACE = {
    "lr":      (1e-4, 1e-2, "log"),
    "hidden":  (64, 256, "uniform"),
    "layers":  (1, 4, "uniform"),
    "dropout": (0.1, 0.4, "uniform"),
}

LSTM_SEARCH_SPACE = {
    "lr":      (1e-4, 1e-2, "log"),
    "hidden":  (64, 256, "uniform"),
    "layers":  (1, 4, "uniform"),
    "dropout": (0.1, 0.4, "uniform"),
}

HYPER_PARAM_INIT_MODELS = 20
HYPER_PARAM_SEARCH_SCHEDULE = [
    {"epochs": 10,  "keep": math.ceil(HYPER_PARAM_INIT_MODELS / 2)},
    {"epochs": 10, "keep": math.ceil(HYPER_PARAM_INIT_MODELS / 4)},
    {"epochs": 20, "keep": 1},
]

# DEBUG — parse_args allows CLI overrides for quick testing
# e.g. python train.py --max_series 10 --epochs 2 --num_workers 0
# REMOVE parse_args() and uncomment comment out bit below before submission
def parse_args():
    p = argparse.ArgumentParser(description="Train GRU/LSTM on M5")
    for k, v in TRAIN_CONFIG.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        elif v is None:
            p.add_argument(f"--{k}", type=str, default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    return vars(p.parse_args())


def main():
    cfg = parse_args()  # REMOVE before submission and uncomment the commented out bit below

    # try:
    #     cfg = TRAIN_CONFIG.copy()
    # except NameError:
    #     raise RuntimeError(
    #         "TRAIN_CONFIG must be defined before calling main(). "
    #         "It defines the experiment hyperparameters."
    #     )

    # shared data kwargs passed to build_dataloaders_from_batches
    data_kwargs = dict(
        data_dir    = cfg["data_dir"],
        seq_len     = cfg["seq_len"],
        horizon     = cfg["horizon"],
        batch_size  = cfg["batch_size"],
        store_id    = cfg["store_id"],
        max_series  = cfg["max_series"],
        num_workers = cfg["num_workers"],
        seed        = cfg["seed"],
    )

    experiments = {
        "gru_deterministic": dict(
            builder        = build_gru,
            training_step  = gru_step,
            search_space   = GRU_SEARCH_SPACE if SEARCH else None,
        ),
        "lstm_deterministic": dict(
            builder        = build_lstm,
            training_step  = gru_step,
            search_space   = LSTM_SEARCH_SPACE if SEARCH else None,
        ),
        # "gru_probabilistic": dict(
        #     builder        = build_prob_gru,
        #     training_step  = prob_gru_step,
        #     search_space   = GRU_SEARCH_SPACE if SEARCH else None,
        # ),
        # "tft": dict(
        #     builder        = build_tft,
        #     training_step  = tft_step,
        #     search_space   = TFT_SEARCH_SPACE if SEARCH else None,
        # ),
    }

    for name, kwargs in experiments.items():
        exp = Experiment(name, cfg, model_dir=get_model_dir(name, PROJECT_DIR))
        exp.run(
            data_fn=build_dataloaders_from_batches,
            schedule=HYPER_PARAM_SEARCH_SCHEDULE,
            initial_models=HYPER_PARAM_INIT_MODELS,
            **data_kwargs,
            **kwargs,
        )
        save_json(exp.stats, exp.model_dir / "normalisation_stats.json")


if __name__ == "__main__":
    main()