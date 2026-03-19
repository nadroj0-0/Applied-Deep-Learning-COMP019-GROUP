import torch.nn as nn
import torch


from utils.data import N_FEATURES


class SalesGRU(nn.Module):
    """
    Multi-layer GRU for deterministic sales point prediction.

    Takes a sequence of historical sales (and optionally extra features)
    and predicts the next timestep's sales as a single scalar.

    Args:
        input_size  (int): Number of features per timestep (1 if sales only).
        hidden_size (int): Number of units in each GRU layer.
        num_layers  (int): Number of stacked GRU layers.
        dropout     (float): Dropout probability between GRU layers.
    """
    def __init__(self, input_size=N_FEATURES, hidden_size=128, num_layers=2, dropout=0.2, horizon=28):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # input shape: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon)            # single scalar output
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # take only the final timestep's hidden state
        last = out[:, -1, :]            # shape: (batch, hidden_size)
        return self.head(last)  # shape: (batch,)


def build_gru(cfg):
    """
    Builder for deterministic GRU baseline.
    Returns (model, criterion, optimiser, training_kwargs).
    """
    from utils.data import N_BATCH_FEATURES
    from utils.common import device, rmse, mae, mape, r2
    cfg["layers"] = int(cfg["layers"])
    cfg["hidden"] = int(cfg["hidden"])
    model = SalesGRU(
        input_size=N_BATCH_FEATURES,
        hidden_size=int(cfg["hidden"]),
        num_layers=int(cfg["layers"]),
        dropout=cfg["dropout"],
        horizon=int(cfg["horizon"]),
    ).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    training_kwargs = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=5, factor=0.5
        ),
        "clip_grad_norm": 1.0,
        "extra_metrics": {"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2},
    }
    return model, criterion, optimiser, training_kwargs




class SalesLSTM(nn.Module):
    """
    Multi-layer LSTM for deterministic multi-step sales forecasting.

    Identical interface to SalesGRU — swap in/out as a drop-in baseline.

    Input  : (batch, seq_len, N_FEATURES)
    Output : (batch, horizon)

    Args
    ----
    input_size  : Number of features per timestep. Defaults to N_FEATURES (13).
    hidden_size : LSTM hidden units per layer.
    num_layers  : Number of stacked LSTM layers.
    dropout     : Dropout probability between LSTM layers (ignored if num_layers=1).
    horizon     : Number of future timesteps to predict.
    """

    def __init__(
        self,
        input_size:  int   = N_FEATURES,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
        horizon:     int   = 28,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last   = out[:, -1, :]        # (batch, hidden_size)
        return self.head(last)        # (batch, horizon)


def build_lstm(cfg):
    """
    Builder for deterministic LSTM baseline.
    Returns (model, criterion, optimiser, training_kwargs).
    """
    from utils.data import N_BATCH_FEATURES
    from utils.common import device, rmse, mae, mape, r2
    cfg["layers"] = int(cfg["layers"])
    cfg["hidden"] = int(cfg["hidden"])
    model = SalesLSTM(
        input_size=N_BATCH_FEATURES,
        hidden_size=int(cfg["hidden"]),
        num_layers=int(cfg["layers"]),
        dropout=cfg["dropout"],
        horizon=int(cfg["horizon"]),
    ).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    training_kwargs = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=5, factor=0.5
        ),
        "clip_grad_norm": 1.0,
        "extra_metrics": {"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2},
    }
    return model, criterion, optimiser, training_kwargs

class ProbGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
            # nn.Softplus()  # ensures mu > 0
        )
        # self.alpha_head = nn.Sequential(
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
            nn.Softplus()  # ensures alpha > 0
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        # return self.mu_head(last), self.alpha_head(last)
        return self.mu_head(last), self.sigma_head(last) + 1e-3

def build_prob_gru(cfg):
    from utils.data import N_BATCH_FEATURES
    from utils.common import device, rmse, mae, mape, r2, gaussian_nll_loss #, nb_nll_loss
    model = ProbGRU(
        input_size  = N_BATCH_FEATURES,
        hidden_size = int(cfg["hidden"]),
        num_layers  = int(cfg["layers"]),
        dropout     = cfg["dropout"],
        horizon     = int(cfg["horizon"]),
    ).to(device)
    # criterion = nb_nll_loss
    criterion = gaussian_nll_loss
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    training_kwargs = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=5, factor=0.5
        ),
        "clip_grad_norm": 0.5,
        "extra_metrics": {"val_rmse": rmse, "val_mae": mae},
        "sigma_reg": cfg.get("sigma_reg", 0.0),
    }
    return model, criterion, optimiser, training_kwargs

class ProbLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
            nn.Softplus()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.mu_head(last), self.sigma_head(last) + 1e-3


def build_prob_lstm(cfg):
    from utils.data import N_BATCH_FEATURES
    from utils.common import device, rmse, mae, gaussian_nll_loss
    model = ProbLSTM(
        input_size  = N_BATCH_FEATURES,
        hidden_size = int(cfg["hidden"]),
        num_layers  = int(cfg["layers"]),
        dropout     = cfg["dropout"],
        horizon     = int(cfg["horizon"]),
    ).to(device)
    criterion = gaussian_nll_loss
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    training_kwargs = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=5, factor=0.5
        ),
        "clip_grad_norm": 0.5,
        "extra_metrics": {"val_rmse": rmse, "val_mae": mae},
        "sigma_reg": cfg.get("sigma_reg", 0.0),
    }
    return model, criterion, optimiser, training_kwargs