from models.h_lstm import HierarchicalLSTMModel
from models.baseline_lstm import LSTMBaseline
from models.linear_model import LinearModel 
from models.lightgbm_nn import LightGBM_NN
from models.tft_model import TFTModel  # <--- Add this

MODEL_REGISTRY = {
    "lgbm_baseline": LightGBM_NN,  # Reusing LSTM baseline for simplicity
    "h_lstm": HierarchicalLSTMModel,
    "lstm_baseline": LSTMBaseline,     # <--- Add this
    "linear": LinearModel, 
 #   "tft": TFTModel,                   # <--- Add this
}