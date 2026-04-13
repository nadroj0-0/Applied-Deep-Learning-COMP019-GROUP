from models.h_lstm import HierarchicalLSTMModel
from models.baseline_lstm import LSTMBaseline
from models.lightgbm_nn import LightGBM_NN
from models.tft_model import TFTModel 
from models.regular_gru import RegularGRUModel
from models.hierarchical_quantile_gru import HierarchicalQuantileGRUModel
from models.hierarchical_prob_gru import HierarchicalProbGRUModel

MODEL_REGISTRY = {
    "h_lstm": HierarchicalLSTMModel,
    "lstm_baseline": LSTMBaseline,
    "regular_gru": RegularGRUModel,
    "hierarchical_quantile_gru": HierarchicalQuantileGRUModel,
    "hierarchical_prob_gru": HierarchicalProbGRUModel,
    "tft": TFTModel,                
    "lgbm_baseline": LightGBM_NN,  
    "tft": TFTModel,   
}

