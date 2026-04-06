from models.h_lstm import HierarchicalLSTMModel
from models.linear_model import LinearModel 

MODEL_REGISTRY = {
    "h_lstm": HierarchicalLSTMModel,
    "linear": LinearModel, 
}