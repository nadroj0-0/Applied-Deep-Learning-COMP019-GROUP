import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gru_step(model, inputs, labels, criterion, **kwargs):
    """
    Standard GRU training step for regression.
    Returns MSE loss and predictions.
    """
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    return loss, outputs

gru_step.valid_train_accuracy = False