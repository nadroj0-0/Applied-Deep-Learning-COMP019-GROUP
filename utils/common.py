from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import json
from .early_stopping import EarlyStopping
from .data import *
from .training_strategies import *
from .network import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def init_model(images, dropout_prob=0.0):
    # Create the model
    print('\nCreating model...')
    model = SalesGRU(dropout_prob).to(device)
    print(model)
    # Test a forward pass
    print('\nTesting forward pass...')
    images = images.to(device)
    outputs = model(images)
    print('Model output shape:', outputs.shape)
    return model, outputs


def init_optimiser(model, method, **kwargs):
    import inspect
    # Optimiser
    print('\nCreating optimiser...')
    # optimMethod = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    if not hasattr(optim, method):
        raise ValueError(f'Optimizer {method} not found in torch.optim')
    optimiser_class = getattr(optim, method)
    try:
        optim_method = optimiser_class(model.parameters(), **kwargs)
    except TypeError as e:
        expected_signature = inspect.signature(optimiser_class)
        print(f"\nInvalid arguments for optimizer '{method}'")
        print('Expected constructor signature:')
        print(f'{method}{expected_signature}')
        raise TypeError(
            f"Invalid arguments for optimizer '{method}'."
            f'Expected signature: {method}{expected_signature}'
        )
    #optim_method = optimiser_class(model.parameters(), lr=learn_rate, momentum=momentum_par)
    print('Optimiser created:', optim_method)
    return optim_method


def evaluate_model(data_loader, model, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    average_loss = total_loss / total_samples
    accuracy = None
    return average_loss, accuracy

def train_model(epochs, train_loader, val_loader, model, criterion, optim_method,
                training_step=gru_step,  early_stopping_patience=None,early_stopping_min_delta=0.0, **kwargs):
    # Training
    print('\nStarting training...')
    # num_epochs = 50
    history: dict = {'epoch_metrics': [],'early_stopping': None,'batch_losses': []}
    #batch_losses = []
    #epoch_losses = []
    accuracy_valid = getattr(training_step, "valid_train_accuracy", True)
    early_stopping_enabled = (
            early_stopping_patience is not None and early_stopping_patience > 0
    )
    early_stopper = EarlyStopping(early_stopping_patience,min_delta=early_stopping_min_delta) if early_stopping_enabled else None
    for epoch in range(epochs):
        model.train()
        #epoch_loss = 0
        #num_batches = 0
        epoch_train_loss_sum = 0.0
        epoch_train_correct = 0
        epoch_train_samples = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optim_method.zero_grad()
            #outputs = model(inputs)
            #loss = criterion(outputs, labels)
            loss, outputs = training_step(model,inputs,labels,criterion,**kwargs)
            loss.backward()
            optim_method.step()
            batch_size = labels.size(0)
            loss_value = loss.item()
            history['batch_losses'].append({
                'epoch': epoch+1,
                'batch': i+1,
                'loss': loss_value
            })
            #epoch_loss += loss_value
            #num_batches += 1
            epoch_train_loss_sum += loss_value * batch_size
            if accuracy_valid:
                predictions = outputs.argmax(dim=1)
                epoch_train_correct += (predictions == labels).sum().item()
            epoch_train_samples += batch_size
        #avg_epoch_loss = epoch_loss / num_batches
        #epoch_losses.append(avg_epoch_loss)
        train_loss = epoch_train_loss_sum / epoch_train_samples
        if accuracy_valid:
            train_accuracy = epoch_train_correct / epoch_train_samples
        val_loss, val_accuracy = evaluate_model(val_loader, model, criterion)
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'validation_loss': val_loss,
            'validation_accuracy': val_accuracy
        }
        if accuracy_valid:
            epoch_record['train_accuracy'] = train_accuracy
        history['epoch_metrics'].append(epoch_record)
        if early_stopper:
            stop = early_stopper.update(val_loss, model, epoch + 1)
            if stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                early_stopper.triggered = True
                break
        if accuracy_valid:
            print(
                f"Epoch {epoch + 1:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={train_accuracy:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_accuracy:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_accuracy:.4f}"
            )
        #print(f'Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}')
    print('Training finished.')
    if early_stopper and early_stopper.stopped_epoch is None:
        early_stopper.stopped_epoch = epochs
    best_val_accuracy = None
    if early_stopper and early_stopper.best_model_state is not None:
        model.load_state_dict(early_stopper.best_model_state)
        print("Restored best model from early stopping.")
        print(f"Best validation loss {early_stopper.best_val_loss:.4f} at epoch {early_stopper.best_epoch}")
        for m in history["epoch_metrics"]:
            if m["epoch"] == early_stopper.best_epoch:
                best_val_accuracy = m["validation_accuracy"]
                break
    if early_stopping_enabled:
        history["early_stopping"] = {
            "enabled": True,
            "triggered": early_stopper.triggered,
            "patience": early_stopper.patience,
            "min_delta": early_stopper.min_delta,
            "best_epoch": early_stopper.best_epoch,
            "best_validation_loss": early_stopper.best_val_loss,
            "best_validation_accuracy": best_val_accuracy,
            "stopped_epoch": early_stopper.stopped_epoch
        }
    #return batch_losses, epoch_losses
    return history

def save_model(model, name, model_dir):
    """
    Saves trained PyTorch model inside 'models' dir. If dir doesn't exist dir created
    Parameters
    model : torch.nn.Module - Trained model to save
    name : str - Name for file
    Returns
    path : Path - Full path to saved file
    """
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f'{name}_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to: {model_path}')
    return model_path

def save_history(history, name, stage, model, model_dir, config=None):
    model_dir.mkdir(exist_ok=True)
    history_path = model_dir / f'{name}_{stage}_history.json'
    payload = {
        "model": name,
        "architecture": str(model),
        "stage": stage,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config or {},
        "metrics": history
    }
    with open(history_path, 'w') as f:
        json.dump(payload, f, indent=4)
    print(f'History saved to: {history_path}')
    return history_path

def full_train(name, images, labels, train_loader, val_loader, method, epochs, model_dir,
               config=None, dropout_prob=0.0, training_step=gru_step, save_outputs=True,
               session=None,**kwargs):
    from utils.training_session import create_training_session
    start_time = time.time()
    if session is None:
        session = create_training_session(images, labels, method, dropout_prob, config, training_step, **kwargs)
    session.train(epochs,train_loader,val_loader)
    model_path , history_path = None, None
    if save_outputs:
        model_path = save_model(session.model, name, model_dir)
        history_path = save_history(session.history, name, "train", session.model, model_dir, config=config)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n{name} training completed in {elapsed:.2f} seconds")
    return session.model, session.history, model_path, history_path

def load_history(path: Path) -> dict:
    """
    Load a JSON history file saved by save_history() in utils/common.py.

    Args:
        path (Path): Path to the JSON history file.

    Returns:
        dict: Full JSON payload including metrics.
    """
    with open(path) as f:
        return json.load(f)


def extract_epoch_metrics(history: dict):
    """
    Pull per-epoch train/val accuracy from a loaded history dict.

    Args:
        history (dict): Loaded JSON history dict from load_history().

    Returns:
        tuple:
            epochs     (list[int])   — epoch numbers
            train_acc  (list[float]) — training accuracy per epoch
            val_acc    (list[float]) — validation accuracy per epoch
    """
    metrics   = history["metrics"]["epoch_metrics"]
    epochs    = [m["epoch"]               for m in metrics]
    train_acc = [m.get("train_accuracy")      for m in metrics]
    val_acc   = [m["validation_accuracy"] for m in metrics]
    return epochs, train_acc, val_acc


def load_model(dropout_prob: float, weights_path: Path) -> torch.nn.Module:
    """
    Instantiate a CNN and load saved weights from a .pt file.

    Args:
        dropout_prob  (float): Dropout probability used when the model was trained.
        weights_path  (Path):  Path to the saved state dict (.pt file).

    Returns:
        torch.nn.Module: Model with loaded weights in eval mode, on device.
    """
    model = SalesGRU(dropout=dropout_prob).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def evaluate_test_set(model, test_loader):
    """
    Evaluate a trained model on the test dataset.
    Args:
        model (torch.nn.Module)
        test_loader (DataLoader)
    Returns:
        dict containing test_loss and test_accuracy
    """
    criterion = nn.MSELoss()
    test_loss, test_acc = evaluate_model(test_loader, model, criterion)
    print("\nTest performance")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc={test_acc:.4f}")
    return {"test_loss": test_loss, "test_accuracy": test_acc}


def save_json(data, path):
    """
    Save dictionary as formatted JSON.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()

def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean(torch.abs(preds - targets)).item()


def train_gru(model, train_loader, val_loader, epochs=50, lr=1e-3):
    """
    Train the GRU model and return full history.

    Args:
        model       (SalesGRU):   Instantiated model.
        train_loader (DataLoader): Training data.
        val_loader   (DataLoader): Validation data.
        epochs       (int):        Number of training epochs.
        lr           (float):      Learning rate.

    Returns:
        dict: Training history with per-epoch metrics.
    """
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5)

    history = {"epoch_metrics": []}
    start = time.time()

    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss_sum = 0.0
        train_samples  = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            preds = model(x)
            loss  = criterion(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss_sum += loss.item() * len(y)
            train_samples  += len(y)

        train_loss = train_loss_sum / train_samples

        # --- validate ---
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_preds.append(model(x))
                val_targets.append(y)
        val_preds   = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        val_loss = criterion(val_preds, val_targets).item()
        val_rmse = rmse(val_preds, val_targets)
        val_mae  = mae(val_preds, val_targets)

        scheduler.step(val_loss)

        record = {
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_rmse":   val_rmse,
            "val_mae":    val_mae,
        }
        history["epoch_metrics"].append(record)
        print(
            f"Epoch {epoch+1:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_rmse={val_rmse:.4f} | "
            f"val_mae={val_mae:.4f}"
        )

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")
    return history
