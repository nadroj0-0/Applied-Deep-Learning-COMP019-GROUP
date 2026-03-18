import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image
import json
import math
import time
from utils.common import evaluate_model


class NoisyDataset(Dataset):
    """
    Dataset wrapper that adds Gaussian noise to images.
    Used to evaluate robustness to input perturbations.
    """
    def __init__(self, dataset, noise_std=0.1):
        self.dataset = dataset
        self.noise_std = noise_std
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        noise = torch.randn_like(x) * self.noise_std
        x_noisy = x + noise
        # keep values in normalised range
        x_noisy = torch.clamp(x_noisy, -1.0, 1.0)
        return x_noisy, y


def build_noisy_test_loader(test_dataset, batch_size, noise_std=0.1):
    noisy_dataset = NoisyDataset(test_dataset, noise_std)
    return DataLoader(noisy_dataset, batch_size=batch_size, shuffle=False)


def save_mixup_demo(mixup_fn, dataset, save_path, alpha=0.4, device="cpu"):
    """
    Create a 4x4 grid of MixUp images to demonstrate the augmentation.
    """
    import numpy as np
    samples = torch.stack([dataset[i][0] for i in range(16)]).to(device)
    labels = torch.tensor([dataset[i][1] for i in range(16)]).to(device)
    mixed, _, _, _ = mixup_fn(samples, labels, alpha)
    # denormalise CIFAR10 for visualisation
    mixed = (mixed * 0.5) + 0.5
    mixed = mixed.clamp(0, 1)
    grid = torch.zeros(3, 32 * 4, 32 * 4)
    idx = 0
    for r in range(4):
        for c in range(4):
            grid[:, r*32:(r+1)*32, c*32:(c+1)*32] = mixed[idx]
            idx += 1
    img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)
    print(f"MixUp demo saved to: {save_path}")

def evaluate_noise_robustness(model, test_dataset, batch_size, save_path, noise_levels=None):
    """
    Evaluate model accuracy across a range of Gaussian noise levels.

    Args:
        model        (torch.nn.Module): Trained model in eval mode.
        test_dataset:                   Clean test dataset.
        batch_size   (int):             Batch size for evaluation.
        noise_levels (list[float]):     Noise std values to test.

    Returns:
        dict: Mapping of noise_std (float) -> accuracy (float).
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    criterion = nn.CrossEntropyLoss()
    results = {}
    print("\n--- Noise Robustness ---")
    for std in noise_levels:
        loader = build_noisy_test_loader(test_dataset, batch_size, noise_std=std)
        _, acc = evaluate_model(loader, model, criterion)
        print(f"  noise_std={std:.2f}  accuracy={acc:.4f}")
        results[std] = acc
    # save noise results to JSON
    payload = {
        "stage": "noise_test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "noise_levels": noise_levels,
            "accuracy_by_noise": {str(k): v for k, v in results.items()}
        }
    }
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Noise robustness results saved to: {save_path}")
    return results