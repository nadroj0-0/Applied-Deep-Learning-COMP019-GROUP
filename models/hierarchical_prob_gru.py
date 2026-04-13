import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from .base_model import BaseModel

class M5TimeSeriesDataset(Dataset):
    def __init__(self, df, cat_mappings, weight_dict, seq_length=28):
        self.seq_length = seq_length
        num_items = df['id'].nunique()
        num_days = len(df) // num_items
        
        self.sales = torch.tensor(df['sales'].values.astype(np.float32).reshape(num_items, num_days))
        self.wday = torch.tensor(df['wday'].values.astype(np.float32).reshape(num_items, num_days))
        
        first_rows = df.iloc[0::num_days]
        static_list = []
        for col in ['state_id', 'store_id', 'cat_id', 'dept_id']:
            mapped_col = first_rows[col].map(cat_mappings[col]).fillna(0).astype(np.float32).values
            static_list.append(mapped_col)
        self.static_feats = torch.tensor(np.stack(static_list, axis=1)) 
        
        item_ids = first_rows['id'].values
        self.weights = torch.tensor([weight_dict.get(i_id, 0.0) for i_id in item_ids], dtype=torch.float32)
        
        self.num_items = num_items
        self.valid_starts_per_item = max(0, num_days - self.seq_length)
        
    def __len__(self):
        return self.num_items * self.valid_starts_per_item
        
    def __getitem__(self, idx):
        item_idx = idx // self.valid_starts_per_item
        t = idx % self.valid_starts_per_item
        sales_window = self.sales[item_idx, t : t + self.seq_length].unsqueeze(1)
        wday_window = self.wday[item_idx, t : t + self.seq_length].unsqueeze(1)
        static_window = self.static_feats[item_idx].unsqueeze(0).expand(self.seq_length, -1)
        X = torch.cat([sales_window, static_window, wday_window], dim=1) # Shape: (28, 6)
        y = self.sales[item_idx, t + self.seq_length]
        w = self.weights[item_idx]
        return X, y, w


def gaussian_nll_loss(mu, sigma, target, weights):
    variance = sigma ** 2
    loss = 0.5 * torch.log(2 * np.pi * variance) + ((target - mu) ** 2) / (2 * variance)
    return (loss * weights).mean()

class HierarchicalProbGRU(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        mu = self.mu_head(last).squeeze()
        sigma = torch.clamp(self.softplus(self.sigma_head(last)), min=1e-3, max=5.0).squeeze()
        return mu, sigma

class HierarchicalProbGRUModel(BaseModel):
    @property
    def model_name(self): return "hierarchical_prob_gru"

    def preprocess(self):
        cat_mappings = {col: {val: i for i, val in enumerate(self.train_raw[col].unique())} for col in ['state_id', 'store_id', 'cat_id', 'dept_id']}
        self.train_processed = M5TimeSeriesDataset(self.train_raw, cat_mappings, self.item_weights.to_dict())
        self.val_processed   = M5TimeSeriesDataset(self.val_raw, cat_mappings, self.item_weights.to_dict())
        self.test_processed  = M5TimeSeriesDataset(self.test_raw, cat_mappings, self.item_weights.to_dict())

    def train(self, epochs=5, lr=0.001, batch_size=4096):
        train_loader = DataLoader(self.train_processed, batch_size=batch_size, shuffle=True)
        self.model = HierarchicalProbGRU(input_dim=6).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        batch_size = 2048
        epochs = 1

        for epoch in range(epochs):
            self.model.train()
            for batch_X, batch_y, batch_w in train_loader:
                batch_X, batch_y, batch_w = batch_X.to(self.device), batch_y.to(self.device), batch_w.to(self.device)
                optimizer.zero_grad()
                mu, sigma = self.model(batch_X)
                loss = gaussian_nll_loss(mu, sigma, batch_y, batch_w)
                loss.backward()
                optimizer.step()

        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"{self.model_name}.ckpt"))

    def predict(self):
        self.model = HierarchicalProbGRU(input_dim=6).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, f"{self.model_name}.ckpt"), map_location=self.device))
        self.model.eval()
        
        test_loader = DataLoader(self.test_processed, batch_size=4096, shuffle=False)
        all_preds = []
        q_tensor = torch.tensor(self.QUANTILES, device=self.device)
        
        with torch.no_grad():
            for batch_X, _, _ in test_loader:
                mu, sigma = self.model(batch_X.to(self.device))
                dist = torch.distributions.Normal(mu, sigma)
                preds = torch.stack([dist.icdf(q) for q in q_tensor], dim=1) # Generates exact quantiles mathematically
                all_preds.append(preds.cpu().numpy())
                
        preds_array = np.clip(np.concatenate(all_preds), a_min=0, a_max=None)
        preds_array = np.sort(preds_array, axis=1) 
        
        q_cols = [f"q{q}" for q in self.QUANTILES]
        num_items = self.test_raw['id'].nunique()
        item_ids = self.test_raw.iloc[0::len(self.test_raw)//num_items]['id'].values
        
        preds_df = pd.DataFrame(preds_array, columns=q_cols)
        preds_df.insert(0, "id", np.repeat(item_ids, 28))
        preds_df.insert(1, "day_ahead", np.tile(np.arange(1, 29), num_items))
        
        out_path = os.path.join(self.output_dir, f"{self.model_name}_predictions.csv")
        preds_df.to_csv(out_path, index=False)
        return preds_df