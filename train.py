# train.py (Updated)

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformer import F1Transformer

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG = {
    # --- REMOVED: data paths are now loaded from the manifest ---
    "model_params": {
        "d_model": 64, "nhead": 8, "num_layers": 5, "dropout": 0.1195
    },
    "training": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32, "num_epochs": 100, "learning_rate": 0.000305,
        "output_model_path": "final_f1_model.pth"
    },
    "features_config_path": "feature_list.json",
    "manifest_path": "data_manifest.json" # --- ADDED: Path to the manifest
}

# --- Data Loading (Updated) ---
def load_data(config: dict) -> tuple[DataLoader, DataLoader, int, int]:
    """Loads data paths from manifest and creates PyTorch DataLoaders."""
    logging.info("Loading and preparing data...")
    
    # --- ADDED: Load paths from the manifest file ---
    with open(config["manifest_path"], 'r') as f:
        data_paths = json.load(f)
    logging.info(f"Loaded data paths from manifest: {data_paths}")
    x_path = data_paths["x_path"]
    y_path = data_paths["y_path"]
    # --- END ADDITION ---

    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True).reshape(-1, 1)

    logging.info(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=config["training"]["batch_size"]
    )
    input_dim = X.shape[2]
    return train_loader, val_loader, len(y_val), input_dim

# --- Training and Validation (No changes) ---
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for sequences, targets in tqdm(dataloader, desc="Training"):
        sequences, targets = sequences.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = loss_fn(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device, val_len):
    model.eval()
    total_loss, total_mae = 0.0, 0.0
    with torch.no_grad():
        for sequences, targets in tqdm(dataloader, desc="Validating"):
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            total_loss += loss_fn(outputs, targets).item()
            total_mae += torch.abs(outputs - targets).sum().item()
    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae / val_len
    return avg_loss, avg_mae

# --- Training Orchestration (No changes) ---
def run_training():
    with open(CONFIG["features_config_path"], "r") as f:
        feature_cols = json.load(f)
    logging.info(f"Loaded feature list ({len(feature_cols)} features): {feature_cols}")
    train_loader, val_loader, val_len, input_dim_from_data = load_data(CONFIG)
    if input_dim_from_data != len(feature_cols):
        logging.error(
            f"FATAL MISMATCH! feature_list.json = {len(feature_cols)} features, "
            f"but data = {input_dim_from_data} features. Exiting."
        )
        return # Exit if mismatch still occurs
    else:
        logging.info("✅ Feature count matches between data and JSON configuration.")
    device = CONFIG["training"]["device"]
    logging.info(f"Using device: {device}")
    logging.info(f"Model input dimension automatically set to: {input_dim_from_data}")
    model = F1Transformer(input_dim=input_dim_from_data, **CONFIG["model_params"]).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    best_val_mae = float('inf')
    logging.info("Starting training...")
    for epoch in range(CONFIG["training"]["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss, val_mae = validate(model, val_loader, loss_function, device, val_len)
        logging.info(
            f"Epoch {epoch + 1}/{CONFIG['training']['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}"
        )
        scheduler.step(val_mae)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), CONFIG["training"]["output_model_path"])
            logging.info(f"🎯 New best model saved! MAE: {best_val_mae:.4f}")
    logging.info(f"Training complete! Best model saved to {CONFIG['training']['output_model_path']}")

if __name__ == "__main__":
    run_training()