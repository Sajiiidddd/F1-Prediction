import torch
import torch.nn as nn
import numpy as np
import math
import optuna
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import logging
import sys

# --- Configuration ---
# Optuna is verbose, so we can tune down some logging
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_PATH = 'sequences_X.npy'  # Use your latest data file
Y_PATH = 'targets_y.npy'
EPOCHS_PER_TRIAL = 15 # Number of epochs to train each trial
N_TRIALS = 50 # Number of different architectures to test

# --- Data Loading Function ---
def load_and_prepare_data(batch_size):
    """Loads and prepares data for a single trial."""
    X = np.load(X_PATH, allow_pickle=True)
    y = np.load(Y_PATH, allow_pickle=True).reshape(-1, 1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)
    
    return train_loader, val_loader, len(y_val)

# --- Model Definitions ---
# (These are the same blueprints as before)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class F1Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[:, -1, :]
        output = self.fc_out(output)
        return output

# --- Optuna Objective Function ---
def objective(trial):
    """This function is called by Optuna for each trial."""
    # 1. Suggest Hyperparameters
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Constraint: nhead must divide d_model
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned()

    # 2. Setup Model and Data
    train_loader, val_loader, val_len = load_and_prepare_data(batch_size)
    input_dim = train_loader.dataset.tensors[0].shape[2]
    
    model = F1Transformer(input_dim, d_model, nhead, num_layers, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    # 3. Training and Validation Loop
    best_trial_mae = float('inf')
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
                outputs = model(sequences)
                val_mae += torch.abs(outputs - targets).sum().item()
        
        avg_val_mae = val_mae / val_len
        
        # Report progress to Optuna for pruning
        trial.report(avg_val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if avg_val_mae < best_trial_mae:
            best_trial_mae = avg_val_mae
            
    return best_trial_mae # Return the final best MAE for this trial

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting hyperparameter search on device: {DEVICE}")
    # Create a study to minimize the MAE
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    # Start the optimization
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Print the results
    print("\nStudy statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Best MAE): {trial.value:.4f}")
    
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")