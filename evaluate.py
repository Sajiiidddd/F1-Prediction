import torch
import numpy as np
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformer import F1Transformer  # Assumes transformer.py exists

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG = {
    "model_params": {
        "d_model": 64, "nhead": 8, "num_layers": 5, "dropout": 0.1195
    },
    "training": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 128,  # Can be larger for evaluation
        # --- NOTE: Ensure this path is correct (v1 or v2) ---
        "output_model_path": "final_f1_model.pth"
    },
    "features_config_path": "feature_list.json",
    "manifest_path": "data_manifest.json"
}

# --- Data Loading ---
def load_test_data(config: dict) -> tuple[DataLoader, int]:
    """Loads and prepares the test dataset using the manifest."""
    logging.info("Loading test data...")
    
    with open(config["manifest_path"], 'r') as f:
        data_paths = json.load(f)
    logging.info(f"Loaded data paths from manifest: {data_paths}")
    
    X = np.load(data_paths["x_path"], allow_pickle=True)
    
    # --- FIX: Changed --1 to -1 ---
    y = np.load(data_paths["y_path"], allow_pickle=True).reshape(-1, 1)

    # CRITICAL: Use the exact same split as in train.py to get the test set
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info(f"Test data loaded. X shape: {X_val.shape}, y shape: {y_val.shape}")

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=config["training"]["batch_size"]
    )
    
    return val_loader, len(y_val)

# --- Model Loading ---
def load_model(config: dict) -> tuple[torch.nn.Module, str]:
    """Loads the trained model architecture and weights."""
    logging.info("Loading model...")
    
    with open(config["features_config_path"], "r") as f:
        feature_cols = json.load(f)
    input_dim = len(feature_cols)
    logging.info(f"Model input dimension set to: {input_dim}")

    device = config["training"]["device"]
    model = F1Transformer(input_dim=input_dim, **config["model_params"]).to(device)
    
    model_path = config["training"]["output_model_path"]
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        logging.error(f"FATAL: Model file not found at {model_path}")
        logging.error("Please run train.py to train and save the model first.")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading model state_dict: {e}")
        logging.error("Ensure model parameters in evaluate.py match train.py.")
        exit(1)
        
    model.eval()  # Set model to evaluation mode
    logging.info(f"Model loaded from {model_path} and set to eval mode on {device}.")
    return model, device

# --- Evaluation ---
def run_evaluation(model: torch.nn.Module, dataloader: DataLoader, device: str, test_len: int):
    """Runs the model over the test set and computes metrics."""
    logging.info("Starting evaluation...")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sequences, targets in tqdm(dataloader, desc="Evaluating"):
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    # --- Calculate Metrics ---
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    rounded_preds = np.round(all_preds)
    exact_accuracy = accuracy_score(all_targets, rounded_preds)
    pred_podium = (rounded_preds <= 3)
    actual_podium = (all_targets <= 3)
    podium_accuracy = accuracy_score(actual_podium, pred_podium)
    
    # --- Log Results ---
    logging.info("--- Evaluation Complete ---")
    logging.info(f"Total Test Samples: {test_len}")
    logging.info("---")
    logging.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logging.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    logging.info("---")
    logging.info(f"Exact Position Accuracy: {exact_accuracy*100:.2f}%")
    logging.info(f"Podium Accuracy (Top 3): {podium_accuracy*100:.2f}%")
    logging.info("---------------------------")


# --- Main Execution ---
def main():
    model, device = load_model(CONFIG)
    test_loader, test_len = load_test_data(CONFIG)
    run_evaluation(model, test_loader, device, test_len)

if __name__ == "__main__":
    main()



