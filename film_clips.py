# film_clips.py (Updated)

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

CONFIG = {
    "input_path": "final_features_data.parquet",
    "x_output_path": f"sequences_X_{timestamp}.npy",
    "y_output_path": f"targets_y_{timestamp}.npy",
    "features_config_path": "feature_list.json",
    "manifest_path": "data_manifest.json", 
    "sequence_length": 10,
    "feature_cols": [
        'LapNumber', 'Position', 'LapTimeSeconds', 'TyreLife', 'Stint',
        'AirTemp', 'TrackTemp', 'Rainfall', 'WindSpeed', 'TempDifference',
        'GapToCarAhead',
        # --- ADDED: Include all new telemetry features ---
        'TopSpeed', 
        'AvgSpeed', 
        'ThrottleApplication', 
        'BrakeUsage', 
        'DRS_Enabled_Pct'
    ]
}

# --- Data Loading (No changes) ---
def load_data(path: str) -> pd.DataFrame:
    logging.info(f"📂 Loading feature-engineered data from {path}...")
    return pd.read_parquet(path)

# --- Target Calculation (No changes) ---
def calculate_targets(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("🎯 Calculating final positions (targets)...")
    final_positions = df.loc[df.groupby(['Year', 'RaceName', 'DriverNumber'])['LapNumber'].idxmax()]
    return final_positions[['Year', 'RaceName', 'DriverNumber', 'Position']].rename(columns={'Position': 'FinalPosition'})

# --- Sequence Generation (No changes) ---
def generate_sequences(df: pd.DataFrame, targets_df: pd.DataFrame, feature_cols: list, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    logging.info(f"🔄 Generating sequences of length {seq_length} using {len(feature_cols)} features...")
    sequences, targets = [], []
    grouped = df.groupby(['Year', 'RaceName', 'DriverNumber'])
    for name, group in grouped:
        target_info = targets_df[
            (targets_df['Year'] == name[0]) &
            (targets_df['RaceName'] == name[1]) &
            (targets_df['DriverNumber'] == name[2])
        ]
        if not target_info.empty and len(group) >= seq_length:
            target = target_info['FinalPosition'].iloc[0]
            for i in range(len(group) - seq_length + 1):
                sequences.append(group.iloc[i:i+seq_length][feature_cols].values)
                targets.append(target)
    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    logging.info(f"✅ Generated {len(X)} sequences with shape {X.shape} (Samples, SeqLen, Features)")
    return X, y

# --- Save Artifacts (Updated) ---
def save_artifacts(X: np.ndarray, y: np.ndarray, config: dict):
    logging.info("💾 Saving sequence arrays and feature configuration...")
    np.save(config["x_output_path"], X)
    np.save(config["y_output_path"], y)
    with open(config["features_config_path"], 'w') as f:
        json.dump(config["feature_cols"], f, indent=2)

    # --- ADDED: Create and save the manifest file ---
    manifest = {
        "x_path": config["x_output_path"],
        "y_path": config["y_output_path"]
    }
    with open(config["manifest_path"], 'w') as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"📝 Saved data manifest to {config['manifest_path']}")
    # --- END ADDITION ---

    logging.info(f"🗂️ Data saved: {config['x_output_path']}, {config['y_output_path']}")

# --- Main Execution ---
def main():
    df_processed = load_data(CONFIG["input_path"])
    df_targets = calculate_targets(df_processed)
    X, y = generate_sequences(df_processed, df_targets, CONFIG["feature_cols"], CONFIG["sequence_length"])
    save_artifacts(X, y, CONFIG)

if __name__ == "__main__":
    main()