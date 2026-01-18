import torch
import numpy as np
import fastf1
import pandas as pd
import json
from transformer import F1Transformer
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG = {
    "model_path": "final_f1_model.pth",
    "features_config_path": "feature_list.json",
    "model_params": {
        # The winning parameters from your Optuna search
        "d_model": 64,
        "nhead": 8,
        "num_layers": 5,
        "dropout": 0.1195
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sequence_length": 10
}

def load_artifacts(config: dict) -> tuple[F1Transformer, list]:
    """Loads the trained model and the feature list."""
    logging.info("Loading model and feature configuration...")
    with open(config["features_config_path"], 'r') as f:
        feature_cols = json.load(f)
    
    input_dim = len(feature_cols)
    model = F1Transformer(input_dim=input_dim, **config["model_params"]).to(config["device"])
    model.load_state_dict(torch.load(config["model_path"]))
    model.eval()
    
    logging.info("Trained model and features loaded successfully.")
    return model, feature_cols

def get_prediction_data(year: int, race_name: str, seq_length: int) -> pd.DataFrame:
    """Fetches and processes the first N laps of a race for prediction."""
    logging.info(f"Fetching data for {year} {race_name}...")
    session = fastf1.get_session(year, race_name, 'R')
    session.load(laps=True, weather=True)
    laps = session.laps.copy()
    weather = session.weather_data.copy()
    
    laps_to_process = laps.loc[laps['LapNumber'] <= seq_length].copy()

    # --- Mirror the production data engineering pipeline ---
    laps_to_process['Time'] = pd.to_timedelta(laps_to_process['Time'])
    weather['Time'] = pd.to_timedelta(weather['Time'])
    merged_laps = pd.merge_asof(laps_to_process.sort_values('Time'), weather.sort_values('Time'), on='Time', direction='nearest')
    
    merged_laps['LapTimeSeconds'] = pd.to_timedelta(merged_laps['LapTime'], errors='coerce').dt.total_seconds()
    merged_laps['TempDifference'] = merged_laps['TrackTemp'] - merged_laps['AirTemp']
    
    merged_laps.sort_values(['LapNumber', 'Position'], inplace=True)
    merged_laps['GapToCarAhead'] = merged_laps.groupby('LapNumber')['Time'].diff().dt.total_seconds()
    
    return merged_laps, session.results

def main(year: int, race_name: str):
    """Main function to run the prediction."""
    model, feature_cols = load_artifacts(CONFIG)
    live_data, session_results = get_prediction_data(year, race_name, CONFIG["sequence_length"])
    
    all_driver_sequences, driver_numbers = [], []
    for driver in live_data['DriverNumber'].unique():
        driver_laps = live_data[live_data['DriverNumber'] == driver].copy()
        
        # Fill missing data robustly
        for col in feature_cols:
            if col not in driver_laps.columns: driver_laps[col] = np.nan
        driver_laps[feature_cols] = driver_laps[feature_cols].ffill().bfill().fillna(0)

        if len(driver_laps) >= CONFIG["sequence_length"]:
            sequence = driver_laps[feature_cols].tail(CONFIG["sequence_length"]).values
            all_driver_sequences.append(sequence)
            driver_numbers.append(driver)
            
    if not all_driver_sequences:
        logging.warning("Could not generate any valid sequences from the provided race data.")
        return

    # Make predictions
    X_pred = np.array(all_driver_sequences, dtype=np.float32)
    sequences_tensor = torch.tensor(X_pred).to(CONFIG["device"])
    with torch.no_grad():
        predictions = model(sequences_tensor)

    # Display results
    results_df = pd.DataFrame({
        'DriverNumber': driver_numbers,
        'PredictedPosition': predictions.cpu().numpy().flatten()
    }).sort_values('PredictedPosition')
    
    driver_info = session_results[['DriverNumber', 'FullName']].drop_duplicates()
    final_results = pd.merge(results_df, driver_info, on='DriverNumber')

    print("\n--- Predicted Race Outcome ---")
    print(final_results[['FullName', 'PredictedPosition']].round(2).to_string(index=False))

if __name__ == "__main__":
    # Example: predict 
    main(year=2025, race_name="Mexican Grand Prix")