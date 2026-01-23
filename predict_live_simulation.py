import torch
import numpy as np
import fastf1
import pandas as pd
import json
import logging
import time
from tqdm import tqdm
from transformer import F1Transformer # Assumes transformer.py is present

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
fastf1.Cache.enable_cache('cache')

CONFIG = {
    "model_path": "final_f1_model.pth",
    "features_config_path": "feature_list.json",
    "model_params": {
        "d_model": 64, "nhead": 8, "num_layers": 5, "dropout": 0.1195
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sequence_length": 10,
    # --- Simulation Target ---
    "sim_year": 2024,
    "sim_race": "Melbourne",
    "sim_delay_seconds": 0.5 # Delay between lap predictions
}

# --- Artifact Loading (Same as predict.py) ---
def load_artifacts(config: dict) -> tuple[F1Transformer, list]:
    """Loads the trained model and the feature list."""
    logging.info("Loading model and feature configuration...")
    with open(config["features_config_path"], 'r') as f:
        feature_cols = json.load(f)
    
    input_dim = len(feature_cols)
    model = F1Transformer(input_dim=input_dim, **config["model_params"]).to(config["device"])
    
    try:
        model.load_state_dict(torch.load(config["model_path"]))
    except FileNotFoundError:
        logging.error(f"FATAL: Model file not found at {config['model_path']}")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading model: {e}. Check model_params.")
        exit(1)
        
    model.eval()
    logging.info("Trained model and features loaded successfully.")
    return model, feature_cols

# --- Telemetry Engineering (FIX for predict.py) ---
# This logic is copied from engineer_telemetry.py and is
# necessary for the model to work.
def get_aggregated_telemetry(lap):
    """Calculates aggregated telemetry features for a single lap."""
    features = {
        'TopSpeed': np.nan, 'AvgSpeed': np.nan,
        'ThrottleApplication': np.nan, 'BrakeUsage': np.nan,
        'DRS_Enabled_Pct': np.nan
    }
    try:
        telemetry = lap.get_car_data().add_distance()
        if not telemetry.empty:
            if 'Speed' in telemetry.columns:
                features['TopSpeed'] = telemetry['Speed'].max()
                features['AvgSpeed'] = telemetry['Speed'].mean()
            if 'Throttle' in telemetry.columns:
                features['ThrottleApplication'] = (telemetry['Throttle'] >= 99).mean()
            if 'Brake' in telemetry.columns:
                features['BrakeUsage'] = (telemetry['Brake'] == True).mean()
            if 'DRS' in telemetry.columns:
                features['DRS_Enabled_Pct'] = telemetry['DRS'].isin([10, 12, 14]).mean()
    except Exception:
        pass # Ignore errors for single lap telemetry
    return pd.Series(features)

# --- Data Preparation (Corrected & Consolidated) ---
def load_and_engineer_full_race_data(year: int, race_name: str, feature_cols: list) -> pd.DataFrame:
    """
    Loads all data for a race and applies the *full* feature
    engineering pipeline, including telemetry.
    """
    logging.info(f"Loading full session data for {year} {race_name}...")
    try:
        session = fastf1.get_session(year, race_name, 'R')
        # CRITICAL: Must load telemetry=True
        session.load(laps=True, weather=True, telemetry=True)
        session_laps = session.laps.copy()
        session_weather = session.weather_data.copy()
    except Exception as e:
        logging.error(f"Could not load session data: {e}")
        return pd.DataFrame()

    logging.info("Applying telemetry engineering (this may take a few minutes)...")
    tqdm.pandas(desc="Aggregating Telemetry")
    aggregated_features_df = session_laps.progress_apply(get_aggregated_telemetry, axis=1)
    
    # Combine original laps with new telemetry features
    laps_with_telemetry = pd.concat([session_laps, aggregated_features_df], axis=1)

    logging.info("Applying production feature engineering...")
    # --- Mirror the production data engineering pipeline ---
    laps_with_telemetry['Time'] = pd.to_timedelta(laps_with_telemetry['Time'])
    session_weather['Time'] = pd.to_timedelta(session_weather['Time'])
    
    merged_laps = pd.merge_asof(
        laps_with_telemetry.sort_values('Time'),
        session_weather.sort_values('Time'),
        on='Time',
        direction='nearest'
    )
    
    merged_laps['LapTimeSeconds'] = pd.to_timedelta(merged_laps['LapTime'], errors='coerce').dt.total_seconds()
    merged_laps['TempDifference'] = merged_laps['TrackTemp'] - merged_laps['AirTemp']
    
    merged_laps.sort_values(['LapNumber', 'Position'], inplace=True)
    merged_laps['GapToCarAhead'] = merged_laps.groupby('LapNumber')['Time'].diff().dt.total_seconds()
    
    # Fill missing data robustly for all required features
    for col in feature_cols:
        if col not in merged_laps.columns:
            merged_laps[col] = np.nan
            
    # Group by driver to fill gaps
    driver_groups = []
    for driver in merged_laps['DriverNumber'].unique():
        driver_df = merged_laps[merged_laps['DriverNumber'] == driver].copy()
        driver_df[feature_cols] = driver_df[feature_cols].ffill().bfill().fillna(0)
        driver_groups.append(driver_df)
    
    final_df = pd.concat(driver_groups).sort_values(by=['LapNumber', 'Position'])
    logging.info("Full race data engineering complete.")
    return final_df

# --- Prediction Logic ---
def run_prediction_for_window(model, all_data, start_lap, end_lap, drivers, feature_cols, device, seq_len):
    """
    Generates sequences for the current window and runs prediction.
    """
    all_driver_sequences, driver_numbers = [], []
    
    # Get data for the current window
    window_data = all_data[
        (all_data['LapNumber'] >= start_lap) &
        (all_data['LapNumber'] <= end_lap)
    ]
    
    for driver in drivers:
        driver_laps = window_data[window_data['DriverNumber'] == driver]
        
        # We need exactly seq_len laps to predict
        if len(driver_laps) == seq_len:
            sequence = driver_laps[feature_cols].values
            all_driver_sequences.append(sequence)
            driver_numbers.append(driver)
            
    if not all_driver_sequences:
        return None # Not enough data to predict yet

    # Make predictions
    X_pred = np.array(all_driver_sequences, dtype=np.float32)
    sequences_tensor = torch.tensor(X_pred).to(device)
    
    with torch.no_grad():
        predictions = model(sequences_tensor)

    # Format results
    results_df = pd.DataFrame({
        'DriverNumber': driver_numbers,
        'PredictedPosition': predictions.cpu().numpy().flatten()
    }).sort_values('PredictedPosition')
    
    return results_df

# --- Main Simulation Loop ---
def main():
    model, feature_cols = load_artifacts(CONFIG)
    
    # 1. Load and process the *entire* race data once
    full_race_data = load_and_engineer_full_race_data(
        CONFIG["sim_year"], CONFIG["sim_race"], feature_cols
    )
    
    if full_race_data.empty:
        logging.error("Failed to load data. Exiting.")
        return

    # Get driver abbreviations for display
    try:
        session = fastf1.get_session(CONFIG["sim_year"], CONFIG["sim_race"], 'R')
        session.load(telemetry=False, laps=False) # Load minimal data for results
        driver_info = session.results[['DriverNumber', 'Abbreviation']].drop_duplicates()
    except Exception:
        # Fallback if results fail
        driver_info = pd.DataFrame({
            'DriverNumber': full_race_data['DriverNumber'].unique(),
            'Abbreviation': full_race_data['DriverNumber'].unique()
        })
        driver_info['DriverNumber'] = driver_info['DriverNumber'].astype(str)


    active_drivers = full_race_data['DriverNumber'].unique()
    max_laps = int(full_race_data['LapNumber'].max())
    seq_len = CONFIG["sequence_length"]

    logging.info(f"--- Starting Live Prediction Simulation for {CONFIG['sim_year']} {CONFIG['sim_race']} ---")
    
    # 2. Loop from the first possible prediction lap to the end
    for current_lap in range(seq_len, max_laps + 1):
        start_lap = (current_lap - seq_len) + 1
        end_lap = current_lap
        
        print("\n" * 5) # Clear screen
        logging.info(f"=== PREDICTING LAP {current_lap}/{max_laps} (Window: {start_lap}-{end_lap}) ===")
        
        # 3. Run prediction for the current sliding window
        prediction_df = run_prediction_for_window(
            model, full_race_data, start_lap, end_lap,
            active_drivers, feature_cols, CONFIG["device"], seq_len
        )
        
        # 4. Display results
        if prediction_df is not None:
            final_results = pd.merge(prediction_df, driver_info, on='DriverNumber')
            final_results['Rank'] = range(1, len(final_results) + 1)
            print("--- Predicted Race Outcome ---")
            print(final_results[['Rank', 'Abbreviation', 'PredictedPosition']].round(2).to_string(index=False))
        else:
            logging.warning("No complete sequences found for this window.")
            
        time.sleep(CONFIG["sim_delay_seconds"])

    logging.info("--- Simulation Complete ---")

if __name__ == "__main__":
    main()