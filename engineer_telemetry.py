import fastf1
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
fastf1.Cache.enable_cache('cache')

CONFIG = {
    "input_path": "processed_f1_data.parquet",
    "output_path": "final_features_data.parquet",
    # --- ADDED: Define the new list of features to be engineered ---
    "telemetry_features_list": [
        'TopSpeed', 
        'AvgSpeed', 
        'ThrottleApplication', 
        'BrakeUsage', 
        'DRS_Enabled_Pct'
    ]
}

def get_aggregated_telemetry(lap):
    """
    Calculates aggregated telemetry features for a single lap.
    Returns a pd.Series with NaNs if telemetry is not available.
    """
    # --- Define a dictionary with NaN as default for all features ---
    features = {
        'TopSpeed': np.nan,
        'AvgSpeed': np.nan,
        'ThrottleApplication': np.nan,
        'BrakeUsage': np.nan,
        'DRS_Enabled_Pct': np.nan
    }
    
    try:
        # Get car data once
        telemetry = lap.get_car_data().add_distance()
        
        if not telemetry.empty:
            # --- Calculate all features from the single telemetry object ---
            
            # Speed features
            if 'Speed' in telemetry.columns:
                features['TopSpeed'] = telemetry['Speed'].max()
                features['AvgSpeed'] = telemetry['Speed'].mean()
            
            # Throttle: Percentage of lap at >= 99% throttle
            if 'Throttle' in telemetry.columns:
                features['ThrottleApplication'] = (telemetry['Throttle'] >= 99).mean()
            
            # Brake: Percentage of lap with brake applied
            if 'Brake' in telemetry.columns:
                features['BrakeUsage'] = (telemetry['Brake'] == True).mean()

            # DRS: Percentage of lap with DRS active (codes 10, 12, 14)
            if 'DRS' in telemetry.columns:
                features['DRS_Enabled_Pct'] = telemetry['DRS'].isin([10, 12, 14]).mean()

    except Exception as e:
        # Log telemetry access errors but allow the process to continue
        logging.warning(f"Could not load telemetry for a lap: {e}")
    
    return pd.Series(features)


def main():
    """Main pipeline to engineer telemetry features."""
    logging.info(f"Loading processed data from {CONFIG['input_path']}...")
    df_processed = pd.read_parquet(CONFIG['input_path'])
    
    all_races_with_telemetry = []
    
    grouped_races = df_processed.groupby(['Year', 'RaceName'])
    
    for (year, race_name), group in tqdm(grouped_races, desc="Processing Races"):
        logging.info(f"Fetching session and telemetry for {year} {race_name}...")
        try:
            session = fastf1.get_session(year, race_name, 'R')
            session.load(laps=True, telemetry=True) # Must load telemetry
            
            session_laps = session.laps.copy()
            
            # --- UPDATED: Use the new aggregated function ---
            tqdm.pandas(desc=f"Aggregating Telemetry for {race_name}")
            # Apply the function to get a DataFrame of new features
            aggregated_features_df = session_laps.progress_apply(get_aggregated_telemetry, axis=1)
            
            # Combine original lap identifiers with new features
            session_laps_with_features = pd.concat([session_laps, aggregated_features_df], axis=1)

            # --- UPDATED: Select all new features for the merge ---
            telemetry_features_to_merge = session_laps_with_features[
                ['DriverNumber', 'LapNumber'] + CONFIG['telemetry_features_list']
            ]
            
            # Merge the new features back into our original data for that race
            race_group_with_telemetry = pd.merge(
                group, telemetry_features_to_merge,
                on=['DriverNumber', 'LapNumber'],
                how='left'
            )
            all_races_with_telemetry.append(race_group_with_telemetry)
            
        except Exception as e:
            logging.error(f"Failed to process {year} {race_name}: {e}")
            # If a session fails, add empty columns and append original data
            for col in CONFIG['telemetry_features_list']:
                group[col] = np.nan
            all_races_with_telemetry.append(group)

    # Combine all the processed race groups back into one final DataFrame
    final_df = pd.concat(all_races_with_telemetry)
    
    logging.info("Finalizing telemetry features...")
    
    # --- UPDATED: Clean up all new telemetry columns ---
    for col in CONFIG['telemetry_features_list']:
        # Forward-fill, backward-fill, then fill remaining NaNs with 0
        final_df[col] = final_df.groupby(['Year', 'RaceName', 'DriverNumber'])[col].ffill().bfill()
        final_df[col] = final_df[col].fillna(0)
    
    logging.info(f"Saving final feature-engineered data to {CONFIG['output_path']}...")
    final_df.to_parquet(CONFIG['output_path'], index=False)
    logging.info("Telemetry feature engineering complete.")

if __name__ == "__main__":
    main()
