import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# UPDATED: File paths now point to the .parquet files
CONFIG = {
    "raw_laps_path": "historical_f1_laps.parquet",
    "raw_weather_path": "historical_f1_weather.parquet",
    "output_path": "processed_f1_data.parquet",
    "feature_cols": [
        'RaceName', 'Year', 'DriverNumber', 'Team', 'LapNumber', 'Position', 
        'LapTimeSeconds', 'TyreLife', 'Stint', 'AirTemp', 'TrackTemp', 
        'Rainfall', 'WindSpeed', 'TempDifference', 'GapToCarAhead'
    ]
}

def load_data(laps_path: str, weather_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads raw lap and weather data from Parquet files."""
    logging.info(f"Loading data from {laps_path} and {weather_path}...")
    # UPDATED: Use pd.read_parquet instead of pd.read_csv
    df_laps = pd.read_parquet(laps_path)
    df_weather = pd.read_parquet(weather_path)
    return df_laps, df_weather

# --- The rest of the script is the same ---

def merge_data(df_laps: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    logging.info("Merging lap and weather data...")
    for df in [df_laps, df_weather]:
        df['Time'] = pd.to_timedelta(df['Time'], errors='coerce')
        df.sort_values('Time', inplace=True)
    
    return pd.merge_asof(
        left=df_laps, right=df_weather, on='Time',
        by=['Year', 'RaceName'], direction='nearest'
    )

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Engineering features...")
    df['LapTimeSeconds'] = pd.to_timedelta(df['LapTime'], errors='coerce').dt.total_seconds()
    df['TempDifference'] = df['TrackTemp'] - df['AirTemp']
    
    df.sort_values(['Year', 'RaceName', 'LapNumber', 'Position'], inplace=True)
    df['GapToCarAhead'] = df.groupby(['Year', 'RaceName', 'LapNumber'])['Time'].diff().dt.total_seconds()
    df['GapToCarAhead'].fillna(0, inplace=True)
    
    return df

def clean_and_finalize(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    logging.info("Finalizing and cleaning data...")
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    df_clean = df[feature_cols].copy()
    
    fill_cols = [col for col in df_clean.columns if col not in ['RaceName', 'Year', 'DriverNumber', 'Team']]
    df_clean[fill_cols] = df_clean.groupby(['Year', 'RaceName', 'DriverNumber'])[fill_cols].ffill().bfill()
    df_clean.fillna(0, inplace=True)
    
    return df_clean

def main():
    df_laps, df_weather = load_data(CONFIG["raw_laps_path"], CONFIG["raw_weather_path"])
    merged_data = merge_data(df_laps, df_weather)
    featured_data = engineer_features(merged_data)
    final_data = clean_and_finalize(featured_data, CONFIG["feature_cols"])
    
    logging.info(f"Saving final processed data to {CONFIG['output_path']}...")
    final_data.to_parquet(CONFIG['output_path'], index=False)
    logging.info("Data processing complete.")

if __name__ == "__main__":
    main()