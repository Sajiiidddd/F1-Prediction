import fastf1
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# 1. SETUP
# Create output folder
OUTPUT_DIR = "race_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
fastf1.Cache.enable_cache('cache')  # Creates a local cache folder for speed

# 2. CONFIGURATION
# We process these years. (2025 is the latest full season available in this context)
TARGET_YEARS = [2023, 2024, 2025]

def get_aggregated_telemetry(lap):
    """
    Calculates detailed telemetry features for a specific lap.
    This is the computationally expensive part.
    """
    features = {
        'TopSpeed': np.nan, 
        'AvgSpeed': np.nan, 
        'ThrottleApplication': np.nan, 
        'BrakeUsage': np.nan, 
        'DRS_Enabled_Pct': np.nan
    }
    try:
        # Fetch high-frequency car data for this lap
        telemetry = lap.get_car_data().add_distance()
        
        if not telemetry.empty:
            # 1. Speed Stats
            if 'Speed' in telemetry.columns:
                features['TopSpeed'] = telemetry['Speed'].max()
                features['AvgSpeed'] = telemetry['Speed'].mean()
            
            # 2. Driver Inputs
            if 'Throttle' in telemetry.columns:
                # % of time throttle was fully pinned (>= 99%)
                features['ThrottleApplication'] = (telemetry['Throttle'] >= 99).mean()
            
            if 'Brake' in telemetry.columns:
                # % of time brake was active
                features['BrakeUsage'] = (telemetry['Brake'] == True).mean()
            
            # 3. DRS Usage
            if 'DRS' in telemetry.columns:
                # DRS status 10, 12, 14 usually indicate open/active
                features['DRS_Enabled_Pct'] = telemetry['DRS'].isin([10, 12, 14]).mean()
                
    except Exception:
        pass
        
    return pd.Series(features)

# 3. MAIN LOOP
print(f"🏎️  Starting Pre-Computation for years: {TARGET_YEARS}")
print(f"📂 Output Directory: {os.path.abspath(OUTPUT_DIR)}\n")

for year in TARGET_YEARS:
    try:
        # Get the full schedule for the year
        schedule = fastf1.get_event_schedule(year)
        
        # Filter for actual races (exclude testing)
        races = schedule[schedule['EventFormat'] != 'testing']
        
        print(f"--- Processing Season {year} ({len(races)} Races) ---")
        
        for _, event in races.iterrows():
            race_name = event['EventName']
            round_num = event['RoundNumber']
            
            # Filename for saving
            safe_race_name = race_name.replace(" ", "") # Remove spaces for safer filenames (e.g., SaudiArabia)
            filename = f"{year}_{event['Location']}.parquet" # Using Location (e.g. 'Bahrain') matches our App logic best
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            # Skip if already exists
            if os.path.exists(output_path):
                print(f"   ⚠️ Skipping {year} {race_name} (Already Exists)")
                continue

            print(f"   📥 Downloading: {race_name} (Round {round_num})...")
            
            try:
                # Load Session
                session = fastf1.get_session(year, round_num, 'R')
                session.load(laps=True, weather=True, telemetry=True)
                
                laps = session.laps
                weather = session.weather_data
                
                if laps.empty:
                    print(f"      ❌ No lap data found. Skipping.")
                    continue

                # Merge Weather Data
                laps['Time'] = pd.to_timedelta(laps['Time'])
                weather['Time'] = pd.to_timedelta(weather['Time'])
                
                # Merge nearest weather row to each lap
                merged = pd.merge_asof(
                    laps.sort_values('Time'), 
                    weather.sort_values('Time'), 
                    on='Time', 
                    direction='nearest'
                )
                
                # Filter valid laps to save processing time
                valid_laps = merged[merged['LapTime'].notna()].copy()
                
                print(f"      ⚙️ Computing telemetry for {len(valid_laps)} laps...")
                
                # Apply Feature Engineering with Progress Bar
                tqdm.pandas(desc="Processing Laps")
                telemetry_df = valid_laps.progress_apply(get_aggregated_telemetry, axis=1)
                
                # Combine original data with new telemetry features
                final_df = pd.concat([valid_laps, telemetry_df], axis=1)
                
                # Select & Clean Columns for the App
                keep_cols = [
                    'DriverNumber', 'LapNumber', 'Position', 'LapTime', 
                    'TyreLife', 'Stint', 'Team',
                    'AirTemp', 'TrackTemp', 'Rainfall', 'WindSpeed',
                    'TopSpeed', 'AvgSpeed', 'ThrottleApplication', 'BrakeUsage', 'DRS_Enabled_Pct'
                ]
                
                # Ensure columns exist (fill missing with defaults)
                for col in keep_cols:
                    if col not in final_df.columns:
                        final_df[col] = 0
                
                final_df = final_df[keep_cols]
                
                # Convert LapTime timedelta to seconds (float)
                final_df['LapTime'] = final_df['LapTime'].dt.total_seconds()
                
                # Handle NaN values
                final_df.fillna(0, inplace=True)
                
                # Save to Parquet (High speed, low size)
                final_df.to_parquet(output_path, index=False)
                print(f"      ✅ Saved to {filename}")
                
            except Exception as e:
                print(f"      ❌ Error processing {race_name}: {e}")

    except Exception as e:
        print(f"❌ Critical Error for Year {year}: {e}")

print("\n🎉 All Processing Complete!")