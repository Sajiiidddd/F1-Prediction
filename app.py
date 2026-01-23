import gradio as gr
import torch
import numpy as np
import pandas as pd
import fastf1
import os
from transformer import F1Transformer

# --- 1. SYSTEM SETUP ---
MODEL_PATH = "final_f1_model.pth"

# Auto-create cache for FastF1 (Fixes "missing folder" error)
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

fastf1.Cache.enable_cache(CACHE_DIR)

# In-Memory Cache (Stores race data while App is running)
SESSION_CACHE = {} 

MODEL_PARAMS = {
    "num_features": 16,
    "d_model": 64,
    "nhead": 8,
    "num_encoder_layers": 5,
    "dropout": 0.1195
}

# --- 2. STATIC DATA & MAPPING ---
CIRCUIT_OPTIONS = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami",
    "Emilia Romagna", "Monaco", "Canada", "Spain", "Austria", "Great Britain",
    "Hungary", "Belgium", "Netherlands", "Italy", "Azerbaijan", "Singapore",
    "USA", "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"
]

RACE_LOC_MAP = {
    "Bahrain": "Sakhir", "Saudi Arabia": "Jeddah", "Australia": "Melbourne",
    "Japan": "Suzuka", "China": "Shanghai", "Miami": "Miami",
    "Emilia Romagna": "Imola", "Monaco": "Monaco", "Canada": "Montréal",
    "Spain": "Barcelona", "Austria": "Spielberg", "Great Britain": "Silverstone",
    "Hungary": "Budapest", "Belgium": "Spa-Francorchamps", "Netherlands": "Zandvoort",
    "Italy": "Monza", "Azerbaijan": "Baku", "Singapore": "Marina Bay",
    "USA": "Austin", "Mexico": "Mexico City", "Brazil": "São Paulo",
    "Las Vegas": "Las Vegas", "Qatar": "Lusail", "Abu Dhabi": "Yas Island"
}

# Fallback names if FastF1 fails
DRIVER_NAMES = {
    "1": "Max Verstappen", "11": "Sergio Perez", "44": "Lewis Hamilton", "63": "George Russell",
    "16": "Charles Leclerc", "55": "Carlos Sainz", "4": "Lando Norris", "81": "Oscar Piastri",
    "14": "Fernando Alonso", "18": "Lance Stroll", "10": "Pierre Gasly", "31": "Esteban Ocon",
    "23": "Alex Albon", "2": "Logan Sargeant", "43": "Franco Colapinto", "77": "Valtteri Bottas",
    "24": "Zhou Guanyu", "27": "Nico Hulkenberg", "20": "Kevin Magnussen", "22": "Yuki Tsunoda",
    "3": "Daniel Ricciardo", "40": "Liam Lawson"
}

def get_driver_name(number):
    return DRIVER_NAMES.get(str(int(number)), f"Driver {int(number)}")

# --- 3. MODEL LOADER ---
def load_model():
    try:
        # Correctly mapping config to class arguments
        model = F1Transformer(
            input_dim=MODEL_PARAMS["num_features"],
            d_model=MODEL_PARAMS["d_model"],
            nhead=MODEL_PARAMS["nhead"],
            num_layers=MODEL_PARAMS["num_encoder_layers"],
            dropout=MODEL_PARAMS["dropout"]
        )
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Model Load Error: {e}")
        return None

model = load_model()

# --- 4. DATA ENGINE ---
def get_aggregated_telemetry(lap):
    features = {'TopSpeed': 300, 'AvgSpeed': 200, 'ThrottleApplication': 0.95, 'BrakeUsage': 0.15, 'DRS_Enabled_Pct': 0.0}
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
    except: pass
    return pd.Series(features)

def fetch_race_data(year, race_name):
    clean_name = race_name.strip()
    mapped_name = RACE_LOC_MAP.get(clean_name, clean_name)
    cache_key = f"{int(year)}_{mapped_name}"
    file_path = f"race_data/{cache_key}.parquet"
    
    # Tier 1: RAM Cache
    if cache_key in SESSION_CACHE:
        return SESSION_CACHE[cache_key]

    # Tier 2: Disk Cache
    if os.path.exists(file_path):
        print(f"⚡ Loading {file_path} from disk...")
        df = pd.read_parquet(file_path)
        SESSION_CACHE[cache_key] = df
        return df

    # Tier 3: Live Download
    print(f"🌍 Downloading {year} {mapped_name} from FastF1...")
    try:
        session = fastf1.get_session(int(year), mapped_name, 'R')
        session.load(laps=True, weather=True, telemetry=True)
        
        # Capture Real Driver Names from Session
        driver_map = {}
        for drv in session.drivers:
            try:
                info = session.get_driver(drv)
                driver_map[str(drv)] = info['BroadcastName']
            except:
                driver_map[str(drv)] = f"Driver {drv}"

        laps = session.laps
        weather = session.weather_data
        
        # Merge Weather
        laps['Time'] = pd.to_timedelta(laps['Time'])
        weather['Time'] = pd.to_timedelta(weather['Time'])
        merged = pd.merge_asof(laps.sort_values('Time'), weather.sort_values('Time'), on='Time', direction='nearest')
        
        # Process Telemetry
        telemetry_df = merged.apply(get_aggregated_telemetry, axis=1)
        final_df = pd.concat([merged, telemetry_df], axis=1)
        
        # Cleanup
        if 'Rainfall' not in final_df.columns: final_df['Rainfall'] = False
        final_df['LapTimeSeconds'] = final_df['LapTime'].dt.total_seconds().fillna(90.0)
        
        # Map Names
        final_df['DriverName'] = final_df['DriverNumber'].astype(str).map(driver_map)
        
        # Save locally
        if not os.path.exists('race_data'): os.makedirs('race_data')
        final_df.to_parquet(file_path, index=False)
        
        SESSION_CACHE[cache_key] = final_df
        return final_df
        
    except Exception as e:
        print(f"❌ Fetch Error: {e}")
        return None

# --- 5. INTERFACE LOGIC ---

# Endpoint 1: Update Slider
def load_race_details(year, race):
    df = fetch_race_data(year, race)
    if df is None:
        return gr.update(maximum=70, value=0, label="Error loading race")
    
    total_laps = int(df['LapNumber'].max())
    return gr.update(maximum=total_laps, value=min(10, total_laps), label=f"Select Lap (Max {total_laps})")

# Endpoint 2: Full Grid Simulation
def run_strategy_simulation(year, race, lap_target, rain_override, temp_override):
    if model is None: return {"error": "Model not loaded"}

    df = fetch_race_data(year, race)
    if df is None: return {"error": "Race data not found"}
        
    target = int(lap_target)
    current_laps = df[df['LapNumber'] == target].copy()
    
    if current_laps.empty:
        target = int(df['LapNumber'].max())
        current_laps = df[df['LapNumber'] == target].copy()

    results = []
    
    for _, row in current_laps.iterrows():
        try:
            # 1. Base Data
            air_temp = row.get('AirTemp', 25)
            track_temp = row.get('TrackTemp', 35)
            rainfall = 1.0 if row.get('Rainfall', False) else 0.0
            
            # 2. Overrides
            if temp_override > 0: air_temp = temp_override
            if rain_override: rainfall = 1.0
            
            feature_vector = np.array([
                target,
                row.get('Position', 10),
                row.get('LapTimeSeconds', 90),
                row.get('TyreLife', 5),
                row.get('Stint', 1),
                air_temp,
                track_temp,
                rainfall,
                row.get('WindSpeed', 2),
                track_temp - air_temp,
                0.5, 
                row.get('TopSpeed', 300),
                row.get('AvgSpeed', 200),
                row.get('ThrottleApplication', 0.95),
                row.get('BrakeUsage', 0.15),
                1.0 if row.get('DRS_Enabled_Pct', 0) > 0.05 else 0.0
            ], dtype=np.float32)

            input_seq = np.tile(feature_vector, (1, 10, 1))
            with torch.no_grad():
                pred = model(torch.from_numpy(input_seq))
                pred_pos = pred.item()
                
            actual_pos = int(row.get('Position', 0))
            delta = actual_pos - pred_pos
            
            # Use Dynamic Name
            d_name = row.get('DriverName')
            if pd.isna(d_name): d_name = f"Driver {int(row['DriverNumber'])}"
            
            results.append({
                "Driver": str(int(row['DriverNumber'])),
                "DriverName": str(d_name),
                "Team": row.get('Team', 'Unknown'),
                "ActualPos": actual_pos,
                "PredictedPos": round(pred_pos, 2),
                "Delta": round(delta, 2)
            })
        except: continue
        
    results.sort(key=lambda x: x['PredictedPos'])
    return results

# Endpoint 3: Fetch Single Driver State (For Deep Dive)
def get_driver_state(year, race, lap, driver_id):
    df = fetch_race_data(year, race)
    if df is None: return {"error": "Race data missing"}
    
    # Filter
    row = df[(df['LapNumber'] == int(lap)) & (df['DriverNumber'].astype(str) == str(driver_id))]
    if row.empty: return {"error": "Driver not found on this lap"}
    
    r = row.iloc[0]
    return {
        "success": True,
        "Driver": str(r['DriverNumber']),
        "Name": str(r.get('DriverName', 'Unknown')),
        "Team": str(r.get('Team', 'Unknown')),
        "Position": int(r['Position']),
        "TyreLife": int(r['TyreLife']),
        "LapTime": float(r['LapTimeSeconds']),
        "AirTemp": float(r['AirTemp']),
        "Rain": bool(r['Rainfall'])
    }

# Endpoint 4: Predict Single Strategy (For Deep Dive)
def predict_scenario(inputs):
    # inputs = [lap, pos, time, tyre, stint, air, track, rain, wind, gap, top, avg, thr, brk, drs]
    # To simplify frontend, we accept a smaller list and pad it with defaults
    try:
        # Expected inputs: [lap, pos, tyre, gap, rain, air_temp, lap_time]
        lap, pos, tyre, gap, rain, air_temp, lap_time = inputs
        
        feature_vector = np.array([
            lap, pos, lap_time, tyre, 2, # Default Stint 2
            air_temp, air_temp+10,       # Track Temp estimate
            1.0 if rain else 0.0,
            2.0,                         # Wind
            10.0,                        # Diff
            gap,
            310, 210, 0.96, 0.15, 1.0    # Telemetry Defaults
        ], dtype=np.float32)

        input_seq = np.tile(feature_vector, (1, 10, 1))
        
        with torch.no_grad():
            pred = model(torch.from_numpy(input_seq))
        
        return f"P{pred.item():.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

# --- 6. GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏎️ F1 Strategy Research Engine")
    
    # 1. Main Grid Simulation Inputs
    with gr.Row():
        with gr.Column(scale=1):
            in_year = gr.Number(label="Season Year", value=2024, precision=0)
            in_race = gr.Dropdown(choices=CIRCUIT_OPTIONS, label="Circuit", value="Bahrain", allow_custom_value=True, interactive=True)
            btn_load = gr.Button("📂 Load Race Data", variant="secondary")
            in_lap = gr.Slider(minimum=1, maximum=70, value=10, step=1, label="Race Lap")
            
            gr.Markdown("### 🛠️ Global Strategy Overrides")
            in_rain = gr.Checkbox(label="Force Heavy Rain")
            in_temp = gr.Number(label="Force Air Temp (°C)", value=0)
            btn_run = gr.Button("🚀 Run Grid Simulation", variant="primary")

        with gr.Column(scale=2):
            out_result = gr.JSON(label="Grid Analysis")
    
    # 2. Hidden API Inputs for Driver Deep Dive (Called by JS)
    with gr.Row(visible=False):
        in_drv_id = gr.Textbox()
        out_drv_data = gr.JSON()
        btn_fetch_drv = gr.Button("Fetch Driver")
        
        # [Lap, Pos, Tyre, Gap, Rain, AirTemp, LapTime]
        in_strat_params = [gr.Number(), gr.Number(), gr.Number(), gr.Number(), gr.Checkbox(), gr.Number(), gr.Number()]
        out_strat_pred = gr.Textbox()
        btn_run_strat = gr.Button("Predict Strategy")

    # Bindings
    btn_load.click(fn=load_race_details, inputs=[in_year, in_race], outputs=[in_lap])
    btn_run.click(fn=run_strategy_simulation, inputs=[in_year, in_race, in_lap, in_rain, in_temp], outputs=[out_result], api_name="simulate_grid")
    
    # API Bindings for Deep Dive
    btn_fetch_drv.click(fn=get_driver_state, inputs=[in_year, in_race, in_lap, in_drv_id], outputs=[out_drv_data], api_name="get_driver_state")
    btn_run_strat.click(fn=predict_scenario, inputs=in_strat_params, outputs=[out_strat_pred], api_name="predict_scenario")

if __name__ == "__main__":
    demo.launch()