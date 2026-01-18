import fastf1
import pandas as pd
import logging
from tqdm import tqdm
import time
from fastf1.core import DataNotLoadedError

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
fastf1.Cache.enable_cache('cache')

all_laps = []
all_weather = []

# FastF1 data coverage is reliable from 2018 onwards
YEAR_RANGE = range(2018, 2026)

for year in tqdm(YEAR_RANGE, desc="Processing Years"):
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        logging.error(f"❌ Could not load schedule for {year}: {e}")
        continue

    for _, event in schedule.iterrows():
        event_name = event.get("EventName", "")
        round_number = event.get("RoundNumber", None)

        # Skip invalid events
        if pd.isna(event_name) or "Test" in event_name or "Pre-Season" in event_name:
            continue

        logging.info(f"🏁 Processing: {year} {event_name} (Round {round_number})")

        session = None
        try:
            session = fastf1.get_session(year, event_name, "R")
            session.load(laps=True, weather=True, telemetry=False, messages=False)
        except Exception as e_name:
            logging.warning(f"⚠️ Failed by name for {year} {event_name}: {e_name}")
            try:
                # Fallback to round number
                logging.info(f"🔁 Trying fallback using round number {round_number}...")
                session = fastf1.get_session(year, int(round_number), "R")
                session.load(laps=True, weather=True, telemetry=False, messages=False)
            except Exception as e_round:
                logging.error(f"❌ Could not load data for {year} {event_name} (Round {round_number}): {e_round}")
                continue

        # --- Skip sessions where data didn’t load ---
        try:
            laps = session.laps.copy()
            if not laps.empty:
                laps["Year"] = year
                laps["RaceName"] = event_name
                all_laps.append(laps)
                logging.info(f"✅ Added laps for {year} {event_name}: {len(laps)} rows")
            else:
                logging.warning(f"⚠️ No lap data for {year} {event_name}")

        except DataNotLoadedError:
            logging.warning(f"⚠️ Skipping {year} {event_name} — laps data not loaded.")
            continue

        try:
            weather = session.weather_data.copy()
            if not weather.empty:
                weather["Year"] = year
                weather["RaceName"] = event_name
                all_weather.append(weather)
                logging.info(f"🌤️ Added weather for {year} {event_name}: {len(weather)} rows")
            else:
                logging.warning(f"⚠️ No weather data for {year} {event_name}")

        except DataNotLoadedError:
            logging.warning(f"⚠️ Skipping {year} {event_name} — weather data not loaded.")

        time.sleep(1.0)  # rate-limit safety

# --- Data Saving ---
if all_laps:
    laps_df = pd.concat(all_laps, ignore_index=True)
    laps_df.to_parquet("historical_f1_laps.parquet", index=False)
    logging.info(f"💾 Saved {len(laps_df)} lap records to 'historical_f1_laps.parquet'")
else:
    logging.warning("⚠️ No lap data collected.")

if all_weather:
    weather_df = pd.concat(all_weather, ignore_index=True)
    weather_df.to_parquet("historical_f1_weather.parquet", index=False)
    logging.info(f"💾 Saved {len(weather_df)} weather records to 'historical_f1_weather.parquet'")
else:
    logging.warning("⚠️ No weather data collected.")
