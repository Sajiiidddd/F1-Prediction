# 🏎️ F1 Neural Strategist: The Transformer's Line

Welcome to the digital paddock. This isn't just a regression model; it's a **5-layer Transformer architecture** designed to peer through the "dirty air" of racing data and predict exactly where the checkered flag will fall for every driver on the grid.

Whether you're a data wizard or a die-hard F1 fan, this project bridges the gap between raw telemetry and podium-level insights.

---

### 🪄 The Tech Wizard's Grimoire
*"I didn't just train a model; I summoned a digital strategist."*

* **The Architecture**: Built a custom `F1Transformer` using PyTorch, utilizing **sine/cosine positional encoding** to ensure the model understands the temporal significance of every lap.
* **The Optuna Magic**: Rather than manual tuning, I let **Optuna** perform its "automated alchemy." Over 50 trials, it explored the multidimensional hyperparameter space to find the "Golden Configuration"—balancing attention heads and dropout to kill overfitting.
* **Temporal Intelligence**: The model analyzes a **10-lap sliding window**, treating each race as a living, breathing sequence rather than a static dataset.

### 🏁 The Fan's Perspective
*"Box box! The data is clear—we're hunting for a podium."*

* **Beyond the Stopwatch**: We don't just look at lap times. Our sensors track **Throttle Application**, **Brake Usage**, and **DRS Deployment** to see who’s pushing the limit and who’s saving fuel.
* **Weathering the Storm**: By merging live weather feeds (Rainfall, Track Temp, Wind Speed), the model knows when a sudden cloud is about to flip the leaderboard.
* **Live Simulation**: Our `predict_live_simulation.py` script mimics a real race weekend, delivering rank updates lap-by-lap, just like the strategy screens on the pit wall.

---

### 📊 The Scoreboard (Performance Metrics)
High-octane precision in every byte:

* **Mean Absolute Error (MAE)**: **0.9068** (Accurate to within less than a single position).
* **Podium Accuracy (Top 3)**: **95.47%** (If we say they're spraying champagne, they usually are).
* **Exact Position Accuracy**: **43.66%**.
* **RMSE**: **1.3535**.

### 🛠️ The 16-Feature Grid
The model monitors everything that matters:
* **Telemetry**: Top Speed, Avg Speed, Throttle %, Brake Usage, DRS %.
* **Race Dynamics**: Position, Lap Number, Gap to Car Ahead, Tyre Life, Stint.
* **The Elements**: Air/Track Temp, Rainfall, Wind Speed, Temp Difference.

---

### 🚀 Future Roadmap: The "Next Gen" Car
* **Radio NLP**: Analyzing team radio in real-time to detect when a driver says the car "feels like a tractor."
* **Strategy Engine**: Predicting "Under-cut" vs. "Over-cut" success probabilities.
* **Web Dashboard**: A live Streamlit interface for your second-screen viewing experience.

---

**License**: MIT — Feel free to fork, tune, and race. 🏎️💨