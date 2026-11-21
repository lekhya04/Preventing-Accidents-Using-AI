# Form text for IEEE DataPort - SafeDrive-Sim-Crash

**Title:** SafeDrive-Sim-Crash: A Synthetic Road-Crash Dataset with Weather, Road and Vehicle Context for Severity and Outcome Prediction

**Authors:** Lekhya [surname], Srinivas [surname], Likhitha [surname]  –  Amrita Vishwa Vidyapeetham

**Keywords:** synthetic data; road crash; accident severity; weather; hotspots; tabular data; multi-task learning; road safety

**Abstract (paste):**
SafeDrive-Sim-Crash is a fully synthetic dataset of 30,000 simulated road-crash records (2022-2025) placed in a bounding box around Coimbatore, India. Each record contains time, location, road type, signalised-junction flag, light condition, weather, temperature, visibility, wind direction, speed limit, primary vehicle type and three outcomes: severity (1 property damage only, 2 minor injury, 3 grievous injury, 4 fatal), road length affected, and clearance time. Records were produced by a seeded, documented simulator with spatial hotspot clusters, monsoon and winter-fog seasonality, commuter-peak hours and a latent-risk model with substantial random noise so that outcomes remain uncertain. The records are not real crashes, locations do not correspond to real accident black spots, and effect sizes are authors' assumptions. The dataset supports teaching, prototyping and benchmarking of severity classification, multi-task learning and hotspot-aware routing; generator code is included for reproducibility.

**Instructions (paste):**
Load safedrive_sim_crash.csv. Targets: severity (1-4), road_affected_km, clearance_time_min. Suggested tasks: risk level (severity 1-2 Low, 3 Medium, 4 High), road impact (0 / <=0.8 km / >0.8 km), clearance (<=30 / >30 min). The file contains crashes only, so it is not suitable for estimating crash probability. See the documentation PDF and data dictionary. To regenerate: python generate_crash.py --seed 2026.

**Files to upload:** safedrive_sim_crash.csv, SafeDrive-Sim-Crash_data_dictionary.csv, SafeDrive-Sim-Crash_documentation.pdf, generate_crash.py   |   **Dataset image:** SafeDrive-Sim-Crash_cover.png
**Suggested license (state in documentation):** CC BY 4.0 (Creative Commons Attribution 4.0 International)
