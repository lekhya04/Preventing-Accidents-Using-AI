# Form text for IEEE DataPort - SafeDrive-Sim-Alcohol

**Title:** SafeDrive-Sim-Alcohol: A Synthetic In-Cabin Sensor Dataset (Gas, Thermal and Pupil Features) for Driver Alcohol-Impairment Classification

**Authors:** Lekhya [surname], Srinivas [surname], Likhitha [surname]  –  Amrita Vishwa Vidyapeetham

**Keywords:** synthetic data; alcohol detection; driver impairment; MQ-3 gas sensor; thermal imaging; pupil; classification; road safety

**Abstract (paste):**
SafeDrive-Sim-Alcohol is a fully synthetic tabular dataset for developing driver alcohol-impairment classifiers. It contains 2,000 simulated driving sessions from 60 simulated participants, balanced between no-alcohol and alcohol classes. Each session is described by five in-cabin sensor features modelled on a low-cost set-up: cabin alcohol vapour (MQ-3 proxy), cabin temperature, minimum and maximum facial temperature from a thermal array, and pupil-to-iris ratio from a camera. Data were produced by a seeded, documented simulator with participant-level random effects, A/C-dependent temperature shifts, realistic confounders (sanitiser or perfume vapour in sober sessions, weak readings in some alcohol sessions) and sensor noise, so that classes overlap and the task is non-trivial. No person was tested; relationships are stylised assumptions rather than physiological measurements. The dataset is intended for prototyping, teaching and benchmarking, and the generator is included for full reproducibility.

**Instructions (paste):**
Load safedrive_sim_alcohol.csv. Feature columns: alcohol_ml_l, cabin_temp_c, face_temp_min_c, face_temp_max_c, pupil_ratio. Target: label (0 = no alcohol, 1 = alcohol). Use participant_id for grouped cross-validation so that no simulated participant appears in both training and test folds. See the documentation PDF and data dictionary. To regenerate: python generate_alcohol.py --seed 2026.

**Files to upload:** safedrive_sim_alcohol.csv, SafeDrive-Sim-Alcohol_data_dictionary.csv, SafeDrive-Sim-Alcohol_documentation.pdf, generate_alcohol.py   |   **Dataset image:** SafeDrive-Sim-Alcohol_cover.png
**Suggested license (state in documentation):** CC BY 4.0 (Creative Commons Attribution 4.0 International)
