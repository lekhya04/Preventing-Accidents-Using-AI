"""Train every SafeDrive AI module on the team's synthetic datasets (takes a few minutes on a laptop).
Usage:  python train_all.py  [--fast]
Missing datasets are regenerated automatically by the scripts in dataset_generators/.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import behavior, alcohol, crash

fast = "--fast" in sys.argv
t0 = time.time()
print("=" * 70 + "\n SafeDrive AI - training all modules on synthetic datasets\n" + "=" * 70)
print("\n[1/3] Driver behaviour detection (SafeDrive-Sim-IMU, synthetic)"); behavior.train()
print("\n[2/3] Alcohol impairment detection (SafeDrive-Sim-Alcohol, synthetic)"); alcohol.train()
print("\n[3/3] Crash risk & severity (SafeDrive-Sim-Crash, synthetic)"); crash.train("sim", fast=fast)
print(f"\nDone in {(time.time() - t0) / 60:.1f} min.  Start the dashboard:  python src/server.py")
