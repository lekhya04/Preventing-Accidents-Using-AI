"""Train every SafeDrive AI module (takes a few minutes on a laptop).   Usage:  python train_all.py  [--fast]"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import behavior, alcohol, crash

fast = "--fast" in sys.argv
t0 = time.time()
print("=" * 70 + "\n SafeDrive AI – training all modules\n" + "=" * 70)
print("\n[1/4] Driver behaviour detection (real Driver Behavior Dataset)"); behavior.train()
print("\n[2/4] Alcohol impairment detection (IEEE DataPort schema)"); alcohol.train()
print("\n[3/4] Crash risk & severity (real US-Accidents subset)"); crash.train("us", fast=fast)
if crash.stats19_available():
    print("\n[4/4] Crash severity on IEEE DataPort STATS19"); crash.train("stats19", fast=fast)
else:
    print("\n[4/4] STATS19 (IEEE DataPort) file not found in data/crash/stats19_ieee_dataport/ – skipped")
print(f"\nDone in {(time.time() - t0) / 60:.1f} min.  Start the dashboard:  python src/server.py")
