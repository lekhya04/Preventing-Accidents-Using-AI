"""
Sub-area 4 (Member 3): AI Integration - multi-source risk fusion, alerting and safer-route recommendation

Taken from the standard paper (AI-Driven Accident Risk Prediction and Safer Route Recommendation Using
Multi-Model Data, CCIC 2026):
  * multi-source inputs -> one real-time risk score -> Low / Medium / High risk level
  * threshold-triggered alerts / notification to the registered contact
  * route recommendation that weighs safety against travel time and distance
  * heat-map of high-risk zones built from historical accident records

Our extension over the standard paper: the fusion also uses the DRIVER's own state (behaviour model
+ alcohol model), not only weather / traffic / history.
"""
import heapq, math
import numpy as np
from common import load_json

WEIGHTS = {"behaviour": 0.55, "alcohol": 0.90, "context": 0.85}
LEVELS = [(0.35, "Low"), (0.60, "Medium"), (1.01, "High")]


def fuse(aggr, p_alcohol, context_risk, speed_kmh=None, speed_limit=None):
    """Noisy-OR fusion of independent risk sources (each scaled by a trust weight)."""
    parts = {"behaviour": WEIGHTS["behaviour"] * float(aggr),
             "alcohol": WEIGHTS["alcohol"] * float(p_alcohol),
             "context": WEIGHTS["context"] * float(context_risk)}
    if speed_kmh is not None and speed_limit:
        over = max(0.0, (float(speed_kmh) - float(speed_limit)) / float(speed_limit))
        parts["speeding"] = min(0.8, 1.6 * over)
    R = 1 - np.prod([1 - v for v in parts.values()])
    if p_alcohol >= 0.5:
        R = max(R, 0.6)                                   # rule: detected alcohol is always at least High
    level = next(l for t, l in LEVELS if R < t)
    contrib = {k: v for k, v in parts.items()}
    s = sum(contrib.values()) or 1
    actions = []
    if level == "Low":
        actions.append("Monitoring – no action needed")
    if level in ("Medium", "High"):
        actions.append("Audio-visual warning to the driver")
        actions.append("Suggest safer route avoiding accident hotspots")
    if level == "High":
        actions.append("SMS + call to registered emergency contact (simulated)")
        actions.append("Recommend stopping at the nearest safe location")
    if p_alcohol >= 0.5:
        actions.append("Alcohol detected – ignition interlock recommended")
    if aggr >= 0.6:
        actions.append("Aggressive manoeuvre detected – smooth-driving coaching")
    return {"risk": float(R), "level": level, "contributions": {k: v / s for k, v in contrib.items()}, "raw": parts, "actions": actions}


# ------------------------------------------------------------------------------------------- routing
def _grid():
    h = load_json("hotspots.json")
    cnt = np.array(h["count"]); sev = np.array(h["severe"])
    risk = np.log1p(cnt) / np.log1p(cnt.max())
    risk = risk * (1 + 0.6 * np.divide(sev, np.maximum(cnt, 1)))
    return h, risk / risk.max()


def route(start, goal, safety_weight=4.0, context_multiplier=1.0):
    """Dijkstra on the city grid. cost = distance * (1 + safety_weight * cell_risk * context_multiplier)."""
    h, risk = _grid()
    R, C = risk.shape
    la0, la1, lo0, lo1 = h["box"]
    cell_km = ((la1 - la0) / R * 111.0 + (lo1 - lo0) / C * 111.0 * math.cos(math.radians((la0 + la1) / 2))) / 2

    def solve(w):
        dist = {tuple(start): 0.0}; prev = {}; pq = [(0.0, tuple(start))]
        while pq:
            d, u = heapq.heappop(pq)
            if u == tuple(goal): break
            if d > dist.get(u, 1e18): continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if not dr and not dc: continue
                    v = (u[0] + dr, u[1] + dc)
                    if not (0 <= v[0] < R and 0 <= v[1] < C): continue
                    step = math.hypot(dr, dc)
                    nd = d + step * (1 + w * risk[v] * context_multiplier)
                    if nd < dist.get(v, 1e18):
                        dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
        path = [tuple(goal)]
        while path[-1] != tuple(start):
            path.append(prev[path[-1]])
        path = path[::-1]
        length = sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:])) * cell_km
        exposure = float(sum(risk[p] for p in path) * context_multiplier)
        return {"path": [list(p) for p in path], "km": round(length, 2), "minutes": round(length / 35 * 60, 1),
                "exposure": round(exposure, 2), "hotspots_crossed": int(sum(risk[p] > 0.6 for p in path))}

    fast, safe = solve(0.0), solve(safety_weight)
    red = 0 if fast["exposure"] == 0 else round(100 * (1 - safe["exposure"] / fast["exposure"]), 1)
    return {"fastest": fast, "safest": safe, "risk_reduction_pct": red,
            "extra_minutes": round(safe["minutes"] - fast["minutes"], 1), "grid": [R, C]}
