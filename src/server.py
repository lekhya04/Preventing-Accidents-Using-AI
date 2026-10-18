"""SafeDrive AI - web server (Python standard library only). Run:  python src/server.py   ->  http://localhost:8000"""
import json, os, sys, warnings, webbrowser, threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_json
import alcohol, crash, fusion

WEB = os.path.join(ROOT, "web")
PORT = int(os.environ.get("PORT", 8000))


def team():
    with open(os.path.join(ROOT, "team.json"), encoding="utf-8") as f:
        return json.load(f)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=WEB, **k)

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("  " + fmt % args + "\n")

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path == "/api/summary":
                return self._json({"team": team(), "behavior": load_json("behavior.json"), "alcohol": load_json("alcohol.json"),
                                   "crash": load_json("crash_us.json"), "crash_stats19": load_json("crash_stats19.json")})
            if u.path == "/api/replay":
                return self._json(load_json(f"replay_trip_{int(q.get('trip', ['16'])[0])}.json"))
            if u.path == "/api/hotspots":
                return self._json(load_json("hotspots.json"))
        except Exception as e:
            return self._json({"error": str(e)}, 500)
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            if u.path == "/api/alcohol":
                return self._json(alcohol.predict(body))
            if u.path == "/api/crash":
                return self._json(crash.predict_context(body))
            if u.path == "/api/fuse":
                ctx = crash.predict_context(body.get("context", {}))
                alc = alcohol.predict(body.get("alcohol", {}))
                out = fusion.fuse(body.get("aggr", 0.0), alc["p_alcohol"], ctx["risk_score"], body.get("speed"), body.get("speed_limit"))
                out.update({"context": ctx, "alcohol": alc})
                return self._json(out)
            if u.path == "/api/route":
                return self._json(fusion.route(body["start"], body["goal"], float(body.get("safety_weight", 4)), float(body.get("context_multiplier", 1))))
        except Exception as e:
            import traceback; traceback.print_exc()
            return self._json({"error": str(e)}, 500)
        self._json({"error": "unknown endpoint"}, 404)


def models_ok():
    try:
        crash._bundle("us"); alcohol.predict({}); import pickle
        with open(os.path.join(ROOT, "models", "behavior.pkl"), "rb") as f:
            pickle.load(f)
        return load_json("crash_us.json") is not None and load_json("behavior.json") is not None
    except Exception as e:
        print("  Saved models could not be loaded (" + type(e).__name__ + ") - retraining for this computer...")
        return False


def main():
    if not models_ok():
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT, "train_all.py")], check=True)
        crash._BUNDLE.clear(); alcohol._M = None
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"\n  SafeDrive AI dashboard running at  {url}\n  (Ctrl+C to stop)\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
