"""Local web server: serves web/ and a small JSON API around the segmentation pipeline.

Run:  python server.py        (opens http://127.0.0.1:8000)
Uses only the standard library plus the project's ML dependencies.
"""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src import pipeline as pl
from src import sequences as sq
from src.api import build_payload, to_jsonable
from src.data_loader import SchemaError, load_dataset, load_with_cache
from src.dimensionality import fit_representation
from src.feature_engineering import FEATURE_SETS

ROOT = Path(__file__).parent
WEB, DATA_DIR, ART_DIR = ROOT / "web", ROOT / "data", ROOT / "artifacts"

LOCK = threading.RLock()
STATE = {"raw": None, "name": None, "status": "idle", "error": None, "key": 0}
CACHE: dict = {"prep": {}, "rep": {}, "sweeps": {}, "models": {}, "hmm": {}}


def _default_file() -> Path | None:
    return next((p for p in [DATA_DIR / "Online Retail.xlsx", DATA_DIR / "Online Retail.csv"] if p.exists()), None)


def _load_in_background(loader, name: str) -> None:
    def work():
        with LOCK:
            STATE.update(status="loading", error=None, name=name)
        try:
            raw = loader()
            with LOCK:
                for c in CACHE.values():
                    c.clear()
                STATE.update(raw=raw, status="ready", key=STATE["key"] + 1)
        except (SchemaError, ValueError) as e:
            with LOCK:
                STATE.update(status="error", error=str(e), raw=None)
        except Exception as e:  # unreadable/corrupt file
            with LOCK:
                STATE.update(status="error", error=f"Could not read the dataset: {e}", raw=None)

    threading.Thread(target=work, daemon=True).start()


def compute_result(q: dict) -> dict:
    """Cached pipeline stages keyed by the controls, mirroring the old dashboard's behaviour."""
    with LOCK:
        raw, dkey = STATE["raw"], STATE["key"]
        if raw is None:
            raise ValueError("No dataset loaded.")
        fs = q.get("fs", "RFM")
        if fs not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set '{fs}'.")
        thr = min(max(float(q.get("thr", 0.9)), 0.5), 0.99)
        cap = min(max(int(q.get("cap", 3000)), 300), 20000)
        prep = CACHE["prep"].setdefault(dkey, pl.prepare_customers(raw))
        customers = prep["customers"]
        rk = (fs, thr)
        if rk not in CACHE["rep"]:
            CACHE["rep"][rk] = fit_representation(customers, FEATURE_SETS[fs], thr)
        rep = CACHE["rep"][rk]
        if rk not in CACHE["sweeps"]:
            CACHE["sweeps"][rk] = pl.run_sweeps(rep["Z"])
        sw = CACHE["sweeps"][rk]
        ks = sw["kmeans_sweep"]["k"].tolist()
        k = int(q["k"]) if q.get("k") else sw["k_recommended"]
        gk = int(q["gmm_k"]) if q.get("gmm_k") else sw["gmm_k_bic"]
        if k not in ks or gk not in ks:
            raise ValueError(f"Cluster counts must be one of {ks}.")
        mk = (fs, thr, k, gk, cap)
        if mk not in CACHE["models"]:
            CACHE["models"][mk] = pl.fit_models(customers, rep, k, gk, cap)
        return {"feature_set": fs, "hier_cap": cap, **prep, "rep": rep, **sw, **CACHE["models"][mk]}


def hmm_state() -> dict:
    """Monthly sequences and the fitted HMM sweep (2-5 states), cached per dataset."""
    with LOCK:
        raw, dkey = STATE["raw"], STATE["key"]
        if raw is None:
            raise ValueError("No dataset loaded.")
        if dkey not in CACHE["hmm"]:
            prep = CACHE["prep"].setdefault(dkey, pl.prepare_customers(raw))
            seqs = sq.prepare_sequences(prep["clean"])
            if seqs["ids"] != prep["customers"]["CustomerID"].tolist():
                raise ValueError("Internal error: sequence customers do not match the customer table.")
            rows, models = sq.sweep(seqs)
            CACHE["hmm"][dkey] = {"seqs": seqs, "rows": rows, "models": models, "bt": {}}
        return CACHE["hmm"][dkey]


def hmm_payload(q: dict) -> dict:
    st = hmm_state()
    k = int(q["k"]) if q.get("k") else 3
    if k not in st["models"]:
        raise ValueError(f"Number of states must be one of {sorted(st['models'])}.")
    with LOCK:
        if k not in st["bt"]:
            st["bt"][k] = sq.backtest(st["seqs"], k)
        return to_jsonable(sq.build_payload(st["seqs"], st["models"], st["rows"], k, st["bt"][k]))


def hmm_customer(q: dict) -> dict:
    st = hmm_state()
    k = int(q.get("k", 3))
    ids = st["seqs"]["ids"]
    cid = str(q.get("id", ""))
    if cid not in ids:
        raise ValueError(f"No customer with ID '{cid}'.")
    return to_jsonable(sq.customer_detail(st["seqs"], st["models"][k], ids.index(cid)))


def results_payload(q: dict) -> dict:
    dend = min(max(int(q.get("dend", 300)), 50), 600)
    if q.get("saved") == "1":
        result = pl.load_artifacts(ART_DIR)["result"]
        payload = build_payload(result, None, dend)
        payload["meta"]["saved"] = True
        return payload
    result = compute_result(q)
    with LOCK:
        preview = STATE["raw"].head(50).astype(object).where(STATE["raw"].head(50).notna(), None)
    payload = build_payload(result, preview, dend)
    payload["meta"]["saved"] = False
    return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "Segmentation/1.0"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/api/status":
            with LOCK:
                return self._json({"status": STATE["status"], "error": STATE["error"], "name": STATE["name"],
                                   "saved_exists": (ART_DIR / pl.ARTIFACT_NAME).exists()})
        if u.path == "/api/results":
            with LOCK:
                status = STATE["status"]
            if q.get("saved") != "1" and status != "ready":
                return self._json({"error": "Dataset is not ready.", "status": status}, 409)
            try:
                return self._json(results_payload(q))
            except (ValueError, FileNotFoundError) as e:
                return self._json({"error": str(e)}, 400)
            except Exception as e:
                return self._json({"error": f"Unexpected error: {e}"}, 500)
        if u.path in ("/api/hmm", "/api/hmm_customer"):
            with LOCK:
                status = STATE["status"]
            if status != "ready":
                return self._json({"error": "Dataset is not ready (saved artifacts do not include sequences).", "status": status}, 409)
            try:
                return self._json(hmm_payload(q) if u.path == "/api/hmm" else hmm_customer(q))
            except (ValueError, KeyError) as e:
                return self._json({"error": str(e)}, 400)
            except Exception as e:
                return self._json({"error": f"Unexpected error: {e}"}, 500)
        return self._static(u.path)

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/api/upload":
            name = Path(q.get("name", "upload.xlsx")).name
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            _load_in_background(lambda: load_dataset(io.BytesIO(data), name), name)
            return self._json({"ok": True})
        if u.path == "/api/save":
            try:
                result = compute_result(q)
                return self._json({"path": str(pl.save_artifacts(result, ART_DIR))})
            except Exception as e:
                return self._json({"error": str(e)}, 400)
        self.send_error(404)

    def _static(self, path: str):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        f = (WEB / rel).resolve()
        if WEB.resolve() not in f.parents or not f.is_file():
            return self.send_error(404)
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(f.name)[0] or "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    f = _default_file()
    if f:
        _load_in_background(lambda: load_with_cache(f, ART_DIR), f.name)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Customer segmentation dashboard: {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
