#!/usr/bin/env python3
import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path.cwd()
CAPTURE_SH = PROJECT_ROOT / "scripts/photoneo/capturar.sh"
META_PATH = PROJECT_ROOT / "photoneo_inbox/live_capture.json"
PLY_PATH = PROJECT_ROOT / "photoneo_inbox/live_scene.ply"


def run_capture():
    t0 = time.time()

    proc = subprocess.run(
        [str(CAPTURE_SH)],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=70,
    )

    output = proc.stdout

    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            meta = {"ok": False, "message": f"No se pudo leer JSON: {e}"}

    ply_exists = PLY_PATH.exists()
    ply_age = time.time() - PLY_PATH.stat().st_mtime if ply_exists else None
    ply_size = PLY_PATH.stat().st_size if ply_exists else None

    ok = (
        proc.returncode == 0
        and bool(meta.get("ok"))
        and ply_exists
        and ply_age is not None
        and ply_age < 120
        and ply_size is not None
        and ply_size > 1000000
    )

    result = {
        "ok": ok,
        "returncode": proc.returncode,
        "duration_sec": round(time.time() - t0, 3),
        "capture_meta": meta,
        "ply_path_host": str(PLY_PATH),
        "ply_path_docker": "data/photoneo/live_scene.ply",
        "ply_exists": ply_exists,
        "ply_age_sec": ply_age,
        "ply_size_bytes": ply_size,
        "stdout_tail": output[-4000:],
    }

    return result


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code, data):
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            self._send_json(200, {
                "ok": True,
                "service": "photoneo_capture_server",
                "project_root": str(PROJECT_ROOT),
                "capture_sh": str(CAPTURE_SH),
            })
            return

        if path == "/capture":
            try:
                result = run_capture()
                self._send_json(200 if result["ok"] else 500, result)
            except Exception as e:
                self._send_json(500, {"ok": False, "error": repr(e)})
            return

        self._send_json(404, {"ok": False, "error": "Ruta no encontrada"})

    def log_message(self, fmt, *args):
        print(fmt % args)


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8765), Handler)
    print("Photoneo capture server en http://127.0.0.1:8765")
    print("Endpoints: /health /capture")
    server.serve_forever()
