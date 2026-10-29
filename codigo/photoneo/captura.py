#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path
from urllib.request import urlopen


URL = os.environ.get("VLA_PHOTONEO_CAPTURE_URL", "http://127.0.0.1:8765/capture")
PLY = Path(os.environ.get("VLA_PHOTONEO_LIVE_PLY", "data/photoneo/live_scene.ply"))
META = Path(os.environ.get("VLA_PHOTONEO_LIVE_META", "data/photoneo/live_capture.json"))


def stat_info(path: Path):
    try:
        st = path.stat()
        return {
            "exists": True,
            "mtime_ns": st.st_mtime_ns,
            "mtime": st.st_mtime,
            "size": st.st_size,
        }
    except FileNotFoundError:
        return {
            "exists": False,
            "mtime_ns": None,
            "mtime": None,
            "size": None,
        }


def main():
    before_ply = stat_info(PLY)
    request_t0_ns = time.time_ns()

    print("Solicitud de captura Photoneo", flush=True)
    print(f"url: {URL}", flush=True)
    print(f"ply: {PLY}", flush=True)
    print(f"meta_out: {META}", flush=True)
    print(f"before_ply: {before_ply}", flush=True)

    try:
        with urlopen(URL, timeout=120) as r:
            raw = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print("No se pudo contactar con el servidor Photoneo.", flush=True)
        print("Sin una captura nueva no se permite planificar.", flush=True)
        print(f"error: {repr(e)}", flush=True)
        return 20

    print("Respuesta del servidor Photoneo", flush=True)
    print(raw[-4000:], flush=True)

    try:
        data = json.loads(raw)
    except Exception as e:
        print("La respuesta del servidor Photoneo no es JSON válido.", flush=True)
        print(f"error: {repr(e)}", flush=True)
        return 21

    if data.get("ok") is not True:
        print("El servidor Photoneo devolvió ok != true.", flush=True)
        return 22

    capture_meta = data.get("capture_meta") or {}
    if capture_meta.get("ok") is not True:
        print("capture_meta.ok != true.", flush=True)
        print(json.dumps(capture_meta, indent=2, ensure_ascii=False), flush=True)
        return 23

    deadline = time.time() + 5.0
    after_ply = stat_info(PLY)

    while time.time() < deadline:
        after_ply = stat_info(PLY)

        ply_newer_than_before = (
            after_ply["exists"]
            and (
                before_ply["mtime_ns"] is None
                or after_ply["mtime_ns"] > before_ply["mtime_ns"]
            )
        )

        ply_after_request = (
            after_ply["exists"]
            and after_ply["mtime_ns"] is not None
            and after_ply["mtime_ns"] >= request_t0_ns
        )

        if ply_newer_than_before and ply_after_request:
            break

        time.sleep(0.1)

    print("Validación de captura Photoneo", flush=True)
    print(f"after_ply: {after_ply}", flush=True)

    if not after_ply["exists"]:
        print("No existe live_scene.ply tras pedir captura.", flush=True)
        return 30

    if before_ply["mtime_ns"] is not None and after_ply["mtime_ns"] <= before_ply["mtime_ns"]:
        print("Photoneo no generó un PLY nuevo.", flush=True)
        print("Se aborta para no reutilizar captura/cache anterior.", flush=True)
        return 32

    if after_ply["mtime_ns"] < request_t0_ns:
        print("El PLY es anterior a la petición actual.", flush=True)
        print("Se aborta porque no hay una captura nueva.", flush=True)
        return 34

    if after_ply["size"] is None or after_ply["size"] < 1000000:
        print("El PLY generado parece demasiado pequeño o inválido.", flush=True)
        return 35

    # El servidor devuelve la metadata por HTTP. La guardamos aquí como metadata oficial del cliente.
    META.parent.mkdir(parents=True, exist_ok=True)
    data["_client_validation"] = {
        "mandatory_new_capture": True,
        "ply_mtime_ns_before": before_ply["mtime_ns"],
        "ply_mtime_ns_after": after_ply["mtime_ns"],
        "ply_size_bytes_after": after_ply["size"],
        "validated_at_time_ns": time.time_ns(),
    }
    META.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Captura Photoneo nueva validada.", flush=True)
    print(f"PLY nuevo: {PLY}", flush=True)
    print(f"size={after_ply['size']} mtime={after_ply['mtime']}", flush=True)
    print(f"Metadata cliente guardada: {META}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
