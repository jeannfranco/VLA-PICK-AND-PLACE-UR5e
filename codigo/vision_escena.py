#!/usr/bin/env python3

import os as _shape_os
import sys as _shape_sys

if (
    _shape_os.environ.get("VLA_TASK_FAMILY", "").strip()
    == "shape_classification"
):
    _shape_detector = _shape_os.path.join(
        _shape_os.path.dirname(__file__),
        "vision_forma.py",
    )

    _shape_ply = _shape_os.environ.get(
        "VLA_LIVE_PLY",
        "data/photoneo/live_scene.ply",
    )

    print(
        f"detector={_shape_detector} "
        f"ply={_shape_ply}",
        flush=True,
    )

    _shape_os.execv(
        _shape_sys.executable,
        [
            _shape_sys.executable,
            _shape_detector,
            "--ply",
            _shape_ply,
        ],
    )

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


WORKSPACE = Path("/workspace")
SCRIPTS = WORKSPACE / "scripts"

RESULT_ROOTS = [
    Path("results/validacion_robot"),
    Path("results/validacion_robot"),
]

OUT_ROOT = Path("results/validacion_robot")


def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def existing_jsons():
    out = set()
    for root in RESULT_ROOTS:
        if not root.exists():
            continue
        for j in root.glob("*/resultado_zonas.json"):
            try:
                out.add(str(j.resolve()))
            except Exception:
                out.add(str(j))
    return out


def find_new_scene_json(before_set, must_have_key=None, prefer_prefix=None):
    candidates = []

    for root in RESULT_ROOTS:
        if not root.exists():
            continue

        for j in root.glob("*/resultado_zonas.json"):
            js = str(j.resolve()) if j.exists() else str(j)
            if js in before_set:
                continue

            if prefer_prefix and not j.parent.name.startswith(prefer_prefix):
                continue

            try:
                data = load_json(j)
            except Exception:
                continue

            if must_have_key and must_have_key not in data:
                continue

            candidates.append((j.stat().st_mtime_ns, j, data))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def find_latest_scene_json(must_have_key=None, prefer_prefix=None):
    candidates = []

    for root in RESULT_ROOTS:
        if not root.exists():
            continue

        for j in root.glob("*/resultado_zonas.json"):
            if prefer_prefix and not j.parent.name.startswith(prefer_prefix):
                continue

            try:
                data = load_json(j)
            except Exception:
                continue

            if must_have_key and must_have_key not in data:
                continue

            candidates.append((j.stat().st_mtime_ns, j, data))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def run_script_and_get_scene(cmd, must_have_key=None, prefer_prefix=None, allow_fail=False):
    before = existing_jsons()

    print("Comando:", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(cmd, text=True)

    if p.returncode != 0 and not allow_fail:
        raise SystemExit(f"Falló comando con RC={p.returncode}: {' '.join(map(str, cmd))}")

    j, data = find_new_scene_json(
        before_set=before,
        must_have_key=must_have_key,
        prefer_prefix=prefer_prefix,
    )

    if j is None:
        # Si el detector reutiliza un directorio con el mismo nombre o
        # la resolución temporal impide que aparezca como "nuevo", se toma el resultado válido más reciente.
        j, data = find_latest_scene_json(
            must_have_key=must_have_key,
            prefer_prefix=prefer_prefix,
        )

    return j, data, p.returncode


def xyz_of(obj):
    if not isinstance(obj, dict):
        return None

    for k in ("base_xyz_m", "center_base_xyz_m", "xyz_m", "safe_above_base_xyz_m"):
        v = obj.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            try:
                return [float(v[0]), float(v[1]), float(v[2])]
            except Exception:
                pass

    return None


def normalize_object(obj, object_key):
    if not isinstance(obj, dict):
        raise RuntimeError(f"Objeto {object_key} inválido")

    xyz = xyz_of(obj)
    if xyz is None:
        raise RuntimeError(f"Objeto {object_key} sin base_xyz_m usable")

    out = dict(obj)
    out["base_xyz_m"] = xyz

    if "center_base_xyz_m" not in out:
        out["center_base_xyz_m"] = xyz

    if "safe_above_base_xyz_m" not in out:
        out["safe_above_base_xyz_m"] = [
            float(xyz[0]),
            float(xyz[1]),
            max(float(xyz[2]) + 0.18, 0.22),
        ]

    out["semantic_override"] = object_key

    if object_key == "orange_cube":
        out.setdefault("object_shape", "cube")
        out.setdefault("radius_m", 0.030)
    elif object_key == "white_cylinder":
        out.setdefault("object_shape", "cylinder")
        r = out.get("radius_p90_m") or out.get("radius_median_m") or out.get("radius_m")
        try:
            out["radius_m"] = float(max(0.018, min(0.050, float(r))))
        except Exception:
            out["radius_m"] = 0.035
    else:
        out.setdefault("radius_m", 0.030)

    return out


def maybe_copy_diag(src_json: Path, dst_dir: Path, dst_name: str):
    if not src_json:
        return None

    src = src_json.parent / "diagnostico_zonas.png"
    if not src.exists():
        src = src_json.parent / "photoneo_detection_diag.png"

    if not src.exists():
        return None

    dst = dst_dir / dst_name
    try:
        shutil.copy2(src, dst)
        return str(dst)
    except Exception:
        return None


def maybe_copy_clean(src_json: Path, dst_dir: Path, dst_name: str):
    """
    Copia la imagen RGB Photoneo original, antes de dibujar
        cajas, etiquetas o máscaras de detección.
    """
    if not src_json:
        return None

    candidates = [
        src_json.parent / "photoneo_rgb.png",
        src_json.parent / "photoneo_rgb_for_openvla.png",
    ]

    src = next(
        (candidate for candidate in candidates if candidate.exists()),
        None,
    )

    if src is None:
        return None

    dst = dst_dir / dst_name

    try:
        shutil.copy2(src, dst)
        return str(dst)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument(
        "--selected-object",
        choices=["orange_cube", "white_cylinder"],
        default="orange_cube",
    )
    args = ap.parse_args()

    ply = Path(args.ply)
    if not ply.exists():
        raise SystemExit(f"No existe PLY: {ply}")

    ts = now_tag()
    out_dir = OUT_ROOT / f"escena_{ts}_{args.selected_object}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Detección de objetos desde PLY", flush=True)
    print("PLY:", ply, flush=True)
    print("Salida:", out_dir, flush=True)

    orange_cmd = [
        sys.executable,
        str(SCRIPTS / "vision_zonas.py"),
        "--ply",
        str(ply),
    ]

    orange_json, orange_scene, rc = run_script_and_get_scene(
        orange_cmd,
        must_have_key="orange_die",
        prefer_prefix="zonas_",
        allow_fail=False,
    )

    if orange_json is None or not isinstance(orange_scene, dict):
        raise SystemExit("No se pudo localizar JSON orange -> yellow válido tras ejecutar detector.")

    if "orange_die" not in orange_scene:
        raise SystemExit("La escena naranja localizada no tiene orange_die.")

    print("orange_json:", orange_json, flush=True)

    objects = {}
    orange_obj = normalize_object(orange_scene["orange_die"], "orange_cube")
    objects["orange_cube"] = orange_obj

    white_json = None
    white_scene = None

    white_detector = SCRIPTS / "detector_cilindro.py"
    if white_detector.exists():
        white_cmd = [
            sys.executable,
            str(white_detector),
            "--ply",
            str(ply),
        ]

        white_json, white_scene, white_rc = run_script_and_get_scene(
            white_cmd,
            must_have_key=None,
            prefer_prefix=None,
            allow_fail=True,
        )

        raw_white = None
        if isinstance(white_scene, dict):
            raw_white = (
                white_scene.get("white_cylinder")
                or (white_scene.get("objects") or {}).get("white_cylinder")
            )

        if isinstance(raw_white, dict) and xyz_of(raw_white) is not None:
            try:
                objects["white_cylinder"] = normalize_object(raw_white, "white_cylinder")
                print("white_json:", white_json, flush=True)
            except Exception as e:
                print("white_cylinder detectado pero inválido:", e, flush=True)
        else:
            print("No se añadió white_cylinder: detector sin objeto usable.", flush=True)
    else:
        print("No existe detector white_cylinder:", white_detector, flush=True)

    if args.selected_object not in objects:
        raise SystemExit(
            f"selected_object={args.selected_object} no está en objects={sorted(objects)}"
        )

    selected = objects[args.selected_object]

    yellow = (
        orange_scene.get("yellow_target")
        or orange_scene.get("yellow_zone")
        or orange_scene.get("target")
    )

    if not isinstance(yellow, dict) or xyz_of(yellow) is None:
        raise SystemExit("No encuentro yellow_target usable en escena naranja.")

    yellow = dict(yellow)
    yellow["base_xyz_m"] = xyz_of(yellow)
    yellow.setdefault("center_base_xyz_m", yellow["base_xyz_m"])
    yellow.setdefault(
        "safe_above_base_xyz_m",
        [
            float(yellow["base_xyz_m"][0]),
            float(yellow["base_xyz_m"][1]),
            max(float(yellow["base_xyz_m"][2]) + 0.20, 0.23),
        ],
    )

    targets = {
        "yellow_zone": yellow,
        "yellow_target": yellow,
    }

    # Si el detector trae blue_source, se mantiene como zona/target azul.
    blue = orange_scene.get("blue_source") or orange_scene.get("blue_zone")
    if isinstance(blue, dict) and xyz_of(blue) is not None:
        blue = dict(blue)
        blue["base_xyz_m"] = xyz_of(blue)
        blue.setdefault("center_base_xyz_m", blue["base_xyz_m"])
        blue.setdefault(
            "safe_above_base_xyz_m",
            [
                float(blue["base_xyz_m"][0]),
                float(blue["base_xyz_m"][1]),
                max(float(blue["base_xyz_m"][2]) + 0.20, 0.23),
            ],
        )
        targets["blue_zone"] = blue
        targets["blue_source"] = blue

    clean_rgb_image = maybe_copy_clean(
        orange_json or white_json,
        out_dir,
        "photoneo_rgb.png",
    )

    # El ejecutor lee internamente orange_die/yellow_target.
    result = {
        "task": f"all_objects_scene_selected_{args.selected_object}_to_yellow_surface",
        "json_path": str(out_dir / "resultado_zonas.json"),
        "ply": str(ply),
        "selected_object": args.selected_object,
        "objects": objects,
        "targets": targets,
        "orange_die": selected,
        "yellow_target": yellow,
        "blue_source": blue if isinstance(blue, dict) else orange_scene.get("blue_source"),
        "source_jsons": {
            "orange_json": str(orange_json) if orange_json else None,
            "white_json": str(white_json) if white_json else None,
        },
        "diag_images": {
            "orange": maybe_copy_diag(orange_json, out_dir, "orange_diagnostico_zonas.png"),
            "white": maybe_copy_diag(white_json, out_dir, "white_diagnostico_zonas.png"),
        },
        "photoneo_rgb": clean_rgb_image,
        "note": (
            "Escena multiobjeto generada desde PLY live. "
            "orange_die apunta al objeto seleccionado por compatibilidad con el ejecutor."
        ),
    }

    out_json = out_dir / "resultado_zonas.json"
    write_json(out_json, result)

    # Este es el JSON canónico de la última captura de cámara.
    # Así flujo/ejecución no confunde la escena con un JSON interno *_cilindro_blanco.
    latest_file = Path("results/validacion_robot/ultima_escena.txt")
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(str(out_json.resolve()) + "\n", encoding="utf-8")

    # Se actualiza mtime para que el archivo tenga prioridad en búsquedas por resultado reciente.
    import os as _os
    _os.utime(out_json, None)

    print(f"{out_json.resolve()}", flush=True)

    print()
    print("Escena multiobjeto generada.")
    print("json:", out_dir / "resultado_zonas.json")
    print("selected/base_xyz_m:", selected.get("base_xyz_m"))
    print("yellow/base_xyz_m:", yellow.get("base_xyz_m"))
    print("objects:", sorted(objects.keys()))


if __name__ == "__main__":
    main()
