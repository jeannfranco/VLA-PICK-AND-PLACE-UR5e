#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/workspace")
RESULTS = ROOT / "results"
SCENE_ROOT = RESULTS / "validacion_robot"
PROFILE_DIR = RESULTS / "perfiles"
LOG_ROOT = RESULTS / "demo"

EXECUTOR = ROOT / "scripts" / "ejecucion.py"


OBJECT_ALIASES = {
    "orange_cube": [
        "orange cube", "orange block", "orange object", "orange die",
        "cubo naranja", "bloque naranja", "dado naranja", "objeto naranja",
        "cube", "block", "cubo", "dado"
    ],
    "white_cylinder": [
        "white cylinder", "white object", "cylinder",
        "cilindro blanco", "objeto blanco", "cilindro"
    ],
}

TARGET_ALIASES = {
    "yellow_zone": [
        "yellow zone", "yellow area", "yellow target", "yellow box",
        "zona amarilla", "area amarilla", "área amarilla", "amarillo", "amarilla", "yellow"
    ],
    "blue_zone": [
        "blue zone", "blue area", "blue target", "blue box",
        "zona azul", "area azul", "área azul", "azul", "blue"
    ],

    "square_box": [
        "square box",
        "square container",
        "square receptacle",
        "square-shaped box",
        "square-shaped container",
        "caja cuadrada",
        "recipiente cuadrado",
        "contenedor cuadrado",
    ],

    "circular_box": [
        "circular box",
        "circular container",
        "circular receptacle",
        "round box",
        "round container",
        "circle-shaped container",
        "caja circular",
        "recipiente circular",
        "contenedor circular",
        "caja redonda",
    ],
}


def clean_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9ñ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def choose_from_instruction(instruction: str):
    txt = clean_text(instruction)

    obj = None
    for key, aliases in OBJECT_ALIASES.items():
        for a in aliases:
            if clean_text(a) in txt:
                obj = key
                break
        if obj:
            break

    target = None
    for key, aliases in TARGET_ALIASES.items():
        for a in aliases:
            if clean_text(a) in txt:
                target = key
                break
        if target:
            break

    if obj is None:
        raise RuntimeError(
            "No pude identificar el objeto. Usa algo tipo: orange cube / cubo naranja / white cylinder / cilindro blanco."
        )
    if target is None:
        raise RuntimeError(
            "No pude identificar el destino. Usa algo tipo: yellow zone / zona amarilla / blue zone / zona azul."
        )

    return obj, target


def iter_scene_jsons():
    forced = os.environ.get("VLA_FORCE_LIVE_SCENE_JSON", "").strip()
    if forced:
        fp = Path(forced)
        if not fp.exists():
            raise RuntimeError(f"VLA_FORCE_LIVE_SCENE_JSON apunta a un JSON que no existe: {fp}")
        print(f"Usando escena capturada:  {fp}", flush=True)
        return [fp]

    # La búsqueda genérica puede escoger un resultado equivocado:
    # SCENE_ROOT.glob("**/resultado_zonas.json") puede escoger un JSON interno
    # *_cilindro_blanco generado por el detector auxiliar. Si existe el JSON canónico
    # de la última captura multiobjeto, debe ganar siempre.
    try:
        from pathlib import Path as _Path
        import json as _json
        _latest_camera_file = SCENE_ROOT / "ultima_escena.txt"
        if _latest_camera_file.exists():
            _camera_json = _Path(_latest_camera_file.read_text(encoding="utf-8").strip())
            if _camera_json.exists():
                _camera_data = _json.loads(_camera_json.read_text(encoding="utf-8"))
                _camera_objects = _camera_data.get("objects") or {}
                _has_multi = (
                    isinstance(_camera_objects, dict)
                    and "orange_cube" in _camera_objects
                    and "white_cylinder" in _camera_objects
                )
                _fresh_enough = True
                if "start_ns" in locals() and start_ns is not None:
                    _fresh_enough = int(_camera_json.stat().st_mtime_ns) >= int(start_ns)
                if _has_multi and _fresh_enough:
                    print(f"uso JSON canónico de cámara: {_camera_json}", flush=True)
                    return [_camera_json]
                else:
                    print(
                        f"LATEST_CAMERA_SCENE_JSON no válido: "
                        f"has_multi={_has_multi} fresh={_fresh_enough} path={_camera_json}",
                        flush=True,
                    )
    except Exception as _e:
        print(f"no pude usar JSON canónico: {_e}", flush=True)

    paths = list(SCENE_ROOT.glob("**/resultado_zonas.json"))
    paths += list(SCENE_ROOT.glob("**/*scene*.json"))
    paths += list(SCENE_ROOT.glob("**/*result*.json"))

    uniq = []
    seen = set()
    for p in paths:
        if p.is_file() and p not in seen:
            seen.add(p)
            uniq.append(p)

    return sorted(uniq, key=lambda p: p.stat().st_mtime, reverse=True)


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def looks_like_white_cylinder(obj: dict):
    s = json.dumps(obj, ensure_ascii=False).lower()
    return (
        "white_cylinder" in s
        or "white cylinder" in s
        or '"object_color": "white"' in s
        or '"object_shape": "cylinder"' in s
        or '"semantic_override": "white_cylinder"' in s
    )


def looks_like_orange_cube(obj: dict):
    s = json.dumps(obj, ensure_ascii=False).lower()
    return (
        "orange_cube" in s
        or "orange die" in s
        or "orange_die" in s
        or '"object_color": "orange"' in s
        or '"object_shape": "cube' in s
        or '"object_shape": "die' in s
    )


def has_xyz(d: dict):
    if not isinstance(d, dict):
        return False
    for k in ["base_xyz_m", "safe_above_base_xyz_m", "center_base_xyz_m", "xyz_m"]:
        v = d.get(k)
        if isinstance(v, list) and len(v) >= 3:
            return True
    return False


def collect_scene():
    objects = {}
    targets = {}
    sources = {}

    for path in iter_scene_jsons():
        data = load_json(path)
        if not isinstance(data, dict):
            continue

        # Formato general si ya existe.
        for key, val in (data.get("objects") or {}).items():
            if not isinstance(val, dict):
                continue
            lk = key.lower()
            if "white" in lk or "cylinder" in lk or looks_like_white_cylinder(val):
                objects.setdefault("white_cylinder", val)
                sources.setdefault("white_cylinder", str(path))
            elif "orange" in lk or "cube" in lk or "die" in lk or looks_like_orange_cube(val):
                objects.setdefault("orange_cube", val)
                sources.setdefault("orange_cube", str(path))

        for key, val in (data.get("targets") or {}).items():
            if not isinstance(val, dict):
                continue
            lk = key.lower()
            if "yellow" in lk:
                targets.setdefault("yellow_zone", val)
                sources.setdefault("yellow_zone", str(path))
            elif "blue" in lk:
                targets.setdefault("blue_zone", val)
                sources.setdefault("blue_zone", str(path))

        # Compatibilidad con scripts actuales.
        if isinstance(data.get("orange_die"), dict):
            obj = data["orange_die"]
            if looks_like_white_cylinder(obj):
                objects.setdefault("white_cylinder", obj)
                sources.setdefault("white_cylinder", str(path))
            else:
                objects.setdefault("orange_cube", obj)
                sources.setdefault("orange_cube", str(path))

        for k in ["white_cylinder", "cylinder"]:
            if isinstance(data.get(k), dict):
                objects.setdefault("white_cylinder", data[k])
                sources.setdefault("white_cylinder", str(path))

        for k in ["orange_cube", "orange_object", "orange_die"]:
            if isinstance(data.get(k), dict) and not looks_like_white_cylinder(data[k]):
                objects.setdefault("orange_cube", data[k])
                sources.setdefault("orange_cube", str(path))

        for k in ["yellow_target", "yellow_zone", "yellow_box"]:
            if isinstance(data.get(k), dict):
                targets.setdefault("yellow_zone", data[k])
                sources.setdefault("yellow_zone", str(path))

        for k in ["blue_target", "blue_zone", "blue_box"]:
            if isinstance(data.get(k), dict):
                targets.setdefault("blue_zone", data[k])
                sources.setdefault("blue_zone", str(path))

    # Filtrado mínimo: no aceptar entidades sin coordenadas.
    objects = {k: v for k, v in objects.items() if has_xyz(v)}
    targets = {k: v for k, v in targets.items() if has_xyz(v)}

    # Alternativa sencilla para la demo:
    # Si no existe blue_zone explícita, se toma la zona inicial de objetos como zona azul.
    # En modo live esto no se permite porque, si el objeto se ha movido,
    # la "zona inicial" pasaría a ser erróneamente la posición actual del objeto.
    if "blue_zone" not in targets and objects:
        forced_live = bool(os.environ.get("VLA_FORCE_LIVE_SCENE_JSON", "").strip())
        requested_target = (
            os.environ.get("VLA_SELECTED_TARGET", "")
            or os.environ.get("VLA_TARGET", "")
        ).strip()

        if forced_live and requested_target == "blue_zone":
            raise RuntimeError(
                "Escena live sin blue_zone explícita. "
                "No infiero blue_zone desde la posición actual del objeto."
            )

        import copy
        import numpy as np

        pts = []
        for obj in objects.values():
            for key in ["base_xyz_m", "center_base_xyz_m", "xyz_m", "safe_above_base_xyz_m"]:
                v = obj.get(key)
                if isinstance(v, list) and len(v) >= 3:
                    pts.append([float(v[0]), float(v[1]), float(v[2])])
                    break

        if pts:
            pts = np.array(pts, dtype=float)
            c = pts.mean(axis=0).tolist()

            # Se usa una z baja; PLACE_Z_OVERRIDE_M fija después la altura real de place.
            c[2] = min(float(c[2]), 0.04)

            targets["blue_zone"] = {
                "base_xyz_m": c,
                "center_base_xyz_m": c,
                "object_color": "blue",
                "target_color": "blue",
                "selected_as": "blue_zone",
                "source": "demo_fallback_from_initial_object_area",
                "note": "Zona azul inferida como zona inicial de los objetos porque no había blue_zone explícita en los JSON."
            }
            sources["blue_zone"] = "demo_fallback_from_initial_object_area"

    return objects, targets, sources




def _first_xyz(entity):
    for key in ["base_xyz_m", "center_base_xyz_m", "xyz_m", "safe_above_base_xyz_m"]:
        v = entity.get(key)
        if isinstance(v, list) and len(v) >= 3:
            return [float(v[0]), float(v[1]), float(v[2])]
    raise RuntimeError(f"Entidad sin XYZ usable: {entity}")


def _normalize_for_executor(entity, kind):
    """
    El ejecutor espera siempre:
        - base_xyz_m
        - safe_above_base_xyz_m
    
        Aunque la interfaz use blue_zone/yellow_zone/cylinder/cube,
        por compatibilidad interna se entregan esas claves.
    """
    e = dict(entity)
    base = _first_xyz(e)

    if "base_xyz_m" not in e:
        e["base_xyz_m"] = base

    if "center_base_xyz_m" not in e:
        e["center_base_xyz_m"] = base

    if "safe_above_base_xyz_m" not in e:
        safe = list(base)
        if kind == "target":
            safe[2] = float(base[2]) + 0.200
        else:
            safe[2] = float(base[2]) + 0.170
        e["safe_above_base_xyz_m"] = safe

    return e


def make_scene(instruction: str, object_key: str, target_key: str):
    objects, targets, sources = collect_scene()

    if object_key not in objects:
        raise RuntimeError(
            f"No encuentro {object_key} en los JSON de escena. Encontrados: {sorted(objects.keys())}"
        )
    if target_key not in targets:
        raise RuntimeError(
            f"No encuentro {target_key} en los JSON de escena. Encontrados: {sorted(targets.keys())}"
        )

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = SCENE_ROOT / f"zonas_{ts}_flujo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "resultado_zonas.json"

    selected_obj = _normalize_for_executor(objects[object_key], "object")
    selected_target = _normalize_for_executor(targets[target_key], "target")

    selected_obj["selected_as"] = object_key
    selected_target["selected_as"] = target_key

    scene = {
        "json_path": str(out_json),
        "task": "demo",
        "instruction": instruction,
        "selected_object": object_key,
        "selected_target": target_key,

        "objects": objects,
        "targets": targets,
        "sources": sources,

        # El ejecutor mantiene estas claves por compatibilidad.
        # Internamente sigue leyendo orange_die/yellow_target.
        "orange_die": selected_obj,
        "yellow_target": selected_target,

        "demo_note": (
            "Escena adaptada por flujo.py. La interfaz usa nombres generales, "
            "pero se mantienen claves internas orange_die/yellow_target por compatibilidad."
        ),
    }

    out_json.write_text(json.dumps(scene, indent=2, ensure_ascii=False), encoding="utf-8")
    os.utime(out_dir, None)
    os.utime(out_json, None)

    return out_json, scene


def profile_for_model(model: str):
    p = PROFILE_DIR / f"{model}.json"
    if not p.exists():
        raise RuntimeError(f"No existe perfil limpio: {p}")
    return p


def run_executor(model: str, instruction: str, real: bool):
    executor_path = Path(
        os.environ.get(
            "VLA_EXECUTOR_OVERRIDE",
            str(EXECUTOR),
        )
    ).expanduser().resolve()

    if not executor_path.exists():
        raise RuntimeError(
            f"No existe ejecutor: {executor_path}"
        )

    object_key, target_key = choose_from_instruction(instruction)
    scene_json, scene = make_scene(instruction, object_key, target_key)

    profile = profile_for_model(model)

    run_id = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%S_%fZ")
        + f"_{model}"
    )

    out_dir = LOG_ROOT / run_id

    out_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    log_path = out_dir / f"{model}.log"

    print(
        f"{out_dir}",
        flush=True,
    )

    # El entorno del ejecutor se construye después
    # de fijar definitivamente la escena.

    cmd = [
        sys.executable,
        str(executor_path),
    ]

    header = {
        "model": model,
        "instruction": instruction,
        "real": real,
        "object": object_key,
        "target": target_key,
        "scene_json": str(scene_json),
        "profile": os.environ.get("VLA_VLA_GUIDANCE_PROFILE_OVERRIDE", "").strip() or str(profile),
        "log_path": str(log_path),
    }


    # Si se usa el detector multiobjeto, scene_json no debe acabar
    # apuntando al JSON interno *_cilindro_blanco. Debe apuntar al escena_*.
    try:
        _detector_s = str(detector) if "detector" in locals() else ""
        if "vision_escena.py" in _detector_s:
            _all_json = _latest_all_objects_scene_json(
                start_ns=start_ns if "start_ns" in locals() else None,
                selected_object=obj_key if "obj_key" in locals() else None,
            )
            if _all_json is None:
                raise RuntimeError("No encuentro JSON escena multiobjeto reciente tras detector multiobjeto")
            if "escena_" not in str(_all_json):
                raise RuntimeError(f"JSON multiobjeto inválido: {_all_json}")
            scene_json = _all_json
            print(f"usando escena multiobjeto exacta: {scene_json}", flush=True)
    except Exception as _e:
        raise RuntimeError(f"fallo resolviendo escena multiobjeto exacta: {_e}")
    print("Flujo de ejecución")
    print(json.dumps(header, indent=2, ensure_ascii=False))
    print()

    with log_path.open("w", encoding="utf-8") as f:

        # Si se ejecuta el detector multiobjeto, se usa siempre el JSON canónico
        # escrito por la última captura de cámara. No buscar "último JSON" genérico.
        try:
            from pathlib import Path as _Path
            import json as _json
            _latest_scene_file = _Path("results/validacion_robot/ultima_escena.txt")
            if _latest_scene_file.exists():
                _camera_scene_json = _Path(_latest_scene_file.read_text(encoding="utf-8").strip())
                if not _camera_scene_json.exists():
                    raise RuntimeError(f"LATEST_CAMERA_SCENE_JSON apunta a un archivo inexistente: {_camera_scene_json}")
                _scene_check = _json.loads(_camera_scene_json.read_text(encoding="utf-8"))
                _objs_check = _scene_check.get("objects") or {}
                if not isinstance(_objs_check, dict) or "orange_cube" not in _objs_check or "white_cylinder" not in _objs_check:
                    raise RuntimeError(f"JSON canónico sin escena multiobjeto completa: {_camera_scene_json} objects={list(_objs_check.keys()) if isinstance(_objs_check, dict) else _objs_check}")
                scene_json = _camera_scene_json
                print(f"scene_json={scene_json}", flush=True)
        except Exception as _e:
            raise RuntimeError(f"{_e}")

        # Antes de construir o ejecutar el caso se usa el JSON canónico
        # producido por la última captura de cámara multiobjeto.
        try:
            from pathlib import Path as _Path
            import json as _json
            _latest_scene_file = _Path("results/validacion_robot/ultima_escena.txt")
            if not _latest_scene_file.exists():
                raise RuntimeError("No existe ultima_escena.txt; el detector multiobjeto no publicó escena canónica")
            _camera_scene_json = _Path(_latest_scene_file.read_text(encoding="utf-8").strip())
            if not _camera_scene_json.exists():
                raise RuntimeError(f"LATEST_CAMERA_SCENE_JSON apunta a archivo inexistente: {_camera_scene_json}")
            _scene_check = _json.loads(_camera_scene_json.read_text(encoding="utf-8"))
            _objs_check = _scene_check.get("objects") or {}
            if not isinstance(_objs_check, dict) or "orange_cube" not in _objs_check or "white_cylinder" not in _objs_check:
                raise RuntimeError(f"JSON canónico sin objetos completos: {_camera_scene_json} objects={list(_objs_check.keys()) if isinstance(_objs_check, dict) else _objs_check}")
            scene_json = _camera_scene_json
            print(f"scene_json={scene_json}", flush=True)
        except Exception as _e:
            raise RuntimeError(f"{_e}")
        # Contrato canónico escena -> perfil -> planificación
        # La escena ya no puede cambiar después de este punto.
        resolved_scene_json = Path(
            scene_json
        ).expanduser().resolve()

        if not resolved_scene_json.is_file():
            raise RuntimeError(
                ""
                f"No existe la escena de cierre: "
                f"{resolved_scene_json}"
            )

        guard_output_dir = (
            out_dir
            / (
                "scene_guard_"
                f"{os.getpid()}_"
                f"{time.time_ns()}"
            )
        )

        guard_output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        experiment_family = os.environ.get(
            "VLA_EXPERIMENT_FAMILY",
            "canonical",
        ).strip()

        if (
            real
            and experiment_family != "comparison"
        ):
            raise RuntimeError(
                ""
                "La familia canónica full-output "
                "no autoriza movimiento real."
            )

        if real:
            planning_state_mode = "live_robot"
            scene_guard_mode = "live_validation"
        else:
            planning_state_mode = "offline_snapshot"
            scene_guard_mode = "offline_validation"

        env = os.environ.copy()

        # Las rutas antiguas no compiten con VLA_SCENE_JSON.
        for compat_name in (
            "DETECTED_SCENE_JSON",
            "VLA_FORCE_LIVE_SCENE_JSON",
            "VLA_LIVE_OBJECTS_SCENE_JSON",
        ):
            env.pop(
                compat_name,
                None,
            )

        env.update({
            "RMW_FASTRTPS_USE_SHM": "0",
            "ROS_DOMAIN_ID": "0",
            "ROS_LOCALHOST_ONLY": "0",
            "MUJOCO_GL": "osmesa",

            "DRY_RUN": "0" if real else "1",
            "CONFIRM_PHYSICAL_EXECUTION": "YES" if real else "",
            "ALLOW_PHYSICAL_EXECUTION": "YES" if real else "",
            "CONFIRM_EXECUTION_RISK": "YES" if real else "",

            "EXECUTE_PLACE": "1",
            "VLA_SELECTED_OBJECT": object_key,
            "VLA_SELECTED_TARGET": target_key,
            "VLA_TARGET": target_key,

            "OPEN_WIDTH_M": "0.070",
            "CLOSE_WIDTH_M": "0.020",
            "MIN_PICK_Z_M": "0.045",
            "PICK_Z_OVERRIDE_M": (
                os.environ.get(
                    "PICK_Z_OVERRIDE_M",
                    "0.040",
                ).strip()
                or "0.040"
            ),
            "PLACE_Z_OVERRIDE_M": "0.060",
            "LIFT_DELTA_M": "0.160",
            "PICK_MAX_ERROR_M": "0.035",

            "VLA_GUIDANCE_PROFILE": (
                os.environ.get(
                    "VLA_VLA_GUIDANCE_PROFILE_OVERRIDE",
                    "",
                ).strip()
                or str(profile)
            ),

            # Contrato inmutable del ejecutor canónico.
            "VLA_SCENE_JSON": str(
                resolved_scene_json
            ),
            "VLA_SCENE_GUARD_OUTPUT_DIR": str(
                guard_output_dir
            ),
            "VLA_SCENE_GUARD_MODE": scene_guard_mode,
            "VLA_PLANNING_STATE_MODE": (
                planning_state_mode
            ),
        })

        if real:
            env.pop(
                "VLA_PLANNING_STATE_SNAPSHOT",
                None,
            )
        else:
            planning_snapshot = Path(
                "results/"
                "planning_state_snapshots/"
                "home_state.json"
            ).resolve()

            if not planning_snapshot.is_file():
                raise RuntimeError(
                    ""
                    "No existe el snapshot offline: "
                    f"{planning_snapshot}"
                )

            env[
                "VLA_PLANNING_STATE_SNAPSHOT"
            ] = str(planning_snapshot)

        # El encabezado debe registrar lo que realmente
        # recibe el subprocess, no la escena preliminar.
        header["scene_json"] = str(
            resolved_scene_json
        )
        header["executor"] = str(
            executor_path
        )
        header["scene_guard_output_dir"] = str(
            guard_output_dir
        )
        header["planning_state_mode"] = (
            planning_state_mode
        )
        header["canonical_scene_contract"] = True

        f.write("Flujo de ejecución\n")
        f.write(json.dumps(header, indent=2, ensure_ascii=False))
        f.write("\n\n")

        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)

        rc = proc.wait()

    def artifact_record(path):
        path = Path(
            path
        ).expanduser().resolve()

        record = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": None,
            "size_bytes": None,
        }

        if path.is_file():
            record["sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

            record["size_bytes"] = int(
                path.stat().st_size
            )

        return record

    profile_path = Path(
        header["profile"]
    ).expanduser().resolve()

    plan_path = (
        resolved_scene_json.parent
        / "plan_ejecucion.json"
    ).resolve()

    manifest = {
        "schema": (
            "evaluation_run"
        ),
        "run_id": run_id,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "model": model,
        "instruction": instruction,
        "task_specification": {
            "expected_object": object_key,
            "expected_target": target_key,
            "source": (
                "experiment_instruction_parser"
            ),
            "not_model_prediction": True,
        },
        "experiment_profile": (
            os.environ.get(
                "VLA_EXPERIMENT_MODE",
                "",
            )
        ),
        "experiment_family": (
            os.environ.get(
                "VLA_EXPERIMENT_FAMILY",
                "",
            )
        ),
        "vla_source": (
            os.environ.get(
                "VLA_SOURCE",
                "",
            )
        ),
        "mode": (
            "real"
            if real
            else "dry-run"
        ),
        "robot_motion_requested": bool(real),
        "executor_return_code": int(rc),
        "executor_completed": bool(rc == 0),
        "physical_task_success": None,
        "failure_stage": None,
        "outcome_pending_annotation": True,
        "eligible_for_offline_metrics": bool(
            rc == 0
        ),
        "eligible_for_physical_success_metrics": False,
        "artifacts": {
            "scene": artifact_record(
                resolved_scene_json
            ),
            "profile": artifact_record(
                profile_path
            ),
            "executor": artifact_record(
                executor_path
            ),
            "plan": artifact_record(
                plan_path
            ),
            "log": artifact_record(
                log_path
            ),
        },
    }

    manifest_path = (
        out_dir
        / "run_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"{manifest_path}",
        flush=True,
    )
    print(f"log: {log_path}")
    print(f"Código de salida: {rc}")

    return rc


def list_scene():
    objects, targets, sources = collect_scene()

    print("Objetos detectados")
    for k in sorted(objects):
        print(f"- {k}: {sources.get(k)}")

    print()
    print("Destinos detectados")
    for k in sorted(targets):
        print(f"- {k}: {sources.get(k)}")



def _latest_all_objects_scene_json(start_ns=None, selected_object=None):
    """
    Devuelve el JSON multiobjeto escena multiobjeto más reciente.
    
    vision_escena.py también ejecuta detectores que escriben
    resultado_zonas.json. Se prioriza escena_* para
    mantener la escena multiobjeto correcta.
    """
    from pathlib import Path
    import json

    roots = [
        Path("results/validacion_robot"),
        Path("results/validacion_robot"),
    ]

    candidates = []

    for root in roots:
        if not root.exists():
            continue

        for j in root.glob("escena_*/resultado_zonas.json"):
            try:
                st = j.stat()
                if start_ns is not None and int(st.st_mtime_ns) < int(start_ns):
                    continue

                d = json.loads(j.read_text(encoding="utf-8"))
                objs = d.get("objects") or {}

                if not isinstance(objs, dict) or not objs:
                    continue

                if selected_object in ("orange_cube", "white_cylinder"):
                    if selected_object not in objs:
                        continue

                candidates.append((st.st_mtime_ns, j, d))
            except Exception:
                continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["openvla", "pi0"], default="openvla")
    ap.add_argument("--instruction", default="")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_scene()
        return 0

    if not args.instruction.strip():
        raise RuntimeError("Falta --instruction")

    real = bool(args.real and not args.dry_run)
    return run_executor(args.model, args.instruction, real)




# Antes de dry-run/real se solicita una captura nueva.

# Se toma una captura Photoneo/PhoXi antes de planificar.
# Invariante:
# - se llama al servidor host.
# - se genera un PLY nuevo.
# - la captura procede de TriggerFrame Software.
# - no se usa Freerun/GetFrame como alternativa.
# - la evidencia se guarda junto al registro de la ejecución.

def _load_photoneo_meta():
    meta_path = Path("data/photoneo/live_capture.json")
    if not meta_path.exists():
        raise RuntimeError(f"No existe metadata Photoneo obligatoria: {meta_path}")
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"No se pudo leer metadata Photoneo obligatoria: {e}")


def _build_photoneo_proof_text(meta: dict, client_output: str) -> str:
    capture_meta = meta.get("capture_meta") or {}
    validation = meta.get("_client_validation") or {}
    stdout_tail = str(meta.get("stdout_tail") or "")

    lines = []
    lines.append("")
    lines.append("=== PHOTONEO MANDATORY CAPTURE PROOF ===")
    lines.append(f"metadata_path: data/photoneo/live_capture.json")
    lines.append(f"ok: {meta.get('ok')}")
    lines.append(f"returncode: {meta.get('returncode')}")
    lines.append(f"timestamp_local: {capture_meta.get('timestamp_local')}")
    lines.append(f"device_id: {capture_meta.get('device_id')}")
    lines.append(f"message: {capture_meta.get('message')}")
    lines.append(f"ply_path_host: {meta.get('ply_path_host')}")
    lines.append(f"ply_path_docker: {meta.get('ply_path_docker')}")
    lines.append(f"ply_size_bytes: {meta.get('ply_size_bytes')}")
    lines.append(f"mandatory_new_capture: {validation.get('mandatory_new_capture')}")
    lines.append(f"ply_mtime_ns_before: {validation.get('ply_mtime_ns_before')}")
    lines.append(f"ply_mtime_ns_after: {validation.get('ply_mtime_ns_after')}")
    lines.append("")
    lines.append("--- client_output_tail ---")
    lines.append(client_output[-4000:])
    lines.append("")
    lines.append("--- photoneo_stdout_tail ---")
    lines.append(stdout_tail[-4000:])
    lines.append("=== END PHOTONEO MANDATORY CAPTURE PROOF ===")
    lines.append("")
    return "\n".join(lines)


def _validate_photoneo_mandatory_capture(meta: dict, client_output: str):
    capture_meta = meta.get("capture_meta") or {}
    validation = meta.get("_client_validation") or {}
    stdout_tail = str(meta.get("stdout_tail") or "")
    all_text = "\n".join([
        str(client_output or ""),
        str(stdout_tail or ""),
        json.dumps(meta, ensure_ascii=False),
    ])

    required = [
        "StartAcquisition Software",
        "TriggerFrame obligatorio",
        "TriggerFrame frame_id=",
        "Captura Photoneo/PhoXi correcta por TriggerFrame Software obligatorio",
    ]

    missing = [x for x in required if x not in all_text]
    if missing:
        raise RuntimeError(
            "Captura Photoneo inválida: faltan evidencias de trigger obligatorio: "
            + ", ".join(missing)
        )

    forbidden = [
        "Probando fallback Freerun/GetFrame",
        "fallback Freerun",
        "TriggerFrame falló",
    ]
    bad = [x for x in forbidden if x in all_text]
    if bad:
        raise RuntimeError(
            "Captura Photoneo inválida: aparece fallback o fallo de trigger: "
            + ", ".join(bad)
        )

    if meta.get("ok") is not True:
        raise RuntimeError("Captura Photoneo inválida: meta.ok != true")

    if capture_meta.get("ok") is not True:
        raise RuntimeError("Captura Photoneo inválida: capture_meta.ok != true")

    if validation.get("mandatory_new_capture") is not True:
        raise RuntimeError("Captura Photoneo inválida: no consta mandatory_new_capture=true")

    before = validation.get("ply_mtime_ns_before")
    after = validation.get("ply_mtime_ns_after")
    if before is not None and after is not None and int(after) <= int(before):
        raise RuntimeError("Captura Photoneo inválida: el PLY no es posterior al anterior")



def _scene_xyz(data: dict, obj_key: str):
    candidates = []

    if obj_key == "white_cylinder":
        candidates += [
            ("objects.white_cylinder", (data.get("objects") or {}).get("white_cylinder")),
            ("white_cylinder", data.get("white_cylinder")),
            ("cylinder", data.get("cylinder")),
            ("orange_die", data.get("orange_die")),
        ]
    else:
        candidates += [
            ("objects.orange_cube", (data.get("objects") or {}).get("orange_cube")),
            ("orange_cube", data.get("orange_cube")),
            ("orange_die", data.get("orange_die")),
        ]

    for label, obj in candidates:
        if not isinstance(obj, dict):
            continue
        v = obj.get("base_xyz_m") or obj.get("center_base_xyz_m")
        if isinstance(v, list) and len(v) >= 3:
            return label, [float(v[0]), float(v[1]), float(v[2])]

    raise RuntimeError(f"No pude extraer base_xyz_m para {obj_key} en escena live")


def _find_recent_live_scene_json(obj_key: str, start_ns: int):
    if (
        __import__("os").environ.get(
            "VLA_TASK_FAMILY", ""
        ).strip() == "shape_classification"
    ):
        import json as _shape_json
        from pathlib import Path as _ShapePath

        _shape_roots = [
            _ShapePath(
                "results/validacion_robot"
            ),
            _ShapePath(
                "results/"
                "validacion_robot"
            ),
        ]

        _shape_candidates = []

        for _shape_root in _shape_roots:
            if not _shape_root.is_dir():
                continue

            for _shape_json_path in _shape_root.glob(
                "forma_*/"
                "resultado_forma.json"
            ):
                try:
                    if (
                        _shape_json_path.stat().st_mtime_ns
                        < int(start_ns)
                    ):
                        continue

                    _shape_data = _shape_json.loads(
                        _shape_json_path.read_text(
                            encoding="utf-8"
                        )
                    )

                    _shape_serialized = _shape_json.dumps(
                        _shape_data,
                        ensure_ascii=False,
                    )

                    _shape_required = (
                        "orange_cube",
                        "white_cylinder",
                        "square_box",
                        "circular_box",
                    )

                    if not all(
                        _name in _shape_serialized
                        for _name in _shape_required
                    ):
                        continue

                    _shape_candidates.append(
                        _shape_json_path
                    )

                except Exception:
                    continue

        if _shape_candidates:
            _shape_latest = max(
                _shape_candidates,
                key=lambda _item:
                    _item.stat().st_mtime_ns,
            )

            print(
                f"{_shape_latest}",
                flush=True,
            )

            return _shape_latest

        raise RuntimeError(
            "La detección shape se ejecutó, pero no existe "
            "un JSON reciente que contenga orange_cube, "
            "white_cylinder, square_box y circular_box"
        )

    """
    Busca el JSON live recién generado aceptando aliases semánticos.

    Caso importante:
    - el pipeline puede pedir obj_key='orange_cube'
    - el detector vision_zonas.py genera top-level 'orange_die'
    - por tanto orange_cube y orange_die deben considerarse equivalentes
    """
    import json
    from pathlib import Path

    def xyz_of(obj):
        if not isinstance(obj, dict):
            return None
        for k in ("base_xyz_m", "center_base_xyz_m", "safe_above_base_xyz_m", "xyz_m"):
            v = obj.get(k)
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                try:
                    return [float(v[0]), float(v[1]), float(v[2])]
                except Exception:
                    pass
        return None

    def has_selected_object(data, key):
        objects = data.get("objects") or {}

        if key in ("orange_cube", "orange_die"):
            candidates = [
                objects.get("orange_cube"),
                objects.get("orange_die"),
                data.get("orange_cube"),
                data.get("orange_die"),
            ]

        elif key == "white_cylinder":
            candidates = [
                objects.get("white_cylinder"),
                data.get("white_cylinder"),
                data.get("cylinder"),
            ]

        else:
            candidates = [
                objects.get(key),
                data.get(key),
            ]

        return any(xyz_of(c) is not None for c in candidates)

    roots = []

    for r in [
        Path("results/validacion_robot"),
        Path("results/validacion_robot"),
    ]:
        if r.exists() and r not in roots:
            roots.append(r)

    try:
        if "SCENE_ROOT" in globals():
            sr = Path(SCENE_ROOT)
            if sr.exists() and sr not in roots:
                roots.append(sr)
    except Exception:
        pass

    candidates = []
    recent_diag = []

    for root in roots:
        for q in root.rglob("resultado_zonas.json"):
            try:
                st = q.stat()
            except Exception:
                continue

            # Se deja un margen amplio porque algunas salidas pueden diferir ligeramente.
            if st.st_mtime_ns < start_ns - 10_000_000_000:
                continue

            recent_diag.append((st.st_mtime_ns, str(q)))

            try:
                data = json.loads(q.read_text(encoding="utf-8"))
            except Exception:
                continue

            if has_selected_object(data, obj_key):
                candidates.append(q)

    if candidates:
        candidates = sorted(candidates, key=lambda q: q.stat().st_mtime_ns, reverse=True)
        return candidates[0]

    recent_diag = sorted(recent_diag, reverse=True)[:12]
    msg = [
        f"No encontré JSON live reciente generado para {obj_key}",
        "JSON recientes vistos:",
    ]
    for _, q in recent_diag:
        msg.append(f"  - {q}")

    raise RuntimeError("\n".join(msg))


def _target_xyz(obj):
    if not isinstance(obj, dict):
        return None
    for k in ["base_xyz_m", "center_base_xyz_m", "xyz_m", "safe_above_base_xyz_m"]:
        v = obj.get(k)
        if isinstance(v, list) and len(v) >= 3:
            return [float(v[0]), float(v[1]), float(v[2])]
    return None


def _looks_blue_target(obj):
    if not isinstance(obj, dict):
        return False
    txt = json.dumps(obj, ensure_ascii=False).lower()
    return (
        '"target_color": "blue"' in txt
        or '"object_color": "blue"' in txt
        or "blue_zone" in txt
        or "zona azul" in txt
    )


def _dist_xy(a, b):
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5


def _extract_blue_candidates_from_data(data: dict):
    targets = data.get("targets") or {}

    raw = [
        ("data.blue_zone", data.get("blue_zone")),
        ("targets.blue_zone", targets.get("blue_zone")),
        ("data.yellow_target_if_blue", data.get("yellow_target")),
        ("targets.yellow_target_if_blue", targets.get("yellow_target")),
        ("targets.yellow_zone_if_blue", targets.get("yellow_zone")),
    ]

    out = []
    for label, obj in raw:
        if not isinstance(obj, dict):
            continue

        xyz = _target_xyz(obj)
        if xyz is None:
            continue

        if not _looks_blue_target(obj):
            continue

        out.append((label, dict(obj), xyz))

    return out


def _find_stable_blue_zone(object_xyz, current_scene_json: Path):
    """
    Devuelve una referencia estable de blue_zone.
    
        - Primero usa solo data.blue_zone / targets.blue_zone explícitos.
        - No usa yellow_target/yellow_zone contaminados de ejecuciones live previas.
        - No rechaza blue_zone por estar cerca del objeto, porque el objeto puede estar ya dentro de la zona azul.
    """
    groups = {}

    def add_candidate(q, label, obj, allow_alias=False):
        if not isinstance(obj, dict):
            return

        xyz = _target_xyz(obj)
        if xyz is None:
            return

        if not _looks_blue_target(obj):
            return

        txt = json.dumps(obj, ensure_ascii=False).lower()

        # Evitar targets que sabemos que fueron generados como alias live contaminado.
        banned = [
            "live_target_alias_from_blue_marked_target",
            "stable_blue_zone_from_history_far_from_object",
            "normalizado desde escena live",
            "no inferido desde la posición actual del objeto",
        ]
        if any(b in txt for b in banned):
            return

        # Si no es blue_zone explícito, solo se acepta como último recurso.
        if not allow_alias and "blue_zone" not in label:
            return

        key = (round(float(xyz[0]), 3), round(float(xyz[1]), 3), round(float(xyz[2]), 3))
        g = groups.setdefault(key, {
            "count": 0,
            "mtime": 0.0,
            "xyz": xyz,
            "obj": dict(obj),
            "examples": [],
            "explicit_blue_zone": "blue_zone" in label,
        })
        g["count"] += 1
        try:
            g["mtime"] = max(g["mtime"], q.stat().st_mtime)
        except Exception:
            pass
        g["examples"].append(f"{label} @ {q}")

    # Primera pasada: solo blue_zone explícita.
    for q in SCENE_ROOT.rglob("resultado_zonas.json"):
        try:
            if q.resolve() == current_scene_json.resolve():
                continue
            data = json.loads(q.read_text(encoding="utf-8"))
        except Exception:
            continue

        targets = data.get("targets") or {}
        add_candidate(q, "data.blue_zone", data.get("blue_zone"), allow_alias=False)
        add_candidate(q, "targets.blue_zone", targets.get("blue_zone"), allow_alias=False)

    if not groups:
        raise RuntimeError(
            "No encontré ninguna referencia explícita targets.blue_zone/data.blue_zone. "
            "No usaré yellow_target/yellow_zone como blue_zone para evitar contaminación."
        )

    best = sorted(
        groups.values(),
        key=lambda g: (1 if g.get("explicit_blue_zone") else 0, g["count"], g["mtime"]),
        reverse=True,
    )[0]

    print(
        "blue_zone explícita elegida: "
        f"{best['xyz']} count={best['count']} example={best['examples'][0]}",
        flush=True,
    )

    return dict(best["obj"]), best["xyz"]



def _ensure_live_target_alias(
    data: dict,
    target_key: str,
    scene_json: Path,
):
    # La zona azul se obtiene solo de la
    # escena live actual. Se elimina la búsqueda en
    # ejecuciones históricas y no se sobrescribe
    # yellow_target con datos azules.
    import copy
    import json
    import math

    if target_key != "blue_zone":
        return data

    if not isinstance(data, dict):
        raise RuntimeError(
            ""
            "La escena live no es un diccionario"
        )

    targets = data.setdefault(
        "targets",
        {},
    )

    if not isinstance(targets, dict):
        raise RuntimeError(
            ""
            "targets no es un diccionario"
        )

    def extract_xyz(
        record,
        label,
    ):
        if not isinstance(record, dict):
            return None

        value = (
            record.get("base_xyz_m")
            or record.get(
                "center_base_xyz_m"
            )
            or record.get("xyz_m")
            or record.get("centroid_base_m")
        )

        if not isinstance(
            value,
            (list, tuple),
        ):
            return None

        if len(value) < 3:
            return None

        xyz = [
            float(value[0]),
            float(value[1]),
            float(value[2]),
        ]

        if not all(
            math.isfinite(component)
            for component in xyz
        ):
            raise RuntimeError(
                ""
                f"{label} contiene xyz no finito: "
                f"{xyz}"
            )

        return xyz

    def is_historical(record):
        if not isinstance(record, dict):
            return False

        provenance = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
        ).lower()

        forbidden = (
            "from_history",
            "historical",
            "stable_explicit_blue_zone",
            "stable_blue_zone_from_history",
        )

        return any(
            token in provenance
            for token in forbidden
        )

    # blue_source es la superficie azul detectada
    # por el pipeline live actual. Tiene prioridad
    # sobre cualquier alias blue_zone preexistente.
    source_candidates = [
        (
            "targets.blue_source",
            targets.get("blue_source"),
        ),
        (
            "data.blue_source",
            data.get("blue_source"),
        ),
    ]

    explicit_candidates = [
        (
            "targets.blue_zone",
            targets.get("blue_zone"),
        ),
        (
            "data.blue_zone",
            data.get("blue_zone"),
        ),
    ]

    valid_sources = []

    for label, record in source_candidates:
        xyz = extract_xyz(
            record,
            label,
        )

        if xyz is None:
            continue

        valid_sources.append(
            (
                label,
                record,
                xyz,
            )
        )

    valid_explicit = []

    for label, record in explicit_candidates:
        if is_historical(record):
            print(
                f"descartado destino histórico: "
                f"{label}",
                flush=True,
            )
            continue

        xyz = extract_xyz(
            record,
            label,
        )

        if xyz is None:
            continue

        valid_explicit.append(
            (
                label,
                record,
                xyz,
            )
        )

    candidates = (
        valid_sources
        if valid_sources
        else valid_explicit
    )

    if not candidates:
        raise RuntimeError(
            ""
            "La captura actual no contiene "
            "blue_source ni blue_zone con "
            "coordenadas válidas. No se utilizará "
            "ninguna coordenada histórica."
        )

    reference_xyz = candidates[0][2]

    for label, _, xyz in candidates[1:]:
        distance = math.sqrt(
            sum(
                (
                    float(xyz[index])
                    - float(
                        reference_xyz[index]
                    )
                )
                ** 2
                for index in range(3)
            )
        )

        if distance > 0.005:
            raise RuntimeError(
                ""
                "La escena actual contiene "
                "referencias azules incompatibles: "
                f"{candidates[0][0]}="
                f"{reference_xyz}, "
                f"{label}={xyz}, "
                f"distancia={distance:.4f}m"
            )

    chosen_label, chosen_record, xyz = (
        candidates[0]
    )

    blue = copy.deepcopy(
        chosen_record
    )

    blue["base_xyz_m"] = [
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),
    ]

    blue["center_base_xyz_m"] = [
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),
    ]

    blue["safe_above_base_xyz_m"] = [
        float(xyz[0]),
        float(xyz[1]),
        max(
            float(xyz[2]) + 0.20,
            0.24,
        ),
    ]

    blue["target_color"] = "blue"
    blue["selected_as"] = "blue_zone"
    blue["canonical_target"] = (
        "blue_zone"
    )
    blue["scene_role"] = (
        "destination_zone"
    )
    blue["source"] = (
        "current_live_scene"
    )
    blue["source_record"] = (
        chosen_label
    )
    blue["source_scene_json"] = str(
        scene_json
    )
    blue["historical_fallback"] = False

    # Se conserva blue_source como elemento de
    # escena, pero solo blue_zone es el destino
    # canónico consumido por el supervisor.
    targets["blue_zone"] = blue
    data["blue_zone"] = blue

    data["selected_target"] = (
        "blue_zone"
    )
    data["target_key"] = "blue_zone"
    data["target_resolved_key"] = (
        "blue_zone"
    )

    # No escribir blue_zone sobre yellow_target.
    # yellow_target debe seguir representando
    # solo la zona amarilla.
    scene_json.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "blue_zone actual validada: "
        f"source={chosen_label} "
        f"xyz={blue['base_xyz_m']} "
        "historical_fallback=false",
        flush=True,
    )

    return data




def _find_recent_orange_scene_json(start_ns: int):
    roots = [
        SCENE_ROOT,
        Path("results/validacion_robot"),
        Path("results/validacion_robot"),
    ]

    candidates = []

    for root in roots:
        if not root.exists():
            continue

        for q in root.rglob("resultado_zonas.json"):
            try:
                st = q.stat()
            except Exception:
                continue

            if st.st_mtime_ns < start_ns - 2_000_000_000:
                continue

            try:
                data = json.loads(q.read_text(encoding="utf-8"))
            except Exception:
                continue

            txt = json.dumps(data, ensure_ascii=False).lower()

            if "orange" not in txt:
                continue

            candidates.append(q)

    if not candidates:
        return None

    return sorted(candidates, key=lambda q: q.stat().st_mtime_ns, reverse=True)[0]


def _extract_orange_obstacle(data: dict):
    candidates = [
        ("objects.orange_cube", (data.get("objects") or {}).get("orange_cube")),
        ("orange_cube", data.get("orange_cube")),
        ("orange_die", data.get("orange_die")),
    ]

    for label, obj in candidates:
        if not isinstance(obj, dict):
            continue

        txt = (str(label) + " " + json.dumps(obj, ensure_ascii=False)).lower()
        if "orange" not in txt and "cube" not in txt and "die" not in txt:
            continue

        xyz = _target_xyz(obj)
        if xyz is None:
            continue

        out = dict(obj)
        out["base_xyz_m"] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        out["center_base_xyz_m"] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        out["object_color"] = out.get("object_color") or "orange"
        out["object_shape"] = out.get("object_shape") or "cube_or_die"
        out["source"] = "optional_live_orange_obstacle_detector"

        return label, out, xyz

    return None, None, None


def _merge_optional_live_obstacles(data: dict, obj_key: str, ply: Path, scene_json: Path, start_ns: int):
    """
    Añade a la escena live los obstáculos detectados desde el mismo PLY.
    
    Se usa el detector disponible para el objeto naranja. Si no encuentra nada,
    dynamic_grasp continúa con obstacles=[].
    """
    proof = []

    proof.append("=== OPTIONAL LIVE OBSTACLE DETECTION ===")
    proof.append(f"selected_object: {obj_key}")
    proof.append(f"ply: {ply}")

    if obj_key == "orange_cube":
        proof.append("skip: selected object already orange_cube")
        proof.append("=== END OPTIONAL LIVE OBSTACLE DETECTION ===")
        return data, "\n".join(proof)

    detector = Path(__file__).resolve().parent / "vision_zonas.py"
    if not detector.exists():
        proof.append(f"skip: detector not found: {detector}")
        proof.append("=== END OPTIONAL LIVE OBSTACLE DETECTION ===")
        return data, "\n".join(proof)

    cmd = [sys.executable, str(detector), "--ply", str(ply)]
    proof.append("cmd: " + " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=240,
        )
    except Exception as e:
        proof.append(f"exception: {repr(e)}")
        proof.append("=== END OPTIONAL LIVE OBSTACLE DETECTION ===")
        data["live_obstacle_detection"] = {
            "ok": False,
            "reason": repr(e),
        }
        scene_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return data, "\n".join(proof)

    out = ""
    if proc.stdout:
        out += proc.stdout
    if proc.stderr:
        out += "\n--- STDERR ---\n" + proc.stderr

    proof.append(f"returncode: {proc.returncode}")
    proof.append("--- detector_output_tail ---")
    proof.append(out[-5000:])

    if proc.returncode != 0:
        data["live_obstacle_detection"] = {
            "ok": False,
            "reason": "orange detector returned nonzero",
            "returncode": proc.returncode,
        }
        scene_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        proof.append("status: no obstacle merged")
        proof.append("=== END OPTIONAL LIVE OBSTACLE DETECTION ===")
        return data, "\n".join(proof)

    orange_json = _find_recent_orange_scene_json(start_ns)
    if orange_json is None:
        data["live_obstacle_detection"] = {
            "ok": False,
            "reason": "no recent orange scene json found",
        }
        scene_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        proof.append("status: no recent orange json")
        proof.append("=== END OPTIONAL LIVE OBSTACLE DETECTION ===")
        return data, "\n".join(proof)

    try:
        orange_data = json.loads(orange_json.read_text(encoding="utf-8"))
    except Exception as e:
        data["live_obstacle_detection"] = {
            "ok": False,
            "reason": f"orange json read failed: {e}",
            "orange_json": str(orange_json),
        }
        scene_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        proof.append(f"status: orange json read failed {e}")
        proof.append("=== END OPTIONAL LIVE OBSTACLE DETECTION ===")
        return data, "\n".join(proof)

    label, orange_obj, orange_xyz = _extract_orange_obstacle(orange_data)
    if orange_obj is None:
        data["live_obstacle_detection"] = {
            "ok": False,
            "reason": "orange json has no usable orange obstacle",
            "orange_json": str(orange_json),
        }
        scene_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        proof.append("status: no usable orange obstacle")
        proof.append("=== END OPTIONAL LIVE OBSTACLE DETECTION ===")
        return data, "\n".join(proof)

    _, selected_xyz = _scene_xyz(data, obj_key)
    dxy = _dist_xy(orange_xyz, selected_xyz)

    if dxy < 0.012:
        data["live_obstacle_detection"] = {
            "ok": False,
            "reason": "orange obstacle too close to selected object, likely duplicate",
            "orange_json": str(orange_json),
            "orange_xyz_m": [float(x) for x in orange_xyz],
            "selected_xyz_m": [float(x) for x in selected_xyz],
            "dist_xy_m": float(dxy),
        }
        scene_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        proof.append(f"status: ignored duplicate, dist_xy={dxy:.4f}")
        proof.append("=== END OPTIONAL LIVE OBSTACLE DETECTION ===")
        return data, "\n".join(proof)

    objects = data.setdefault("objects", {})
    objects["orange_cube"] = orange_obj

    data["live_obstacle_detection"] = {
        "ok": True,
        "source": "vision_zonas.py",
        "orange_json": str(orange_json),
        "orange_source_label": label,
        "orange_xyz_m": [float(x) for x in orange_xyz],
        "selected_xyz_m": [float(x) for x in selected_xyz],
        "dist_xy_m": float(dxy),
    }

    scene_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    proof.append(f"status: merged orange obstacle from {label}")
    proof.append(f"orange_json: {orange_json}")
    proof.append(f"orange_xyz_m: {[float(x) for x in orange_xyz]}")
    proof.append(f"selected_xyz_m: {[float(x) for x in selected_xyz]}")
    proof.append(f"dist_xy_m: {float(dxy)}")
    proof.append("=== END OPTIONAL LIVE OBSTACLE DETECTION ===")

    print(
        "Obstáculo naranja añadido: "
        f"xyz={[float(x) for x in orange_xyz]} dist_xy={dxy:.3f}m",
        flush=True,
    )

    return data, "\n".join(proof)



def _ply_dtype_from_property(ptype):
    import numpy as np
    m = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    if ptype not in m:
        raise RuntimeError(f"Tipo PLY no soportado: {ptype}")
    return m[ptype]


def _read_ply_xyz_camera_mm(ply: Path):
    import numpy as np

    ply = Path(ply)

    with ply.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("PLY sin end_header")
            txt = line.decode("ascii", errors="replace").strip()
            header.append(txt)
            if txt == "end_header":
                break

        fmt = None
        vertex_count = None
        props = []
        in_vertex = False

        for line in header:
            parts = line.split()
            if not parts:
                continue

            if parts[0] == "format":
                fmt = parts[1]

            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])

            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise RuntimeError("PLY con property list en vertex no soportada")
                props.append((parts[1], parts[2]))

        if vertex_count is None:
            raise RuntimeError("No encontré element vertex en PLY")

        prop_names = [name for _, name in props]
        for name in ("x", "y", "z"):
            if name not in prop_names:
                raise RuntimeError(f"PLY sin propiedad {name}")

        ix, iy, iz = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")

        if fmt == "binary_little_endian":
            dtype = [(name, _ply_dtype_from_property(ptype)) for ptype, name in props]
            arr = np.fromfile(f, dtype=np.dtype(dtype), count=vertex_count)
            xyz = np.vstack([arr["x"], arr["y"], arr["z"]]).T.astype(float)
            return xyz

        if fmt == "ascii":
            arr = np.loadtxt(f, max_rows=vertex_count, usecols=(ix, iy, iz))
            return arr.astype(float)

        raise RuntimeError(f"Formato PLY no soportado: {fmt}")


def _camera_mm_to_base_m(xyz_camera_mm):
    import numpy as np

    # Misma matriz que aparece en los detectores Photoneo -> base.
    T_BASE_FROM_CAMERA_MM = np.array([
        [ 4.10000e-02, -9.98000e-01, -5.60000e-02, -2.19488e+02],
        [-9.99000e-01, -4.10000e-02,  0.00000e+00, -9.52756e+02],
        [-2.00000e-03,  5.60000e-02, -9.98000e-01,  8.66163e+02],
        [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00],
    ], dtype=float)

    xyz = np.asarray(xyz_camera_mm, dtype=float)
    ones = np.ones((xyz.shape[0], 1), dtype=float)
    homog = np.hstack([xyz, ones])
    base_mm = homog @ T_BASE_FROM_CAMERA_MM.T
    return base_mm[:, :3] / 1000.0


def _merge_local_ply_grasp_obstacles(data: dict, obj_key: str, ply: Path, scene_json: Path):
    """
    Detecta obstáculos geométricos locales alrededor del objeto seleccionado.
    
    El criterio no depende del nombre del objeto:
    - si se selecciona el cilindro, un dado cercano cuenta como obstáculo;
    - si se selecciona el dado, un cilindro cercano cuenta como obstáculo;
    - el color o nombre del objeto vecino no interviene.
    """
    import os
    import json
    import numpy as np

    proof = []
    proof.append("=== LOCAL PLY GRASP OBSTACLE DETECTION ===")
    proof.append(f"selected_object: {obj_key}")
    proof.append(f"ply: {ply}")

    try:
        _, selected_xyz = _scene_xyz(data, obj_key)
        selected_xyz = np.asarray(selected_xyz, dtype=float)
    except Exception as e:
        proof.append(f"status: failed selected xyz: {e}")
        proof.append("=== END LOCAL PLY GRASP OBSTACLE DETECTION ===")
        return data, "\n".join(proof)

    near_m = float(os.environ.get("VLA_LOCAL_OBSTACLE_NEAR_M", "0.110"))
    exclude_m = float(os.environ.get("VLA_LOCAL_OBSTACLE_EXCLUDE_M", "0.035"))
    min_points = int(os.environ.get("VLA_LOCAL_OBSTACLE_MIN_POINTS", "80"))

    z_min = float(os.environ.get(
        "VLA_LOCAL_OBSTACLE_Z_MIN_M",
        str(max(0.055, float(selected_xyz[2]) - 0.030))
    ))
    z_max = float(os.environ.get(
        "VLA_LOCAL_OBSTACLE_Z_MAX_M",
        str(float(selected_xyz[2]) + 0.090)
    ))

    try:
        xyz_cam = _read_ply_xyz_camera_mm(ply)
        xyz_base = _camera_mm_to_base_m(xyz_cam)
    except Exception as e:
        data["local_ply_grasp_obstacles"] = {
            "ok": False,
            "reason": f"ply read/transform failed: {e}",
        }
        scene_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        proof.append(f"status: failed ply processing: {e}")
        proof.append("=== END LOCAL PLY GRASP OBSTACLE DETECTION ===")
        return data, "\n".join(proof)

    dxy = np.linalg.norm(xyz_base[:, :2] - selected_xyz[:2], axis=1)

    mask = (
        (dxy > exclude_m)
        & (dxy < near_m)
        & (xyz_base[:, 2] >= z_min)
        & (xyz_base[:, 2] <= z_max)
    )

    pts = xyz_base[mask]
    dpts = dxy[mask]

    proof.append(f"selected_xyz_m: {[float(x) for x in selected_xyz]}")
    proof.append(f"near_m: {near_m}")
    proof.append(f"exclude_m: {exclude_m}")
    proof.append(f"z_min_m: {z_min}")
    proof.append(f"z_max_m: {z_max}")
    proof.append(f"candidate_points: {int(pts.shape[0])}")

    if pts.shape[0] < min_points:
        data["local_ply_grasp_obstacles"] = {
            "ok": False,
            "reason": "not_enough_local_points",
            "candidate_points": int(pts.shape[0]),
            "min_points": int(min_points),
            "selected_xyz_m": [float(x) for x in selected_xyz],
            "near_m": float(near_m),
            "exclude_m": float(exclude_m),
            "z_min_m": float(z_min),
            "z_max_m": float(z_max),
        }
        scene_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        proof.append("status: no local obstacle")
        proof.append("=== END LOCAL PLY GRASP OBSTACLE DETECTION ===")
        return data, "\n".join(proof)

    # Se usan los puntos más cercanos para evitar que una zona grande domine la mediana.
    order = np.argsort(dpts)
    pts_near = pts[order[: min(2500, len(order))]]

    center = np.median(pts_near, axis=0)
    dist_center = float(np.linalg.norm(center[:2] - selected_xyz[:2]))

    # Radio aproximado del blob local.
    r90 = float(np.percentile(np.linalg.norm(pts_near[:, :2] - center[:2], axis=1), 90))

    # Si la región local es enorme, normalmente no es un objeto discreto:
    # suele ser superficie, borde de zona, parte del propio objeto o ruido amplio.
    # No debe entrar como obstáculo de agarre.
    max_radius = float(os.environ.get("VLA_LOCAL_OBSTACLE_MAX_RADIUS_M", "0.040"))
    max_points = int(os.environ.get("VLA_LOCAL_OBSTACLE_MAX_POINTS", "8000"))

    if r90 >= max_radius or int(pts.shape[0]) > max_points:
        data["local_ply_grasp_obstacles"] = {
            "ok": False,
            "reason": "broad_local_region_ignored",
            "selected_object": obj_key,
            "selected_xyz_m": [float(x) for x in selected_xyz],
            "candidate_points": int(pts.shape[0]),
            "radius_m": float(r90),
            "max_radius_m": float(max_radius),
            "max_points": int(max_points),
            "near_m": float(near_m),
            "exclude_m": float(exclude_m),
            "z_min_m": float(z_min),
            "z_max_m": float(z_max),
        }
        scene_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        proof.append("status: broad local region ignored")
        proof.append(f"candidate_points: {int(pts.shape[0])}")
        proof.append(f"radius_m: {float(r90)}")
        proof.append("=== END LOCAL PLY GRASP OBSTACLE DETECTION ===")
        print(
            "Región local amplia ignorada como obstáculo: "
            f"radius={r90:.3f}m points={pts.shape[0]}",
            flush=True,
        )
        return data, "\n".join(proof)

    obs = {
        "base_xyz_m": [float(x) for x in center],
        "center_base_xyz_m": [float(x) for x in center],
        "safe_above_base_xyz_m": [float(center[0]), float(center[1]), float(max(center[2] + 0.18, 0.22))],
        "radius_m": float(max(0.018, min(0.050, r90))),
        "object_color": "unknown",
        "object_shape": "local_ply_obstacle",
        "semantic_override": "local_ply_obstacle",
        "source": "local_ply_geometric_annulus",
        "dist_xy_to_selected_m": float(dist_center),
        "num_points": int(pts.shape[0]),
    }

    objects = data.setdefault("objects", {})
    objects["local_ply_obstacle_0"] = obs

    data["local_ply_grasp_obstacles"] = {
        "ok": True,
        "selected_object": obj_key,
        "selected_xyz_m": [float(x) for x in selected_xyz],
        "obstacle_key": "local_ply_obstacle_0",
        "obstacle_xyz_m": obs["base_xyz_m"],
        "dist_xy_to_selected_m": float(dist_center),
        "candidate_points": int(pts.shape[0]),
        "radius_m": obs["radius_m"],
        "near_m": float(near_m),
        "exclude_m": float(exclude_m),
        "z_min_m": float(z_min),
        "z_max_m": float(z_max),
    }

    scene_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    proof.append("status: merged local_ply_obstacle_0")
    proof.append(f"obstacle_xyz_m: {obs['base_xyz_m']}")
    proof.append(f"dist_xy_to_selected_m: {dist_center}")
    proof.append(f"radius_m: {obs['radius_m']}")
    proof.append("=== END LOCAL PLY GRASP OBSTACLE DETECTION ===")

    print(
        "Obstáculo geométrico local añadido: "
        f"xyz={obs['base_xyz_m']} dist_xy={dist_center:.3f}m points={pts.shape[0]}",
        flush=True,
    )

    return data, "\n".join(proof)



def _normalize_selected_live_object(data: dict, obj_key: str):
    """
    Normaliza el objeto seleccionado dentro de data['objects'].
    
        Ejemplo:
        - obj_key='orange_cube'
        - detector devuelve data['orange_die']
        - se crea data['objects']['orange_cube'] con la misma posición live
    """
    objects = data.setdefault("objects", {})

    def xyz_of(obj):
        if not isinstance(obj, dict):
            return None
        for k in ("base_xyz_m", "center_base_xyz_m", "safe_above_base_xyz_m", "xyz_m"):
            v = obj.get(k)
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                try:
                    return [float(v[0]), float(v[1]), float(v[2])]
                except Exception:
                    pass
        return None

    if obj_key in ("orange_cube", "orange_die"):
        candidates = [
            ("objects.orange_cube", objects.get("orange_cube")),
            ("objects.orange_die", objects.get("orange_die")),
            ("orange_cube", data.get("orange_cube")),
            ("orange_die", data.get("orange_die")),
        ]
        canonical_key = "orange_cube"
        color = "orange"
        shape = "cube_or_die"

    elif obj_key == "white_cylinder":
        candidates = [
            ("objects.white_cylinder", objects.get("white_cylinder")),
            ("white_cylinder", data.get("white_cylinder")),
            ("cylinder", data.get("cylinder")),
        ]
        canonical_key = "white_cylinder"
        color = "white"
        shape = "cylinder"

    else:
        candidates = [
            (f"objects.{obj_key}", objects.get(obj_key)),
            (obj_key, data.get(obj_key)),
        ]
        canonical_key = obj_key
        color = None
        shape = None

    for label, obj in candidates:
        xyz = xyz_of(obj)
        if xyz is None:
            continue

        out = dict(obj)
        out["base_xyz_m"] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        out["center_base_xyz_m"] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]

        if color:
            out["object_color"] = out.get("object_color") or color
        if shape:
            out["object_shape"] = out.get("object_shape") or shape

        out["semantic_override"] = out.get("semantic_override") or canonical_key
        out["source"] = out.get("source") or f"normalized_selected_live_object:{label}"

        objects[canonical_key] = out
        data["selected_object"] = canonical_key

        # Se mantienen estos nombres por compatibilidad.
        if canonical_key == "orange_cube":
            data["orange_die"] = out
            data["orange_cube"] = out
        elif canonical_key == "white_cylinder":
            data["white_cylinder"] = out

        return data

    raise RuntimeError(f"No pude normalizar selected_object={obj_key}; no hay xyz válido")


def _run_live_ply_scene_detection(instruction: str):
    obj_key, target_key = choose_from_instruction(instruction)

    ply = Path("data/photoneo/live_scene.ply")
    if not ply.exists():
        raise RuntimeError(f"No existe PLY live para detección: {ply}")

    # La escena live debe ser multiobjeto para que place_occupancy vea
    # otros objetos ya colocados en la zona destino.
    detector = Path(__file__).resolve().parent / "vision_forma.py"
    if not detector.exists():
        raise RuntimeError(f"No existe detector live desde PLY: {detector}")

    start_ns = time.time_ns()

    cmd = [sys.executable, str(detector), "--ply", str(ply)]
    print("Ejecutando detección desde PLY nuevo:", " ".join(cmd), flush=True)

    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=240,
    )

    det_output = ""
    if proc.stdout:
        det_output += proc.stdout
    if proc.stderr:
        det_output += "\n--- STDERR ---\n" + proc.stderr

    print(det_output, flush=True)

    if proc.returncode != 0:
        raise RuntimeError(f"Detector live desde PLY falló con RC={proc.returncode}")

    scene_json = _find_recent_live_scene_json(obj_key, start_ns)
    data = json.loads(scene_json.read_text(encoding="utf-8"))
    data = _normalize_selected_live_object(data, obj_key)
    scene_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    data = _ensure_live_target_alias(data, target_key, scene_json)
    data, obstacle_proof = _merge_optional_live_obstacles(data, obj_key, ply, scene_json, start_ns)
    data, local_obstacle_proof = _merge_local_ply_grasp_obstacles(data, obj_key, ply, scene_json)
    obstacle_proof = obstacle_proof + "\n\n" + local_obstacle_proof
    xyz_label, xyz = _scene_xyz(data, obj_key)

    os.environ["VLA_SELECTED_TARGET"] = target_key
    os.environ["VLA_TARGET"] = target_key
    os.environ["VLA_FORCE_LIVE_SCENE_JSON"] = str(scene_json)
    os.environ["DETECTED_SCENE_JSON"] = str(scene_json)

    proof = []
    proof.append("")
    proof.append("=== LIVE PLY SCENE DETECTION PROOF ===")
    proof.append(f"instruction: {instruction}")
    proof.append(f"selected_object: {obj_key}")
    proof.append(f"selected_target: {target_key}")
    proof.append(f"ply: {ply}")
    proof.append(f"detector: {detector}")
    proof.append(f"scene_json: {scene_json}")
    proof.append(f"xyz_source: {xyz_label}")
    proof.append(f"selected_object_base_xyz_m: {xyz}")
    proof.append("")
    proof.append("--- detector_output_tail ---")
    proof.append(det_output[-5000:])
    proof.append("")
    proof.append(obstacle_proof[-7000:])
    proof.append("=== END LIVE PLY SCENE DETECTION PROOF ===")
    proof.append("")

    print(f"Escena live forzada: {scene_json}", flush=True)
    print(f"{obj_key} base_xyz_m={xyz}", flush=True)

    return "\\n".join(proof)



def _request_photoneo_live_capture(instruction: str = ""):
    import subprocess

    enabled = os.environ.get("VLA_PHOTONEO_LIVE", "1").strip().lower()
    if enabled in ("0", "false", "no"):
        raise RuntimeError("VLA_PHOTONEO_LIVE=0 no permitido en flujo real/GUI con Photoneo obligatoria")

    client = Path(__file__).resolve().parent / "photoneo" / "captura.py"
    if not client.exists():
        raise RuntimeError(f"No existe cliente Photoneo live: {client}")

    print("Solicitando una captura nueva antes de planificar...", flush=True)

    proc = subprocess.run(
        [sys.executable, str(client)],
        text=True,
        capture_output=True,
        timeout=180,
    )

    client_output = ""
    if proc.stdout:
        client_output += proc.stdout
    if proc.stderr:
        client_output += "\n--- STDERR ---\n" + proc.stderr

    print(client_output, flush=True)

    if proc.returncode != 0:
        raise RuntimeError(f"Cliente Photoneo falló con RC={proc.returncode}")

    meta = _load_photoneo_meta()
    _validate_photoneo_mandatory_capture(meta, client_output)

    scene_proof = _run_live_ply_scene_detection(instruction)
    proof = _build_photoneo_proof_text(meta, client_output) + scene_proof

    print("Captura nueva validada con TriggerFrame Software.", flush=True)
    return proof


def _append_photoneo_proof_to_log(model: str, proof: str):
    logs = sorted(
        LOG_ROOT.glob(f"run_*/{model}.log"),
        key=lambda q: q.stat().st_mtime,
        reverse=True,
    )
    if not logs:
        raise RuntimeError(f"No encontré log de demo para modelo {model}; no puedo guardar prueba Photoneo")

    log_path = logs[0]
    with log_path.open("a", encoding="utf-8") as f:
        f.write(proof)
        f.flush()

    print(f"Prueba de captura obligatoria añadida al log: {log_path}", flush=True)


if "run_executor" in globals():
    _run_executor_base = run_executor

    def run_executor(model: str, instruction: str, real: bool):
        source = os.environ.get(
            "VLA_SOURCE",
            "live",
        ).strip().lower()

        if source == "replay":
            print(
                "Replay seleccionado: "
                "no se solicita captura live.",
                flush=True,
            )

            return _run_executor_base(
                model,
                instruction,
                real,
            )

        photoneo_proof = (
            _request_photoneo_live_capture(
                str(instruction)
            )
        )

        try:
            return _run_executor_base(
                model,
                instruction,
                real,
            )
        finally:
            _append_photoneo_proof_to_log(
                str(model),
                photoneo_proof,
            )

    print("Captura nueva activa: se validará TriggerFrame y se guardará constancia en el registro.")
else:
    print("No existe run_executor; no se pudo activar la captura nueva.")

if __name__ == "__main__":
    raise SystemExit(main())
