#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import rclpy


ROOT = Path("/workspace")
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results" / "pi0"

FLOW_PATH = SCRIPTS / "flujo.py"
ACTION_CONTROLLER_PATH = SCRIPTS / "control_acciones.py"
EXECUTION_PATH = (
    SCRIPTS
    / "ejecucion.py"
)
ROBOT_BASE_PATH = (
    SCRIPTS / "robot_base.py"
)

MODEL_TO_BASE = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=float,
)

TARGET_AXIS = np.array([0.0, 0.0, -1.0], dtype=float)

DEFAULT_GLOBAL_HOME_Q = np.array(
    [
        1.4202332496643066,
        -1.3768111777356644,
        1.5765402952777308,
        -1.747592111627096,
        -1.5658510367022913,
        -0.10327083269228154,
    ],
    dtype=float,
)


class EpisodeStop(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    instruction: str
    mode: str
    open_width_m: float
    close_width_m: float
    pick_z_m: float
    pre_pick_z_m: float
    transit_z_m: float
    place_z_m: float
    chunk_steps: int
    max_cycles_approach: int
    max_cycles_transport: int
    approach_goal_tolerance_m: float
    transport_goal_tolerance_m: float
    maximum_step_distance_increase_m: float
    minimum_chunk_progress_m: float
    max_chunk_joint_step_rad: float
    max_contact_joint_step_rad: float
    max_fk_error_m: float
    max_tilt_deg: float
    pick_actual_error_m: float
    place_actual_error_m: float
    waypoint_duration_s: float
    gripper_hold_s: float


def load_module(name: str, path: Path):
    if not path.is_file():
        raise EpisodeStop(f"No existe módulo requerido: {path}")

    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise EpisodeStop(f"No pude cargar módulo: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


flow = load_module("flujo", FLOW_PATH)
action_controller = load_module("control_acciones", ACTION_CONTROLLER_PATH)
robot_base = load_module("robot_base", ROBOT_BASE_PATH)
execution = load_module("ejecucion", EXECUTION_PATH)


for _alias in (
    "orange block",
    "orange die",
    "orange object",
    "bloque naranja",
    "dado naranja",
    "objeto naranja",
):
    if _alias not in flow.OBJECT_ALIASES["orange_cube"]:
        flow.OBJECT_ALIASES["orange_cube"].append(_alias)

for _alias in (
    "white cylindrical object",
    "white can-shaped object",
    "white can shaped object",
    "white object",
    "objeto cilíndrico blanco",
    "objeto cilindrico blanco",
    "objeto blanco",
):
    if _alias not in flow.OBJECT_ALIASES["white_cylinder"]:
        flow.OBJECT_ALIASES["white_cylinder"].append(_alias)

for _alias in (
    "yellow area",
    "yellow region",
    "área amarilla",
    "area amarilla",
    "región amarilla",
    "region amarilla",
):
    if _alias not in flow.TARGET_ALIASES["yellow_zone"]:
        flow.TARGET_ALIASES["yellow_zone"].append(_alias)

for _alias in (
    "blue area",
    "blue region",
    "área azul",
    "area azul",
    "región azul",
    "region azul",
):
    if _alias not in flow.TARGET_ALIASES["blue_zone"]:
        flow.TARGET_ALIASES["blue_zone"].append(_alias)


_PI0_BUILD_STATE_BASE = (
    action_controller.build_pi0_virtual_state
)

_PI0_REAL_STATE_ANCHOR: dict[str, np.ndarray] | None = None


def _snapshot_vector(
    snapshot: dict[str, Any],
    keys: tuple[str, ...],
    size: int,
    label: str,
) -> np.ndarray:
    for key in keys:
        value = snapshot.get(key)

        if value is None:
            continue

        vector = np.asarray(
            value,
            dtype=float,
        ).reshape(-1)

        if (
            vector.shape == (size,)
            and np.all(np.isfinite(vector))
        ):
            return vector

    raise EpisodeStop(
        f"{label}: no encuentro vector válido "
        f"en {keys}"
    )


def _build_pi0_virtual_state_live(
    real_snapshot: dict[str, Any],
    gripper_open_width_m: float,
    gripper_close_width_m: float,
) -> dict[str, Any]:
    global _PI0_REAL_STATE_ANCHOR

    document = dict(
        _PI0_BUILD_STATE_BASE(
            real_snapshot=real_snapshot,
            gripper_open_width_m=(
                gripper_open_width_m
            ),
            gripper_close_width_m=(
                gripper_close_width_m
            ),
        )
    )

    position_base = _snapshot_vector(
        real_snapshot,
        (
            "tcp_position_base_m",
            "tcp_position_m",
        ),
        3,
        "pi0_state.position",
    )

    quaternion_base = _snapshot_vector(
        real_snapshot,
        (
            "tcp_quaternion_xyzw",
            "tcp_orientation_xyzw",
        ),
        4,
        "pi0_state.quaternion",
    )

    quaternion_base = (
        action_controller.normalize_quaternion_xyzw(
            quaternion_base
        )
    )

    if _PI0_REAL_STATE_ANCHOR is None:
        _PI0_REAL_STATE_ANCHOR = {
            "position_base_m": (
                position_base.copy()
            ),
            "quaternion_base_xyzw": (
                quaternion_base.copy()
            ),
        }

    anchor_position = (
        _PI0_REAL_STATE_ANCHOR[
            "position_base_m"
        ]
    )

    anchor_quaternion = (
        _PI0_REAL_STATE_ANCHOR[
            "quaternion_base_xyzw"
        ]
    )

    translation_delta_base = (
        position_base - anchor_position
    )

    translation_delta_model = (
        MODEL_TO_BASE.T
        @ translation_delta_base
    )

    relative_quaternion_base = (
        action_controller.quaternion_multiply_xyzw(
            quaternion_base,
            action_controller.quaternion_conjugate_xyzw(
                anchor_quaternion
            ),
        )
    )

    rotation_delta_base = (
        action_controller.quaternion_to_rotvec_xyzw(
            relative_quaternion_base
        )
    )

    rotation_delta_model = (
        MODEL_TO_BASE.T
        @ rotation_delta_base
    )

    if (
        float(
            np.linalg.norm(
                translation_delta_base
            )
        )
        > 0.800
    ):
        raise EpisodeStop(
            "Desplazamiento TCP imposible "
            "respecto al ancla π0."
        )

    if (
        float(
            np.linalg.norm(
                rotation_delta_base
            )
        )
        > math.pi
    ):
        raise EpisodeStop(
            "Rotación TCP imposible "
            "respecto al ancla π0."
        )

    state = np.asarray(
        action_controller.LIBERO_STATE_MEAN,
        dtype=float,
    ).copy()

    if state.shape != (8,):
        raise EpisodeStop(
            f"LIBERO_STATE_MEAN shape={state.shape}"
        )

    state[:3] = (
        state[:3]
        + translation_delta_model
    )

    state[3:6] = (
        state[3:6]
        + rotation_delta_model
    )

    open_width = float(
        gripper_open_width_m
    )

    close_width = float(
        gripper_close_width_m
    )

    commanded_width = (
        _PI0_COMMANDED_GRIPPER_WIDTH_M
    )

    if commanded_width is None:
        commanded_width = float(
            real_snapshot.get(
                "gripper_width_m",
                open_width,
            )
        )

    denominator = max(
        1e-9,
        open_width - close_width,
    )

    open_fraction = float(
        np.clip(
            (
                float(commanded_width)
                - close_width
            )
            / denominator,
            0.0,
            1.0,
        )
    )

    finger_closed = np.asarray(
        action_controller.LIBERO_FINGER_CLOSED,
        dtype=float,
    ).reshape(2)

    finger_open = np.asarray(
        action_controller.LIBERO_FINGER_OPEN,
        dtype=float,
    ).reshape(2)

    state[6:8] = (
        finger_closed
        + open_fraction
        * (finger_open - finger_closed)
    )

    document["state_8d"] = state.tolist()

    document["state_source"] = (
        "real_robot_persistent_anchor_"
        "mapped_to_libero_reference"
    )

    document["live_state_mapping"] = {
        "anchor_position_base_m": (
            anchor_position.tolist()
        ),
        "current_position_base_m": (
            position_base.tolist()
        ),
        "translation_delta_base_m": (
            translation_delta_base.tolist()
        ),
        "translation_delta_model_m": (
            translation_delta_model.tolist()
        ),
        "rotation_delta_base_rotvec_rad": (
            rotation_delta_base.tolist()
        ),
        "rotation_delta_model_rotvec_rad": (
            rotation_delta_model.tolist()
        ),
        "commanded_gripper_width_m": float(
            commanded_width
        ),
        "gripper_open_fraction": (
            open_fraction
        ),
        "first_frame_t0": bool(
            np.linalg.norm(
                translation_delta_base
            )
            <= 1e-9
        ),
        "real_pose_delta_applied": True,
        "persistent_anchor_within_episode": True,
    }

    print(
        f"delta_base="
        f"{translation_delta_base.tolist()} "
        f"state_xyz={state[:3].tolist()}",
        flush=True,
    )

    return document


action_controller.build_pi0_virtual_state = (
    _build_pi0_virtual_state_live
)
full7d = load_module(
    "control_pi0",
    SCRIPTS / "control_pi0.py",
)


def finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)

    if vector.size != size:
        raise EpisodeStop(
            f"{label}: tamaño {vector.size}; esperado {size}"
        )

    if not np.all(np.isfinite(vector)):
        raise EpisodeStop(f"{label}: contiene NaN o Inf")

    return vector


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def xyz_from_record(record: Any, label: str) -> np.ndarray:
    if not isinstance(record, dict):
        raise EpisodeStop(f"{label}: no es un objeto JSON")

    for key in (
        "base_xyz_m",
        "center_base_xyz_m",
        "safe_above_base_xyz_m",
        "xyz_m",
    ):
        value = record.get(key)

        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return finite_vector(value[:3], 3, f"{label}.{key}")

    raise EpisodeStop(f"{label}: no contiene coordenadas base válidas")


def safe_xyz_from_record(
    record: Any,
    default_z_m: float,
    label: str,
) -> np.ndarray:
    if isinstance(record, dict):
        value = record.get("safe_above_base_xyz_m")

        if isinstance(value, (list, tuple)) and len(value) >= 3:
            result = finite_vector(
                value[:3],
                3,
                f"{label}.safe_above_base_xyz_m",
            )
            result[2] = max(float(result[2]), float(default_z_m))
            return result

    result = xyz_from_record(record, label)
    result[2] = float(default_z_m)
    return result


def resolve_entities(
    scene: dict[str, Any],
    config: Config,
) -> dict[str, Any]:
    objects = scene.get("objects")
    targets = scene.get("targets")

    if not isinstance(objects, dict):
        raise EpisodeStop("La escena live no contiene objects")

    if not isinstance(targets, dict):
        raise EpisodeStop("La escena live no contiene targets")

    try:
        selected_key, selected_object = action_controller.resolve_instruction_object(
            scene,
            config.instruction,
        )
    except Exception as semantic_error:
        try:
            selected_key, _ = flow.choose_from_instruction(
                config.instruction
            )
        except Exception as fallback_error:
            raise EpisodeStop(
                "La instrucción no identifica un objeto soportado: "
                f"primary={semantic_error}; fallback={fallback_error}"
            ) from fallback_error

        selected_object = objects.get(
            selected_key
        )

        if not isinstance(
            selected_object,
            dict,
        ):
            raise EpisodeStop(
                f"El objeto {selected_key!r} no está visible"
            )

    # La instrucción fija un destino canónico.
    # Desde ahí se consulta solo esa
    # entidad de la escena. blue_source puede seguir
    # presente como elemento geométrico, pero no es
    # un segundo destino si ya existe blue_zone.
    try:
        (
            _,
            requested_target_key,
        ) = flow.choose_from_instruction(
            config.instruction
        )
    except Exception as error:
        raise EpisodeStop(
            "No se pudo resolver el destino "
            "canónico de la instrucción: "
            f"{error}"
        ) from error

    supported_target_aliases = {
        "yellow_zone": (
            "yellow_zone",
            "yellow_target",
        ),
        "blue_zone": (
            "blue_zone",
        ),

        "square_box": (
            "square_box",
        ),

        "circular_box": (
            "circular_box",
        ),
    }

    if (
        requested_target_key
        not in supported_target_aliases
    ):
        raise EpisodeStop(
            "Destino canónico no soportado: "
            f"{requested_target_key!r}"
        )

    exact_target = targets.get(
        requested_target_key
    )

    if isinstance(
        exact_target,
        dict,
    ):
        target_key = (
            requested_target_key
        )
        target = exact_target
    else:
        target_candidates = []

        for alias in (
            supported_target_aliases[
                requested_target_key
            ]
        ):
            value = targets.get(alias)

            if isinstance(value, dict):
                target_candidates.append(
                    (
                        alias,
                        value,
                    )
                )

        if not target_candidates:
            raise EpisodeStop(
                "La escena live no contiene "
                "el destino solicitado: "
                f"requested="
                f"{requested_target_key}, "
                f"targets="
                f"{sorted(targets.keys())}"
            )

        resolved_candidates = [
            (
                key,
                value,
                xyz_from_record(
                    value,
                    f"targets.{key}",
                ),
            )
            for key, value
            in target_candidates
        ]

        reference_xyz = (
            resolved_candidates[0][2]
        )

        for key, _, xyz in (
            resolved_candidates[1:]
        ):
            distance = float(
                np.linalg.norm(
                    xyz - reference_xyz
                )
            )

            if distance > 0.003:
                raise EpisodeStop(
                    "Aliases incompatibles para "
                    f"{requested_target_key}: "
                    f"{resolved_candidates[0][0]}="
                    f"{reference_xyz.tolist()}, "
                    f"{key}={xyz.tolist()}, "
                    f"distancia={distance:.4f}m"
                )

        resolved_candidates.sort(
            key=lambda item: (
                0
                if item[0]
                == requested_target_key
                else 1,
                item[0],
            )
        )

        target_key = (
            resolved_candidates[0][0]
        )
        target = (
            resolved_candidates[0][1]
        )

    print(
        f"requested={requested_target_key} "
        f"resolved={target_key} "
        "blue_source_considered_as_"
        "independent_destination=false",
        flush=True,
    )

    object_xyz = xyz_from_record(
        selected_object,
        f"objects.{selected_key}",
    )

    object_safe_xyz = safe_xyz_from_record(
        selected_object,
        max(config.pre_pick_z_m, 0.20),
        f"objects.{selected_key}",
    )

    target_xyz = xyz_from_record(
        target,
        f"targets.{target_key}",
    )

    # El destino es una zona, no un punto cartesiano obligatorio.
    # El centro detectado queda como referencia, pero el
    # ejecutor selecciona un punto interior desplazado hacia la base.
    target_detected_center_xyz = (
        target_xyz.copy()
    )

    target_interior_inset_m = float(
        os.environ.get(
            "VLA_TARGET_INTERIOR_INSET_M",
            "0.070",
        )
    )

    if (
        not math.isfinite(
            target_interior_inset_m
        )
        or target_interior_inset_m < 0.0
        or target_interior_inset_m > 0.075
    ):
        raise EpisodeStop(
            "VLA_TARGET_INTERIOR_INSET_M debe estar "
            "entre 0.000 y 0.075 m"
        )

    target_radius_xy_m = float(
        np.linalg.norm(
            target_detected_center_xyz[:2]
        )
    )

    if (
        target_radius_xy_m
        <= target_interior_inset_m + 1e-6
    ):
        raise EpisodeStop(
            "No se puede calcular un punto interior "
            "del destino hacia la base."
        )

    target_xyz = (
        target_detected_center_xyz.copy()
    )

    if target_interior_inset_m > 0.0:
        radial_direction_xy = (
            target_detected_center_xyz[:2]
            / target_radius_xy_m
        )

        target_xyz[:2] = (
            target_detected_center_xyz[:2]
            - target_interior_inset_m
            * radial_direction_xy
        )

    selected_offset_xy_m = float(
        np.linalg.norm(
            target_xyz[:2]
            - target_detected_center_xyz[:2]
        )
    )

    if isinstance(target, dict):
        target[
            "detected_center_base_xyz_m"
        ] = target_detected_center_xyz.tolist()

        target[
            "selected_reachable_base_xyz_m"
        ] = target_xyz.tolist()

        target[
            "selected_reachable_offset_xy_m"
        ] = selected_offset_xy_m

        target[
            "selected_reachable_policy"
        ] = (
            "interior_point_toward_robot_base"
        )

    print(
        f"detected_center="
        f"{target_detected_center_xyz.tolist()} "
        f"selected_interior="
        f"{target_xyz.tolist()} "
        f"offset_xy="
        f"{selected_offset_xy_m:.4f}m",
        flush=True,
    )

    target_safe_xyz = target_xyz.copy()
    target_safe_xyz[2] = config.transit_z_m

    pick_xyz = object_xyz.copy()
    pick_xyz[2] = config.pick_z_m

    pre_pick_xyz = object_xyz.copy()
    pre_pick_xyz[2] = config.pre_pick_z_m

    # El lift de contacto es local.
    # El acceso al plano global se calcula después
    # mediante plan_transport_handoff().
    lift_xyz = object_xyz.copy()
    lift_xyz[2] = max(
        float(object_safe_xyz[2]),
        float(pre_pick_xyz[2]),
    )

    place_xyz = target_xyz.copy()
    place_xyz[2] = config.place_z_m

    pre_place_xyz = target_safe_xyz.copy()
    retreat_xyz = target_safe_xyz.copy()


    def entity_radius_m(
        record: dict[str, Any],
        default: float,
    ) -> float:
        for key in (
            "radius_m",
            "radius_p90_m",
            "radius_median_m",
        ):
            value = record.get(key)

            try:
                radius = float(value)
            except (TypeError, ValueError):
                continue

            if math.isfinite(radius) and radius > 0.0:
                return radius

        return float(default)

    selected_object_radius_m = entity_radius_m(
        selected_object,
        0.030,
    )

    place_obstacles: list[
        dict[str, Any]
    ] = []

    obstacle_dedup_xy_m = float(
        os.environ.get(
            "VLA_PLACE_OBSTACLE_DEDUP_XY_M",
            "0.025",
        )
    )

    selected_exclusion_xy_m = float(
        os.environ.get(
            "VLA_PLACE_SELECTED_OBJECT_DEDUP_XY_M",
            "0.012",
        )
    )

    for (
        obstacle_key,
        obstacle_record,
    ) in objects.items():
        if obstacle_key == selected_key:
            continue

        if not isinstance(
            obstacle_record,
            dict,
        ):
            continue

        try:
            obstacle_xyz = xyz_from_record(
                obstacle_record,
                f"objects.{obstacle_key}",
            )
        except Exception:
            continue

        distance_to_selected = float(
            np.linalg.norm(
                obstacle_xyz[:2]
                - object_xyz[:2]
            )
        )

        if (
            distance_to_selected
            < selected_exclusion_xy_m
        ):
            continue

        duplicate = False

        for previous in place_obstacles:
            previous_xy = np.asarray(
                previous["center_xyz_m"][:2],
                dtype=float,
            )

            if (
                float(
                    np.linalg.norm(
                        obstacle_xyz[:2]
                        - previous_xy
                    )
                )
                < obstacle_dedup_xy_m
            ):
                duplicate = True
                break

        if duplicate:
            continue

        place_obstacles.append(
            {
                "key": str(obstacle_key),
                "center_xyz_m":
                    obstacle_xyz.tolist(),
                "radius_m":
                    entity_radius_m(
                        obstacle_record,
                        0.030,
                    ),
            }
        )

    return {
        "selected_object": selected_key,
        "selected_target": target_key,
        "selected_object_radius_m":
            selected_object_radius_m,
        "place_obstacles":
            place_obstacles,
        "target_detected_center_xyz_m":
            target_detected_center_xyz,
        "object_xyz_m": object_xyz,
        "object_safe_xyz_m": object_safe_xyz,
        "pick_xyz_m": pick_xyz,
        "pre_pick_xyz_m": pre_pick_xyz,
        "lift_xyz_m": lift_xyz,
        "target_xyz_m": target_xyz,
        "target_safe_xyz_m": target_safe_xyz,
        "pre_place_xyz_m": pre_place_xyz,
        "place_xyz_m": place_xyz,
        "retreat_xyz_m": retreat_xyz,
    }



def workspace_ok(point: np.ndarray, contact: bool = False) -> bool:
    point = finite_vector(point, 3, "workspace.point")
    z_min = 0.030 if contact else 0.045

    return bool(
        -0.90 <= point[0] <= 0.90
        and -1.10 <= point[1] <= 0.20
        and z_min <= point[2] <= 0.45
    )


def make_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS / f"{stamp}_pi0"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _request_fresh_scene_live(
    run_dir: Path,
    instruction: str,
    index: int,
    label: str,
) -> tuple[Path, dict[str, Any], Path, str]:
    capture_started_ns = time.time_ns()

    proof = flow._request_photoneo_live_capture(instruction)

    if not isinstance(proof, str) or not proof.strip():
        raise EpisodeStop("Captura Photoneo sin prueba documental")

    if (
        __import__("os").environ.get(
            "VLA_TASK_FAMILY", ""
        ).strip() == "shape_classification"
    ):
        import json as _shape_json
        import os as _shape_os
        from pathlib import Path as _ShapePath

        _shape_raw = (
            _shape_os.environ.get(
                "VLA_FORCE_LIVE_SCENE_JSON", ""
            ).strip()
            or _shape_os.environ.get(
                "DETECTED_SCENE_JSON", ""
            ).strip()
        )

        if not _shape_raw:
            _shape_pointer = _ShapePath(
                "results/validacion_robot/"
                "ultima_escena.txt"
            )

            if _shape_pointer.is_file():
                _shape_raw = _shape_pointer.read_text(
                    encoding="utf-8"
                ).strip()

        if not _shape_raw:
            raise EpisodeStop(
                "Shape detectado, pero no existe ruta al JSON "
                "de la captura actual"
            )

        scene_path = _ShapePath(
            _shape_raw
        ).expanduser()

        if not scene_path.is_absolute():
            scene_path = (
                _ShapePath("/workspace") / scene_path
            ).resolve()

        if not scene_path.is_file():
            raise EpisodeStop(
                f"JSON shape actual inexistente: {scene_path}"
            )

        scene = _shape_json.loads(
            scene_path.read_text(encoding="utf-8")
        )

        _shape_objects = scene.get("objects") or {}
        _shape_targets = scene.get("targets") or {}

        _shape_missing = []

        for _shape_name in (
            "orange_cube",
            "white_cylinder",
        ):
            if _shape_name not in _shape_objects:
                _shape_missing.append(_shape_name)

        for _shape_name in (
            "square_box",
            "circular_box",
        ):
            if _shape_name not in _shape_targets:
                _shape_missing.append(_shape_name)

        if _shape_missing:
            raise EpisodeStop(
                "JSON shape incompleto: "
                f"missing={_shape_missing} "
                f"path={scene_path}"
            )

        image_path = (
            scene_path.parent / "photoneo_rgb.png"
        )

        if not image_path.is_file():
            raise EpisodeStop(
                "La captura shape actual no contiene "
                f"photoneo_rgb.png: {image_path}"
            )

        _shape_os.environ[
            "VLA_FORCE_LIVE_SCENE_JSON"
        ] = str(scene_path)

        _shape_os.environ[
            "DETECTED_SCENE_JSON"
        ] = str(scene_path)

        _shape_pointer = _ShapePath(
            "results/validacion_robot/"
            "ultima_escena.txt"
        )

        _shape_pointer.write_text(
            str(scene_path) + "\n",
            encoding="utf-8",
        )

        print(
            f"scene={scene_path} "
            f"image={image_path}",
            flush=True,
        )

    else:
        scene_path, scene = action_controller.load_canonical_scene()
        image_path = action_controller.resolve_clean_image(scene)

    if scene_path.stat().st_mtime_ns < capture_started_ns - 2_000_000_000:
        raise EpisodeStop(
            "La escena canónica no pertenece a la captura actual: "
            f"{scene_path}"
        )

    if image_path.stat().st_mtime_ns < capture_started_ns - 2_000_000_000:
        raise EpisodeStop(
            "La RGB no pertenece a la captura actual: "
            f"{image_path}"
        )

    prefix = f"{index:03d}_{label}"

    (run_dir / f"{prefix}_capture_proof.txt").write_text(
        proof,
        encoding="utf-8",
    )

    json_write(
        run_dir / f"{prefix}_capture.json",
        {
            "scene_path": str(scene_path),
            "scene_sha256": sha256_file(scene_path),
            "scene_mtime_ns": scene_path.stat().st_mtime_ns,
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "image_mtime_ns": image_path.stat().st_mtime_ns,
            "capture_started_ns": capture_started_ns,
        },
    )

    return scene_path, scene, image_path, proof


_PI0_INITIAL_VISUAL_CONTEXT: dict[str, Any] | None = None
_PI0_VISUAL_REUSE_LABELS: list[str] = []


def request_fresh_scene(
    run_dir: Path,
    instruction: str,
    index: int,
    label: str,
):
    global _PI0_INITIAL_VISUAL_CONTEXT

    normalized_label = str(label)

    if _PI0_INITIAL_VISUAL_CONTEXT is None:
        (
            scene_path,
            scene,
            image_path,
            proof,
        ) = _request_fresh_scene_live(
            run_dir=run_dir,
            instruction=instruction,
            index=index,
            label=label,
        )

        scene_path = Path(scene_path)
        image_path = Path(image_path)

        if not scene_path.is_file():
            raise EpisodeStop(
                f"No existe escena inicial: {scene_path}"
            )

        if not image_path.is_file():
            raise EpisodeStop(
                f"No existe imagen inicial: {image_path}"
            )

        _PI0_INITIAL_VISUAL_CONTEXT = {
            "scene_path": str(scene_path),
            "scene": copy.deepcopy(scene),
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "initial_label": normalized_label,
        }

        json_write(
            run_dir / "initial_visual_context.json",
            {
                **_PI0_INITIAL_VISUAL_CONTEXT,
                "scene": scene,
                "visual_reuse_labels": [],
                "capture_policy": (
                    "single_initial_unoccluded_capture"
                ),
            },
        )

        print(
            f"label={normalized_label} "
            f"image={image_path} "
            f"sha256={_PI0_INITIAL_VISUAL_CONTEXT['image_sha256']}",
            flush=True,
        )

        return (
            scene_path,
            copy.deepcopy(scene),
            image_path,
            proof,
        )

    # La comprobación posterior sí se hace con una imagen nueva,
    # después de abrir, retirarse y dejar libre la cámara.
    if normalized_label == "place_verification":
        print(
            "Photoneo vuelve a capturar después de la retirada.",
            flush=True,
        )

        return _request_fresh_scene_live(
            run_dir=run_dir,
            instruction=instruction,
            index=index,
            label=label,
        )

    scene_path = Path(
        _PI0_INITIAL_VISUAL_CONTEXT["scene_path"]
    )

    image_path = Path(
        _PI0_INITIAL_VISUAL_CONTEXT["image_path"]
    )

    if not scene_path.is_file():
        raise EpisodeStop(
            f"Desapareció escena inicial: {scene_path}"
        )

    if not image_path.is_file():
        raise EpisodeStop(
            f"Desapareció imagen inicial: {image_path}"
        )

    current_sha = sha256_file(image_path)
    expected_sha = _PI0_INITIAL_VISUAL_CONTEXT[
        "image_sha256"
    ]

    if current_sha != expected_sha:
        raise EpisodeStop(
            "La imagen inicial cambió durante el episodio"
        )

    _PI0_VISUAL_REUSE_LABELS.append(
        normalized_label
    )

    json_write(
        run_dir / "initial_visual_context.json",
        {
            **_PI0_INITIAL_VISUAL_CONTEXT,
            "scene": (
                _PI0_INITIAL_VISUAL_CONTEXT[
                    "scene"
                ]
            ),
            "visual_reuse_labels": list(
                _PI0_VISUAL_REUSE_LABELS
            ),
            "capture_policy": (
                "single_initial_unoccluded_capture"
            ),
        },
    )

    print(
        f"label={normalized_label} "
        f"image_sha256={expected_sha} "
        "photoneo_capture=false",
        flush=True,
    )

    proof = (
        "Visual context reused from initial "
        f"unoccluded capture; label={normalized_label}; "
        f"image_sha256={expected_sha}"
    )

    return (
        scene_path,
        copy.deepcopy(
            _PI0_INITIAL_VISUAL_CONTEXT[
                "scene"
            ]
        ),
        image_path,
        proof,
    )



def _infer_fresh_pi0_uncached(
    run_dir: Path,
    config: Config,
    index: int,
    label: str,
    commanded_gripper_width_m: float,
) -> dict[str, Any]:
    (
        scene_path,
        scene,
        image_path,
        proof,
    ) = request_fresh_scene(
        run_dir=run_dir,
        instruction=config.instruction,
        index=index,
        label=label,
    )

    measured_snapshot = action_controller.read_real_snapshot()
    model_snapshot = dict(measured_snapshot)

    measured_gripper_width = float(
        measured_snapshot["gripper_width_m"]
    )

    # El driver no refleja con fiabilidad finger_width después de mover
    # físicamente la 3FG15. Para el estado del modelo se usa el último ancho
    # enviado y la lectura de ROS se guarda por separado.
    model_snapshot["gripper_width_m"] = float(
        commanded_gripper_width_m
    )

    state_document = action_controller.build_pi0_virtual_state(
        real_snapshot=model_snapshot,
        gripper_open_width_m=config.open_width_m,
        gripper_close_width_m=config.close_width_m,
    )

    prefix = f"{index:03d}_{label}"
    state_path = run_dir / f"{prefix}_pi0_state.json"
    json_write(state_path, state_document)

    inference_started_ns = time.time_ns()

    pi0_document = action_controller.execute_pi0_inference(
        instruction=config.instruction,
        image_path=image_path,
        state_path=state_path,
    )

    raw_actions = np.asarray(
        pi0_document["output"]["actions"],
        dtype=float,
    )

    if raw_actions.shape != (10, 7):
        raise EpisodeStop(
            f"π0.5-LIBERO devolvió {raw_actions.shape}; "
            "esperado (10, 7)"
        )

    if not np.all(np.isfinite(raw_actions)):
        raise EpisodeStop("π0.5-LIBERO devolvió NaN o Inf")

    raw_source = Path(action_controller.PI0_LATEST_OUTPUT)

    if not raw_source.is_file():
        raise EpisodeStop(
            f"No existe salida raw esperada: {raw_source}"
        )

    if raw_source.stat().st_mtime_ns < inference_started_ns - 2_000_000_000:
        raise EpisodeStop(
            "La salida raw no pertenece a la inferencia actual"
        )

    raw_copy = run_dir / f"{prefix}_pi0_raw.json"
    raw_copy.write_text(
        raw_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    record = {
        "scene_path": str(scene_path),
        "scene": scene,
        "image_path": str(image_path),
        "measured_snapshot": measured_snapshot,
        "model_snapshot": model_snapshot,
        "measured_gripper_width_m": measured_gripper_width,
        "commanded_gripper_width_m": commanded_gripper_width_m,
        "state_path": str(state_path),
        "raw_path": str(raw_copy),
        "raw_sha256": sha256_file(raw_copy),
        "raw_actions": raw_actions,
        "raw_gripper": raw_actions[:, 6].tolist(),
        "capture_proof_path": str(
            run_dir / f"{prefix}_capture_proof.txt"
        ),
    }

    json_write(
        run_dir / f"{prefix}_observation.json",
        {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in record.items()
            if key not in {"scene", "raw_actions"}
        }
        | {
            "raw_actions": raw_actions.tolist(),
            "scene_path": str(scene_path),
        },
    )

    return record


_PI0_PHASE_OBSERVATION_CACHE: dict[str, dict[str, Any]] = {}
_PI0_COMMANDED_GRIPPER_WIDTH_M: float | None = None


def _phase_cache_contract(
    label: str,
) -> tuple[str | None, bool]:
    if label.startswith("approach_cycle_"):
        return "approach", True

    if label == "pregrasp_reinfer":
        return "pregrasp", False

    if label == "grasp_reinfer":
        return "pregrasp", True

    if label.startswith("transport_cycle_"):
        return "transport", True

    if label == "preplace_reinfer":
        return "preplace", False

    if label == "release_reinfer":
        return "preplace", True

    return None, False


def infer_fresh_pi0(
    run_dir: Path,
    config: Config,
    index: int,
    label: str,
    commanded_gripper_width_m: float,
) -> dict[str, Any]:
    global _PI0_COMMANDED_GRIPPER_WIDTH_M

    cache_key, may_reuse = _phase_cache_contract(
        str(label)
    )

    if (
        cache_key is not None
        and may_reuse
        and cache_key in _PI0_PHASE_OBSERVATION_CACHE
    ):
        observation = copy.deepcopy(
            _PI0_PHASE_OBSERVATION_CACHE[cache_key]
        )

        fresh_snapshot = action_controller.read_real_snapshot()
        fresh_snapshot = dict(fresh_snapshot)

        fresh_snapshot["gripper_width_m"] = float(
            commanded_gripper_width_m
        )

        observation["measured_snapshot"] = (
            fresh_snapshot
        )

        observation["phase_inference_contract"] = {
            "cache_key": cache_key,
            "requested_label": str(label),
            "fresh_remote_inference": False,
            "fresh_robot_snapshot": True,
            "reused_action_chunk": True,
            "reason": (
                "same_high_level_phase_or_event_pair"
            ),
        }

        print(
            f"label={label} cache={cache_key} "
            "remote_inference=false",
            flush=True,
        )

        return observation

    _PI0_COMMANDED_GRIPPER_WIDTH_M = float(
        commanded_gripper_width_m
    )

    observation = _infer_fresh_pi0_uncached(
        run_dir=run_dir,
        config=config,
        index=index,
        label=label,
        commanded_gripper_width_m=(
            commanded_gripper_width_m
        ),
    )

    if cache_key is not None:
        _PI0_PHASE_OBSERVATION_CACHE[
            cache_key
        ] = copy.deepcopy(observation)

    observation["phase_inference_contract"] = {
        "cache_key": cache_key,
        "requested_label": str(label),
        "fresh_remote_inference": True,
        "fresh_robot_snapshot": True,
        "reused_action_chunk": False,
    }

    return observation



@contextlib.contextmanager
def robot_node() -> Iterator[Any]:
    if rclpy.ok():
        rclpy.shutdown()

    rclpy.init()
    node = execution.PickLiftNode()

    try:
        node.wait_state()
        yield node
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


def require_single_gripper_subscriber(node: Any) -> int:
    deadline = time.monotonic() + 8.0

    while time.monotonic() < deadline:
        subscribers = int(
            node.grip_pub.get_subscription_count()
        )

        if subscribers >= 1:
            break

        rclpy.spin_once(node, timeout_sec=0.10)

    subscribers = int(
        node.grip_pub.get_subscription_count()
    )

    if subscribers != 1:
        raise EpisodeStop(
            "Se exige exactamente un suscriptor "
            "en /onrobot/finger_width_controller/commands; "
            f"recibido={subscribers}"
        )

    return subscribers


def current_robot_state(node: Any) -> tuple[np.ndarray, np.ndarray]:
    node.wait_state()

    q = finite_vector(
        node.current_q(),
        6,
        "robot.current_q",
    )

    tcp = finite_vector(
        node.current_tcp_xyz(),
        3,
        "robot.current_tcp_xyz",
    )

    return q, tcp


def build_ik_context(
    q_reference: np.ndarray,
    tcp_real_reference: np.ndarray,
):
    (
        scene_xml,
        model,
        data,
        tcp_site,
        tcp_id,
        qpos_ids,
        dof_ids,
    ) = robot_base.load_scene()

    robot_base.set_robot_q(
        model,
        data,
        qpos_ids,
        q_reference,
    )

    tcp_sim_reference, _, _ = (
        robot_base.get_site_pos_axis(
            data,
            tcp_id,
        )
    )

    offset = (
        np.asarray(
            tcp_real_reference,
            dtype=float,
        )
        - np.asarray(
            tcp_sim_reference,
            dtype=float,
        )
    )

    return {
        "scene_xml": scene_xml,
        "model": model,
        "data": data,
        "tcp_site": tcp_site,
        "tcp_id": tcp_id,
        "qpos_ids": qpos_ids,
        "dof_ids": dof_ids,
        "offset": offset,
    }


def _validate_free_space_joint_waypoint(
    *,
    context: dict[str, Any],
    q_command: np.ndarray,
    previous_q: np.ndarray,
    reference_tcp: np.ndarray,
    config: Config,
    label: str,
) -> dict[str, Any]:
    """
    Valida cada punto articular generado en espacio libre.
    
        Un endpoint IK válido no basta: todos los puntos que se
        enviarán al controlador deben mantener:
    
        - límites articulares MuJoCo;
        - workspace cartesiano;
        - error FK respecto al recorrido de referencia;
        - verticalidad del TCP.
    """

    q_value = finite_vector(
        q_command,
        6,
        f"{label}.q_command",
    )

    q_previous = finite_vector(
        previous_q,
        6,
        f"{label}.previous_q",
    )

    tcp_reference = finite_vector(
        reference_tcp,
        3,
        f"{label}.reference_tcp",
    )

    model = context["model"]

    qpos_addresses = np.asarray(
        model.jnt_qposadr,
        dtype=int,
    ).reshape(-1)

    joint_limit_checks = []

    for (
        joint_name,
        joint_value,
        qpos_id,
    ) in zip(
        robot_base.JOINTS,
        q_value,
        context["qpos_ids"],
    ):
        joint_ids = np.flatnonzero(
            qpos_addresses == int(qpos_id)
        )

        if joint_ids.size != 1:
            raise EpisodeStop(
                f"{label}: no se pudo resolver "
                f"el límite de {joint_name}"
            )

        joint_id = int(joint_ids[0])

        limited = bool(
            model.jnt_limited[joint_id]
        )

        lower, upper = [
            float(value)
            for value in model.jnt_range[joint_id]
        ]

        within = bool(
            (not limited)
            or (
                lower - 1e-6
                <= float(joint_value)
                <= upper + 1e-6
            )
        )

        check = {
            "joint": joint_name,
            "value_rad": float(joint_value),
            "limited": limited,
            "lower_rad": lower,
            "upper_rad": upper,
            "ok": within,
        }

        joint_limit_checks.append(check)

        if not within:
            raise EpisodeStop(
                f"{label}: {joint_name} fuera "
                f"de límites: {joint_value:.6f} rad "
                f"not in [{lower:.6f}, {upper:.6f}]"
            )

    robot_base.set_robot_q(
        model,
        context["data"],
        context["qpos_ids"],
        q_value,
    )

    (
        tcp_sim,
        z_axis,
        _,
    ) = robot_base.get_site_pos_axis(
        context["data"],
        context["tcp_id"],
    )

    tcp_real_fk = (
        np.asarray(tcp_sim, dtype=float)
        + context["offset"]
    )

    fk_error_m = float(
        np.linalg.norm(
            tcp_real_fk - tcp_reference
        )
    )

    if not workspace_ok(
        tcp_real_fk,
        contact=False,
    ):
        raise EpisodeStop(
            f"{label}: FK intermedia fuera "
            f"del workspace: "
            f"{tcp_real_fk.tolist()}"
        )

    if fk_error_m > float(
        config.max_fk_error_m
    ):
        raise EpisodeStop(
            f"{label}: trayectoria articular "
            f"se separa del recorrido cartesiano: "
            f"fk_error={fk_error_m:.6f} m > "
            f"{float(config.max_fk_error_m):.6f} m; "
            f"reference={tcp_reference.tolist()} "
            f"fk={tcp_real_fk.tolist()}"
        )

    z_axis = np.asarray(
        z_axis,
        dtype=float,
    )

    axis_norm = float(
        np.linalg.norm(z_axis)
    )

    if axis_norm <= 1e-12:
        raise EpisodeStop(
            f"{label}: eje TCP nulo"
        )

    z_axis /= axis_norm

    topdown_dot = float(
        np.clip(
            np.dot(z_axis, TARGET_AXIS),
            -1.0,
            1.0,
        )
    )

    tilt_deg = float(
        math.degrees(
            math.acos(topdown_dot)
        )
    )

    if tilt_deg > float(
        config.max_tilt_deg
    ):
        raise EpisodeStop(
            f"{label}: tilt intermedio "
            f"{tilt_deg:.6f}° > "
            f"{float(config.max_tilt_deg):.6f}°"
        )

    joint_step_rad = float(
        np.max(
            np.abs(
                q_value - q_previous
            )
        )
    )

    return {
        "q_rad": q_value.tolist(),
        "reference_tcp_base_m":
            tcp_reference.tolist(),
        "fk_real_base_m":
            tcp_real_fk.tolist(),
        "fk_error_m": fk_error_m,
        "topdown_dot": topdown_dot,
        "tilt_deg": tilt_deg,
        "workspace_ok": True,
        "joint_limits_ok": True,
        "joint_limit_checks":
            joint_limit_checks,
        "joint_step_rad":
            joint_step_rad,
        "ok": True,
    }

def _solve_cartesian_targets_once(
    q_start: np.ndarray,
    tcp_start: np.ndarray,
    targets: list[np.ndarray],
    config: Config,
    label: str,
    max_joint_step_rad: float,
    contact: bool = False,
    context: dict[str, Any] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    q_start = finite_vector(q_start, 6, f"{label}.q_start")
    tcp_start = finite_vector(
        tcp_start,
        3,
        f"{label}.tcp_start",
    )

    if context is None:
        context = build_ik_context(
            q_reference=q_start,
            tcp_real_reference=tcp_start,
        )

    q_seed = q_start.copy()
    solutions: list[np.ndarray] = []
    waypoints: list[dict[str, Any]] = []

    for index, target_value in enumerate(targets):
        target = finite_vector(
            target_value,
            3,
            f"{label}.target[{index}]",
        )

        if not workspace_ok(target, contact=contact):
            raise EpisodeStop(
                f"{label}: waypoint fuera del workspace: "
                f"{target.tolist()}"
            )

        # El solver vertical admite por defecto errores de hasta
        # 30 mm. Para chunks temporales eso puede devolver la propia semilla
        # como "solución" sin mover el TCP. El supervisor fija una tolerancia
        # coherente con el tamaño real de cada paso y con su propio guard FK.
        robot_base.set_robot_q(
            context["model"],
            context["data"],
            context["qpos_ids"],
            q_seed,
        )

        seed_tcp_sim, _, _ = robot_base.get_site_pos_axis(
            context["data"],
            context["tcp_id"],
        )

        seed_tcp_real = (
            np.asarray(seed_tcp_sim, dtype=float)
            + context["offset"]
        )

        seed_distance_to_target_m = float(
            np.linalg.norm(
                seed_tcp_real - target
            )
        )

        solver_position_tolerance_m = min(
            config.max_fk_error_m,
            max(
                0.0005,
                0.20 * seed_distance_to_target_m,
            ),
        )

        previous_solver_tolerance = os.environ.get(
            "VLA_VERTICAL_IK_POS_TOL_M"
        )

        os.environ["VLA_VERTICAL_IK_POS_TOL_M"] = (
            f"{solver_position_tolerance_m:.12g}"
        )

        try:
            solution, target_sim = execution.solve_point_axis(
                context["model"],
                context["data"],
                context["tcp_id"],
                context["qpos_ids"],
                context["dof_ids"],
                target,
                q_seed,
                context["offset"],
                TARGET_AXIS,
            )
        finally:
            if previous_solver_tolerance is None:
                os.environ.pop(
                    "VLA_VERTICAL_IK_POS_TOL_M",
                    None,
                )
            else:
                os.environ[
                    "VLA_VERTICAL_IK_POS_TOL_M"
                ] = previous_solver_tolerance

        if not isinstance(solution, dict):
            raise EpisodeStop(
                f"{label}: solve_point_axis devolvió valor inválido"
            )

        if not bool(solution.get("ok", False)):
            raise EpisodeStop(
                f"{label}: IK inválida en waypoint {index}: "
                f"{solution}"
            )

        q_solution = finite_vector(
            solution.get("q"),
            6,
            f"{label}.q_solution[{index}]",
        )

        joint_step = float(
            np.max(
                np.abs(
                    q_solution - q_seed
                )
            )
        )

        if joint_step > max_joint_step_rad:
            raise EpisodeStop(
                f"{label}: salto articular {joint_step:.4f} rad "
                f"> {max_joint_step_rad:.4f} rad"
            )

        robot_base.set_robot_q(
            context["model"],
            context["data"],
            context["qpos_ids"],
            q_solution,
        )

        tcp_sim, z_axis, _ = (
            robot_base.get_site_pos_axis(
                context["data"],
                context["tcp_id"],
            )
        )

        tcp_real_fk = (
            np.asarray(tcp_sim, dtype=float)
            + context["offset"]
        )

        fk_error_m = float(
            np.linalg.norm(
                tcp_real_fk - target
            )
        )

        if seed_distance_to_target_m > 1e-9:
            remaining_fraction = (
                fk_error_m
                / seed_distance_to_target_m
            )

            progress_fraction = (
                seed_distance_to_target_m
                - fk_error_m
            ) / seed_distance_to_target_m
        else:
            remaining_fraction = 0.0
            progress_fraction = 1.0

        progress_ok = bool(
            (
                seed_distance_to_target_m
                <= config.max_fk_error_m
                and fk_error_m
                <= config.max_fk_error_m
            )
            or (
                progress_fraction >= 0.75
                and remaining_fraction <= 0.25
            )
        )

        if not progress_ok:
            raise EpisodeStop(
                f"{label}: waypoint {index} no avanza lo suficiente: "
                f"seed_distance={seed_distance_to_target_m:.4f} m, "
                f"fk_error={fk_error_m:.4f} m, "
                f"progress={progress_fraction:.3f}, "
                f"remaining={remaining_fraction:.3f}"
            )

        z_axis = np.asarray(
            z_axis,
            dtype=float,
        )

        z_axis /= max(
            1e-12,
            float(np.linalg.norm(z_axis)),
        )

        topdown_dot = float(
            np.dot(z_axis, TARGET_AXIS)
        )

        topdown_dot = max(
            -1.0,
            min(1.0, topdown_dot),
        )

        tilt_deg = float(
            math.degrees(
                math.acos(topdown_dot)
            )
        )

        if fk_error_m > config.max_fk_error_m:
            raise EpisodeStop(
                f"{label}: error FK {fk_error_m:.4f} m "
                f"> {config.max_fk_error_m:.4f} m"
            )

        if tilt_deg > config.max_tilt_deg:
            raise EpisodeStop(
                f"{label}: tilt {tilt_deg:.3f}° "
                f"> {config.max_tilt_deg:.3f}°"
            )

        waypoints.append(
            {
                "index": index,
                "target_real_base_m": target.tolist(),
                "target_sim_m": (
                    np.asarray(
                        target_sim,
                        dtype=float,
                    ).tolist()
                ),
                "q_seed_rad": q_seed.tolist(),
                "q_solution_rad": q_solution.tolist(),
                "max_abs_joint_delta_rad": joint_step,
                "seed_fk_real_base_m": seed_tcp_real.tolist(),
                "seed_distance_to_target_m": (
                    seed_distance_to_target_m
                ),
                "solver_position_tolerance_m": (
                    solver_position_tolerance_m
                ),
                "fk_real_base_m": tcp_real_fk.tolist(),
                "fk_error_m": fk_error_m,
                "progress_fraction": progress_fraction,
                "remaining_fraction": remaining_fraction,
                "progress_ok": progress_ok,
                "topdown_dot": topdown_dot,
                "tilt_deg": tilt_deg,
                "solver": {
                    key: value
                    for key, value in solution.items()
                    if key != "q"
                },
            }
        )

        solutions.append(q_solution.copy())
        q_seed = q_solution.copy()

    report = {
        "label": label,
        "scene_xml": str(context["scene_xml"]),
        "tcp_site": context["tcp_site"],
        "q_start_rad": q_start.tolist(),
        "tcp_start_base_m": tcp_start.tolist(),
        "offset_real_minus_sim_m": (
            context["offset"].tolist()
        ),
        "waypoints": waypoints,
        "all_steps_ok": True,
        "max_fk_error_m": max(
            (
                item["fk_error_m"]
                for item in waypoints
            ),
            default=0.0,
        ),
        "max_abs_joint_delta_rad": max(
            (
                item[
                    "max_abs_joint_delta_rad"
                ]
                for item in waypoints
            ),
            default=0.0,
        ),
        "max_tilt_deg": max(
            (
                item["tilt_deg"]
                for item in waypoints
            ),
            default=0.0,
        ),
    }

    return solutions, report



def _ik_branch_policy(label: str) -> dict[str, Any] | None:
    normalized = str(label).lower()
    far_side = any(
        token in normalized
        for token in (
            "transport",
            "place",
            "retreat",
            "held_chain",
            "return_chain",
        )
    )
    if not far_side:
        return None
    return {
        "schema": "VLA_branch_aware_cartesian_solver",
        "maximum_shoulder_lift_rad": -0.80,
        "minimum_elbow_rad": 0.65,
        "maximum_shoulder_home_delta_rad": 0.72,
        "maximum_elbow_home_delta_rad": 1.08,
        "minimum_arm_fold_rad": 1.45,
    }


def _ik_branch_metrics(
    q_value: np.ndarray,
    label: str,
) -> dict[str, Any]:
    q = finite_vector(q_value, 6, f"{label}.branch_q")
    policy = _ik_branch_policy(label)
    if policy is None:
        return {
            "enabled": False,
            "safe": True,
            "weakest_margin_rad": float("inf"),
            "weakest_constraint": None,
            "q2_rad": float(q[1]),
            "q3_rad": float(q[2]),
            "arm_fold_rad": float(q[2] - q[1]),
        }

    home = finite_vector(
        expected_global_home_q(),
        6,
        f"{label}.home_reference",
    )
    q2 = float(q[1])
    q3 = float(q[2])
    q2_delta = abs(q2 - float(home[1]))
    q3_delta = abs(q3 - float(home[2]))
    arm_fold = q3 - q2

    margins = {
        "shoulder_hard_limit": (
            policy["maximum_shoulder_lift_rad"] - q2
        ),
        "elbow_hard_limit": (
            q3 - policy["minimum_elbow_rad"]
        ),
        "shoulder_home_distance": (
            policy["maximum_shoulder_home_delta_rad"] - q2_delta
        ),
        "elbow_home_distance": (
            policy["maximum_elbow_home_delta_rad"] - q3_delta
        ),
        "arm_fold": (
            arm_fold - policy["minimum_arm_fold_rad"]
        ),
    }
    weakest_constraint, weakest_margin = min(
        margins.items(),
        key=lambda item: item[1],
    )

    return {
        "enabled": True,
        "safe": bool(weakest_margin >= -1e-9),
        "weakest_margin_rad": float(weakest_margin),
        "weakest_constraint": weakest_constraint,
        "q2_rad": q2,
        "q3_rad": q3,
        "q2_home_delta_rad": q2_delta,
        "q3_home_delta_rad": q3_delta,
        "arm_fold_rad": arm_fold,
        "margins_rad": {
            key: float(value)
            for key, value in margins.items()
        },
        "policy": policy,
    }


def _ik_branch_seed_candidates(
    q_current: np.ndarray,
    label: str,
) -> list[tuple[str, np.ndarray]]:
    current = finite_vector(
        q_current,
        6,
        f"{label}.q_current",
    )
    output: list[tuple[str, np.ndarray]] = [
        ("current_branch", current.copy())
    ]
    if _ik_branch_policy(label) is None:
        return output

    home = finite_vector(
        expected_global_home_q(),
        6,
        f"{label}.home_reference",
    )

    biased_1 = current.copy()
    biased_1[1] -= 0.08
    biased_1[2] += 0.10
    output.append(("fold_bias_small", biased_1))

    biased_2 = current.copy()
    biased_2[1] -= 0.16
    biased_2[2] += 0.20
    output.append(("fold_bias_medium", biased_2))

    home_blend = current.copy()
    home_blend[1] = 0.80 * current[1] + 0.20 * home[1]
    home_blend[2] = 0.80 * current[2] + 0.20 * home[2]
    output.append(("home_fold_blend", home_blend))

    unique: list[tuple[str, np.ndarray]] = []
    for name, candidate in output:
        candidate = np.clip(
            finite_vector(candidate, 6, f"{label}.{name}"),
            -2.0 * math.pi,
            2.0 * math.pi,
        )
        if any(
            float(np.max(np.abs(candidate - previous))) < 1e-6
            for _, previous in unique
        ):
            continue
        unique.append((name, candidate))
    return unique


def _solve_branch_aware_endpoint(
    *,
    q_current: np.ndarray,
    tcp_current: np.ndarray,
    target: np.ndarray,
    config: Config,
    label: str,
    max_joint_step_rad: float,
    context: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    q_current = finite_vector(
        q_current,
        6,
        f"{label}.q_current",
    )
    tcp_current = finite_vector(
        tcp_current,
        3,
        f"{label}.tcp_current",
    )
    target = finite_vector(
        target,
        3,
        f"{label}.target",
    )

    solver_limit_rad = min(
        0.55,
        max(0.30, 3.0 * float(max_joint_step_rad)),
    )

    rejected: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []

    for seed_name, q_seed_candidate in _ik_branch_seed_candidates(
        q_current,
        label,
    ):
        try:
            solutions, report = _solve_cartesian_targets_once(
                q_start=q_seed_candidate,
                tcp_start=tcp_current,
                targets=[target],
                config=config,
                label=f"{label}.{seed_name}",
                max_joint_step_rad=solver_limit_rad,
                contact=False,
                context=context,
            )

            if len(solutions) != 1:
                raise EpisodeStop(
                    f"{label}.{seed_name}: solución no única"
                )

            q_solution = finite_vector(
                solutions[0],
                6,
                f"{label}.{seed_name}.q_solution",
            )
            waypoint_rows = list(report.get("waypoints", []))
            if len(waypoint_rows) != 1:
                raise EpisodeStop(
                    f"{label}.{seed_name}: informe FK no único"
                )

            waypoint = dict(waypoint_rows[0])
            actual_step = float(
                np.max(np.abs(q_solution - q_current))
            )
            branch = _ik_branch_metrics(
                q_solution,
                label,
            )

            if actual_step > float(max_joint_step_rad) + 1e-9:
                raise EpisodeStop(
                    f"{label}.{seed_name}: paso desde la rama actual "
                    f"{actual_step:.6f}rad > "
                    f"{float(max_joint_step_rad):.6f}rad"
                )

            if not bool(branch["safe"]):
                raise EpisodeStop(
                    f"{label}.{seed_name}: solución IK en rama insegura; "
                    f"q2={branch['q2_rad']:.6f}rad "
                    f"q3={branch['q3_rad']:.6f}rad "
                    f"fold={branch['arm_fold_rad']:.6f}rad "
                    f"violacion={branch['weakest_constraint']} "
                    f"margen={branch['weakest_margin_rad']:.6f}rad"
                )

            fk_error = float(
                waypoint.get("fk_error_m", 999.0)
            )
            tilt = float(
                waypoint.get("tilt_deg", 999.0)
            )
            seed_bias = float(
                np.max(np.abs(q_seed_candidate - q_current))
            )
            branch_margin = float(
                branch["weakest_margin_rad"]
            )
            finite_margin = (
                branch_margin
                if math.isfinite(branch_margin)
                else 1.0
            )
            score = (
                actual_step
                + 4.0 * fk_error
                + 0.002 * tilt
                + 0.05 * seed_bias
                - 0.10 * min(0.25, finite_margin)
            )

            valid.append(
                {
                    "seed_name": seed_name,
                    "q_solution": q_solution,
                    "waypoint": waypoint,
                    "branch": branch,
                    "actual_step_rad": actual_step,
                    "seed_bias_rad": seed_bias,
                    "score": float(score),
                }
            )

            if seed_name == "current_branch":
                break

        except Exception as exc:
            rejected.append(
                {
                    "seed_name": seed_name,
                    "reason": str(exc),
                }
            )

    if not valid:
        raise EpisodeStop(
            f"{label}: ninguna semilla conserva una rama segura: "
            f"{rejected}"
        )

    selected = min(
        valid,
        key=lambda item: (
            item["score"],
            item["actual_step_rad"],
            item["seed_bias_rad"],
        ),
    )
    waypoint = dict(selected["waypoint"])
    waypoint["branch_aware_solver"] = {
        "selected_seed": selected["seed_name"],
        "selected_score": selected["score"],
        "actual_step_from_current_rad": (
            selected["actual_step_rad"]
        ),
        "seed_bias_rad": selected["seed_bias_rad"],
        "branch": selected["branch"],
        "rejected_seeds": rejected,
    }

    print(
        f"label={label} "
        f"seed={selected['seed_name']} "
        f"step={selected['actual_step_rad']:.4f}rad "
        f"q2={selected['branch']['q2_rad']:.4f}rad "
        f"q3={selected['branch']['q3_rad']:.4f}rad "
        f"fold={selected['branch']['arm_fold_rad']:.4f}rad "
        f"margin={selected['branch']['weakest_margin_rad']:.4f}rad",
        flush=True,
    )

    return (
        selected["q_solution"].copy(),
        waypoint,
    )


def solve_cartesian_targets(
    q_start: np.ndarray,
    tcp_start: np.ndarray,
    targets: list[np.ndarray],
    config: Config,
    label: str,
    max_joint_step_rad: float,
    contact: bool = False,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    q_initial = finite_vector(
        q_start,
        6,
        f"{label}.q_start",
    )
    tcp_initial = finite_vector(
        tcp_start,
        3,
        f"{label}.tcp_start",
    )
    normalized_targets = [
        finite_vector(
            target,
            3,
            f"{label}.target[{index}]",
        )
        for index, target in enumerate(targets)
    ]

    if not normalized_targets:
        raise EpisodeStop(
            f"{label}: lista de targets vacía"
        )

    if contact:
        return _solve_cartesian_targets_once(
            q_start=q_initial,
            tcp_start=tcp_initial,
            targets=normalized_targets,
            config=config,
            label=label,
            max_joint_step_rad=max_joint_step_rad,
            contact=True,
        )

    if (
        not math.isfinite(max_joint_step_rad)
        or max_joint_step_rad <= 0.0
    ):
        raise EpisodeStop(
            f"{label}: max_joint_step_rad inválido"
        )

    maximum_generated_targets = 160
    maximum_depth = 6
    minimum_cartesian_segment_m = 0.004

    normalized_label = str(label).lower()
    cartesian_segment_limit_m = (
        0.025
        if any(
            token in normalized_label
            for token in (
                "transport",
                "place",
                "retreat",
            )
        )
        else 0.040
    )

    context = build_ik_context(
        q_reference=q_initial,
        tcp_real_reference=tcp_initial,
    )

    current_q = q_initial.copy()
    current_tcp = tcp_initial.copy()

    solutions: list[np.ndarray] = []
    waypoint_reports: list[dict[str, Any]] = []
    elementary_reports: list[dict[str, Any]] = []
    subdivision_count = 0
    maximum_observed_joint_step = 0.0
    maximum_fk_error_m = 0.0
    maximum_tilt_deg = 0.0

    def accept_target(
        target_value: np.ndarray,
        *,
        source_index: int,
        depth: int,
        inserted: bool,
    ) -> None:
        nonlocal current_q
        nonlocal current_tcp
        nonlocal subdivision_count
        nonlocal maximum_observed_joint_step
        nonlocal maximum_fk_error_m
        nonlocal maximum_tilt_deg

        target_value = finite_vector(
            target_value,
            3,
            f"{label}.source[{source_index}].depth[{depth}]",
        )

        if len(solutions) >= maximum_generated_targets:
            raise EpisodeStop(
                f"{label}: se superarían "
                f"{maximum_generated_targets} waypoints"
            )

        segment_distance = float(
            np.linalg.norm(target_value - current_tcp)
        )

        if (
            segment_distance > cartesian_segment_limit_m
            and depth < maximum_depth
        ):
            midpoint = (
                current_tcp
                + 0.5 * (target_value - current_tcp)
            )
            subdivision_count += 1
            print(
                f"label={label} "
                f"source={source_index} "
                f"depth={depth} "
                f"segment={segment_distance:.4f}m",
                flush=True,
            )
            accept_target(
                midpoint,
                source_index=source_index,
                depth=depth + 1,
                inserted=True,
            )
            accept_target(
                target_value,
                source_index=source_index,
                depth=depth + 1,
                inserted=inserted,
            )
            return

        q_before = current_q.copy()
        tcp_before = current_tcp.copy()

        try:
            q_solution, waypoint = (
                _solve_branch_aware_endpoint(
                    q_current=q_before,
                    tcp_current=tcp_before,
                    target=target_value,
                    config=config,
                    label=(
                        f"{label}.source[{source_index}]"
                        f".depth[{depth}]"
                    ),
                    max_joint_step_rad=max_joint_step_rad,
                    context=context,
                )
            )
        except Exception:
            if (
                segment_distance > minimum_cartesian_segment_m
                and depth < maximum_depth
            ):
                midpoint = (
                    tcp_before
                    + 0.5 * (target_value - tcp_before)
                )
                subdivision_count += 1
                print(
                    f"label={label} "
                    f"source={source_index} "
                    f"depth={depth} "
                    f"segment={segment_distance:.4f}m",
                    flush=True,
                )
                accept_target(
                    midpoint,
                    source_index=source_index,
                    depth=depth + 1,
                    inserted=True,
                )
                accept_target(
                    target_value,
                    source_index=source_index,
                    depth=depth + 1,
                    inserted=inserted,
                )
                return
            raise

        step = float(
            np.max(np.abs(q_solution - q_before))
        )
        fk_real = finite_vector(
            waypoint.get("fk_real_base_m"),
            3,
            f"{label}.accepted_fk[{len(solutions)}]",
        )
        fk_error = float(
            waypoint.get("fk_error_m", 999.0)
        )
        tilt = float(
            waypoint.get("tilt_deg", 999.0)
        )

        solutions.append(q_solution.copy())
        waypoint_reports.append(
            {
                **waypoint,
                "index": len(waypoint_reports) + 1,
                "adaptive_source_index": source_index,
                "adaptive_depth": depth,
                "adaptive_inserted": inserted,
                "branch_preserved": True,
                "q_rad": q_solution.tolist(),
                "joint_step_rad": step,
                "reference_tcp_base_m": target_value.tolist(),
                "ok": True,
            }
        )
        elementary_reports.append(
            {
                "source_index": source_index,
                "depth": depth,
                "inserted": inserted,
                "target_base_m": target_value.tolist(),
                "q_rad": q_solution.tolist(),
                "joint_step_rad": step,
                "fk_real_base_m": fk_real.tolist(),
                "fk_error_m": fk_error,
                "tilt_deg": tilt,
                "branch": _ik_branch_metrics(
                    q_solution,
                    label,
                ),
            }
        )

        current_q = q_solution.copy()
        current_tcp = fk_real.copy()
        maximum_observed_joint_step = max(
            maximum_observed_joint_step,
            step,
        )
        maximum_fk_error_m = max(
            maximum_fk_error_m,
            fk_error,
        )
        maximum_tilt_deg = max(
            maximum_tilt_deg,
            tilt,
        )

    for source_index, target in enumerate(
        normalized_targets,
        start=1,
    ):
        accept_target(
            target,
            source_index=source_index,
            depth=0,
            inserted=False,
        )

    if not solutions:
        raise EpisodeStop(
            f"{label}: no se produjeron soluciones"
        )

    report = {
        "label": label,
        "planner": "branch_aware_cartesian_subdivision",
        "contact_mode": False,
        "q_start_rad": q_initial.tolist(),
        "tcp_start_base_m": tcp_initial.tolist(),
        "requested_targets_base_m": [
            target.tolist()
            for target in normalized_targets
        ],
        "target_base_m": normalized_targets[-1].tolist(),
        "result_target_preserved": True,
        "vla_endpoint_modified": False,
        "requested_target_count": len(normalized_targets),
        "executed_target_count": len(solutions),
        "subdivision_count": subdivision_count,
        "generated_target_count": len(solutions),
        "max_allowed_joint_step_rad": float(
            max_joint_step_rad
        ),
        "cartesian_segment_limit_m": (
            cartesian_segment_limit_m
        ),
        "max_abs_joint_delta_rad": (
            maximum_observed_joint_step
        ),
        "max_fk_error_m": maximum_fk_error_m,
        "max_tilt_deg": maximum_tilt_deg,
        "branch_policy": _ik_branch_policy(label),
        "waypoints": waypoint_reports,
        "elementary_reports": elementary_reports,
        "all_steps_ok": bool(
            waypoint_reports
            and all(
                bool(item.get("ok", False))
                for item in waypoint_reports
            )
        ),
    }

    print(
        f"label={label} "
        f"requested={len(normalized_targets)} "
        f"executed={len(solutions)} "
        f"subdivisions={subdivision_count} "
        f"max_joint_step="
        f"{maximum_observed_joint_step:.6f}rad",
        flush=True,
    )

    return solutions, report

def _make_vla_chunk_targets_chunk_only(
    raw_actions: np.ndarray,
    tcp_start: np.ndarray,
    goal_xyz: np.ndarray,
    config: Config,
    phase: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    actions = np.asarray(
        raw_actions,
        dtype=float,
    )

    if (
        actions.shape != (10, 7)
        or not np.all(np.isfinite(actions))
    ):
        raise EpisodeStop(
            f"{phase}: actions inválidas: "
            f"{actions.shape}"
        )

    start = finite_vector(
        tcp_start,
        3,
        f"{phase}.tcp_start",
    )

    goal = finite_vector(
        goal_xyz,
        3,
        f"{phase}.goal",
    )

    alpha = float(
        os.environ.get(
            "VLA_INFLUENCE_ALPHA",
            "1.00",
        )
    )

    if not 0.0 <= alpha <= 1.0:
        raise EpisodeStop(
            "VLA_INFLUENCE_ALPHA debe estar "
            "entre 0 y 1"
        )

    translation_scale_m = float(
        os.environ.get(
            "VLA_PI0_TRANSLATION_SCALE_M",
            "0.050",
        )
    )

    maximum_step_m = float(
        os.environ.get(
            "VLA_PI0_MAX_SAFE_VLA_STEP_M",
            "0.040",
        )
    )

    maximum_phase_displacement_m = float(
        os.environ.get(
            "VLA_PI0_MAX_SAFE_PHASE_DISPLACEMENT_M",
            "0.120",
        )
    )

    if not 0.010 <= maximum_step_m <= 0.060:
        raise EpisodeStop(
            "VLA_PI0_MAX_SAFE_VLA_STEP_M "
            "fuera de rango"
        )

    if not (
        0.050
        <= maximum_phase_displacement_m
        <= 0.200
    ):
        raise EpisodeStop(
            "VLA_PI0_MAX_SAFE_PHASE_DISPLACEMENT_M "
            "fuera de rango"
        )

    steps = int(config.chunk_steps)

    model_deltas_requested = (
        np.clip(
            actions[:steps, :3],
            -1.0,
            1.0,
        )
        * translation_scale_m
        * alpha
    )

    base_deltas_requested = (
        MODEL_TO_BASE
        @ model_deltas_requested.T
    ).T

    accepted = []
    applied_deltas = []
    current = start.copy()
    clipped_steps = 0
    rejected_reason = None

    for index, requested_delta in enumerate(
        base_deltas_requested,
        start=1,
    ):
        requested_delta = np.asarray(
            requested_delta,
            dtype=float,
        )

        requested_norm = float(
            np.linalg.norm(requested_delta)
        )

        if requested_norm > maximum_step_m:
            applied_delta = (
                requested_delta
                * maximum_step_m
                / requested_norm
            )
            clipped_steps += 1
        else:
            applied_delta = (
                requested_delta.copy()
            )

        candidate = current + applied_delta

        total_displacement = float(
            np.linalg.norm(
                candidate - start
            )
        )

        if (
            total_displacement
            > maximum_phase_displacement_m
        ):
            direction = candidate - start
            direction_norm = float(
                np.linalg.norm(direction)
            )

            if direction_norm <= 1e-12:
                rejected_reason = (
                    "zero_direction_after_limit"
                )
                break

            candidate = (
                start
                + direction
                * maximum_phase_displacement_m
                / direction_norm
            )

            applied_delta = (
                candidate - current
            )

            clipped_steps += 1

        if not workspace_ok(candidate):
            rejected_reason = (
                f"workspace_step_{index}"
            )
            break

        accepted.append(candidate.copy())
        applied_deltas.append(
            applied_delta.copy()
        )

        current = candidate.copy()

    if not accepted:
        raise EpisodeStop(
            f"{phase}: ninguna acción VLA puede "
            "aplicarse dentro de los límites seguros"
        )

    requested_path_length = float(
        np.sum(
            np.linalg.norm(
                base_deltas_requested,
                axis=1,
            )
        )
    )

    applied_deltas_array = np.asarray(
        applied_deltas,
        dtype=float,
    )

    applied_path_length = float(
        np.sum(
            np.linalg.norm(
                applied_deltas_array,
                axis=1,
            )
        )
    )

    retention_ratio = (
        applied_path_length
        / requested_path_length
        if requested_path_length > 1e-12
        else 1.0
    )

    distance_before = float(
        np.linalg.norm(start - goal)
    )

    distances = [
        float(np.linalg.norm(point - goal))
        for point in accepted
    ]

    report = {
        "phase": phase,
        "integration": (
            "maximum_vla_influence_then_"
            "geometric_phase_completion"
        ),
        "vla_influence_alpha": alpha,
        "translation_scale_m": (
            translation_scale_m
        ),
        "immediate_progress_required": False,
        "monotonic_progress_required": False,
        "goal_projection_applied": False,
        "tcp_start_base_m": start.tolist(),
        "goal_base_m": goal.tolist(),
        "model_deltas_requested_m": (
            model_deltas_requested.tolist()
        ),
        "base_deltas_requested_m": (
            base_deltas_requested.tolist()
        ),
        "base_deltas_applied_m": (
            applied_deltas_array.tolist()
        ),
        "accepted_candidates_base_m": [
            point.tolist()
            for point in accepted
        ],
        "distance_before_m": (
            distance_before
        ),
        "distance_after_each_step_m": (
            distances
        ),
        "distance_after_vla_chunk_m": (
            distances[-1]
        ),
        "requested_vla_path_length_m": (
            requested_path_length
        ),
        "applied_vla_path_length_m": (
            applied_path_length
        ),
        "effective_vla_retention_ratio": (
            retention_ratio
        ),
        "clipped_steps": clipped_steps,
        "rejected_reason": rejected_reason,
        "raw_gripper": (
            actions[:, 6].tolist()
        ),
    }

    print(
        f"phase={phase} "
        f"alpha={alpha:.2f} "
        f"accepted={len(accepted)}/{steps} "
        f"retention={retention_ratio:.3f}",
        flush=True,
    )

    return accepted, report



def _phase_segment(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    maximum_step_m: float,
    phase: str,
) -> list[np.ndarray]:
    start = finite_vector(
        start_xyz,
        3,
        f"{phase}.segment_start",
    )

    end = finite_vector(
        end_xyz,
        3,
        f"{phase}.segment_end",
    )

    distance = float(
        np.linalg.norm(end - start)
    )

    if distance <= 1e-9:
        return []

    count = max(
        1,
        int(math.ceil(distance / maximum_step_m)),
    )

    output: list[np.ndarray] = []

    for index in range(1, count + 1):
        fraction = index / count

        point = (
            start
            + fraction * (end - start)
        )

        if not workspace_ok(point):
            raise EpisodeStop(
                f"{phase}: finalización por fases "
                f"fuera del workspace: {point.tolist()}"
            )

        output.append(point.copy())

    return output


def make_vla_chunk_targets(
    raw_actions: np.ndarray,
    tcp_start: np.ndarray,
    goal_xyz: np.ndarray,
    config: Config,
    phase: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    chunk_targets, report = (
        _make_vla_chunk_targets_chunk_only(
            raw_actions=raw_actions,
            tcp_start=tcp_start,
            goal_xyz=goal_xyz,
            config=config,
            phase=phase,
        )
    )

    if not chunk_targets:
        raise EpisodeStop(
            f"{phase}: chunk π0 vacío"
        )

    goal = finite_vector(
        goal_xyz,
        3,
        f"{phase}.phase_goal",
    )

    current = finite_vector(
        chunk_targets[-1],
        3,
        f"{phase}.chunk_end",
    )

    maximum_step_m = float(
        os.environ.get(
            "VLA_PI0_PHASE_COMPLETION_STEP_M",
            "0.030",
        )
    )

    if not 0.010 <= maximum_step_m <= 0.040:
        raise EpisodeStop(
            "VLA_PI0_PHASE_COMPLETION_STEP_M "
            "debe estar entre 0.010 y 0.040"
        )

    # No todas las fases deben usar el mismo plano
    # horizontal de tránsito.
    # Approach:
    #   el robot todavía no transporta el objeto;
    #   utilizar siempre z=0.30 puede sacar los puntos
    #   alejados de la región alcanzable con eje vertical.
    # Transport:
    #   el robot sostiene el objeto y mantiene el plano
    # alto configurado para mantener el despeje.
    phase_text = str(
        phase
    ).strip().lower()

    approach_clearance_m = None
    approach_descent_after_chunk_m = None

    if phase_text.startswith(
        "approach"
    ):
        approach_clearance_m = float(
            os.environ.get(
                "VLA_APPROACH_GOAL_CLEARANCE_M",
                "0.040",
            )
        )

        if not (
            math.isfinite(
                approach_clearance_m
            )
            and 0.030
            <= approach_clearance_m
            <= 0.100
        ):
            raise EpisodeStop(
                "VLA_APPROACH_GOAL_CLEARANCE_M "
                "debe estar entre 0.030 "
                "y 0.100 m"
            )

        # El chunk VLA se conserva completo.
        # Después de ejecutarlo, el completion puede
        # descender en el mismo XY hasta un plano de
        # approach alcanzable. No se obliga a mantener
        # la altura result del chunk durante todo el
        # desplazamiento horizontal.
        transit_z = (
            float(goal[2])
            + approach_clearance_m
        )

        approach_descent_after_chunk_m = max(
            0.0,
            float(current[2])
            - transit_z,
        )

        transit_policy = (
            "vla_chunk_then_descend_to_"
            "goal_relative_approach_plane"
        )

        completion_route = (
            "preserve_vla_chunk_"
            "vertical_to_goal_relative_plane_"
            "horizontal_to_goal_"
            "vertical_to_safe_goal"
        )

    else:
        transit_z = max(
            float(config.transit_z_m),
            float(current[2]),
            float(goal[2]) + 0.080,
        )

        transit_policy = (
            "configured_high_transit"
        )

        completion_route = (
            "vertical_to_transit_"
            "horizontal_to_goal_"
            "vertical_to_safe_goal"
        )

    if not math.isfinite(
        transit_z
    ):
        raise EpisodeStop(
            f"{phase}: transit_z no finita"
        )

    if phase_text.startswith(
        "approach"
    ):
        # En approach está permitido descender desde
        # el endpoint del chunk antes de desplazarse
        # horizontalmente. Solo debe mantenerse por
        # encima del safe-above solicitado.
        if (
            transit_z
            < float(goal[2]) - 1e-9
        ):
            raise EpisodeStop(
                f"{phase}: plano de approach "
                f"{transit_z:.4f} m queda por "
                f"debajo del objetivo "
                f"{float(goal[2]):.4f} m"
            )

    else:
        # Transport continúa utilizando un tránsito
        # alto, nunca inferior al inicio o al objetivo.
        if transit_z < max(
            float(current[2]),
            float(goal[2]),
        ) - 1e-9:
            raise EpisodeStop(
                f"{phase}: transit_z="
                f"{transit_z:.4f} queda por "
                "debajo del inicio o del objetivo"
            )

    print(
        f"phase={phase} "
        f"policy={transit_policy} "
        f"current_z={float(current[2]):.4f} "
        f"goal_z={float(goal[2]):.4f} "
        f"transit_z={transit_z:.4f} "
        f"descent_after_chunk="
        f"{float(approach_descent_after_chunk_m or 0.0):.4f} "
        f"configured_transit_z="
        f"{float(config.transit_z_m):.4f}",
        flush=True,
    )

    transit_entry = current.copy()
    transit_entry[2] = transit_z

    horizontal = goal.copy()
    horizontal[2] = transit_z

    completion: list[np.ndarray] = []

    completion.extend(
        _phase_segment(
            current,
            transit_entry,
            maximum_step_m,
            phase,
        )
    )

    completion.extend(
        _phase_segment(
            transit_entry,
            horizontal,
            maximum_step_m,
            phase,
        )
    )

    completion.extend(
        _phase_segment(
            horizontal,
            goal,
            maximum_step_m,
            phase,
        )
    )

    combined = [
        np.asarray(value, dtype=float).copy()
        for value in chunk_targets
    ]

    for point in completion:
        if (
            combined
            and float(
                np.linalg.norm(
                    point - combined[-1]
                )
            )
            <= 1e-9
        ):
            continue

        combined.append(point.copy())

    report["phase_level_execution"] = {
        "enabled": True,
        "fresh_inferences_per_phase": 1,
        "vla_chunk_steps": int(
            config.chunk_steps
        ),
        "vla_chunk_waypoints": len(
            chunk_targets
        ),
        "geometric_completion_waypoints": len(
            completion
        ),
        "completion_route": (
            completion_route
        ),
        "transit_policy": (
            transit_policy
        ),
        "transit_z_m": float(
            transit_z
        ),
        "configured_transit_z_m": float(
            config.transit_z_m
        ),
        "approach_goal_clearance_m": (
            float(
                approach_clearance_m
            )
            if approach_clearance_m
            is not None
            else None
        ),
        "approach_descent_after_chunk_m": (
            float(
                approach_descent_after_chunk_m
            )
            if approach_descent_after_chunk_m
            is not None
            else None
        ),
        "chunk_end_z_m": float(
            current[2]
        ),
        "completion_step_m": (
            maximum_step_m
        ),
        "result_goal_base_m": goal.tolist(),
        "result_distance_to_goal_m": float(
            np.linalg.norm(
                combined[-1] - goal
            )
        ),
        "remote_reinfer_required_before_goal_check": (
            False
        ),
    }

    return combined, report





def send_joint_sequence(
    node: Any,
    q_start_expected: np.ndarray,
    q_targets: list[np.ndarray],
    config: Config,
    label: str,
    duration_s: float | None = None,
) -> dict[str, Any]:
    from action_msgs.msg import GoalStatus
    from control_msgs.action import (
        FollowJointTrajectory,
    )
    from trajectory_msgs.msg import (
        JointTrajectoryPoint,
    )

    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    current_q, current_tcp = (
        current_robot_state(node)
    )

    expected_q = finite_vector(
        q_start_expected,
        6,
        f"{label}.q_start_expected",
    )

    start_error = float(
        np.max(
            np.abs(
                current_q - expected_q
            )
        )
    )

    if start_error > 0.08:
        raise EpisodeStop(
            f"{label}: estado inicial distinto "
            f"del plan: {start_error:.4f} rad"
        )

    compact_targets = []

    previous = current_q.copy()

    for index, value in enumerate(
        q_targets,
        start=1,
    ):
        q_target = finite_vector(
            value,
            6,
            f"{label}.q_target_{index}",
        )

        difference = float(
            np.max(
                np.abs(
                    q_target - previous
                )
            )
        )

        if difference < 1e-7:
            continue

        compact_targets.append(
            q_target.copy()
        )

        previous = q_target.copy()

    if not compact_targets:
        return {
            "label": label,
            "skipped": True,
            "reason": "no_joint_motion",
        }

    normalized_label = str(label).lower()

    if (
        "contact" in normalized_label
        or "place_descent" in normalized_label
    ):
        speed = float(
            os.environ.get(
                "VLA_CONTACT_JOINT_SPEED_RAD_S",
                "0.08",
            )
        )
    elif (
        "lift" in normalized_label
        or "retreat" in normalized_label
    ):
        speed = float(
            os.environ.get(
                "VLA_LIFT_RETREAT_JOINT_SPEED_RAD_S",
                "0.10",
            )
        )
    elif "return" in normalized_label:
        speed = float(
            os.environ.get(
                "VLA_RETURN_JOINT_SPEED_RAD_S",
                "0.14",
            )
        )
    else:
        speed = float(
            os.environ.get(
                "VLA_FREE_JOINT_SPEED_RAD_S",
                "0.16",
            )
        )

    if not 0.05 <= speed <= 0.30:
        raise EpisodeStop(
            f"{label}: velocidad inválida "
            f"{speed:.3f} rad/s"
        )

    minimum_segment_s = float(
        os.environ.get(
            "VLA_MIN_TRAJECTORY_SEGMENT_S",
            "0.50",
        )
    )

    maximum_segment_s = float(
        os.environ.get(
            "VLA_MAX_TRAJECTORY_SEGMENT_S",
            "6.0",
        )
    )

    timeout_scale = float(
        os.environ.get(
            "VLA_TRAJECTORY_TIMEOUT_SCALE",
            "4.0",
        )
    )

    if not 0.25 <= minimum_segment_s <= 2.0:
        raise EpisodeStop(
            "VLA_MIN_TRAJECTORY_SEGMENT_S "
            "fuera de rango"
        )

    if not 2.0 <= maximum_segment_s <= 12.0:
        raise EpisodeStop(
            "VLA_MAX_TRAJECTORY_SEGMENT_S "
            "fuera de rango"
        )

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = (
        joint_names
    )

    previous_q = current_q.copy()
    cumulative_time_s = 0.0
    segment_reports = []

    for index, q_target in enumerate(
        compact_targets,
        start=1,
    ):
        max_dq = float(
            np.max(
                np.abs(
                    q_target - previous_q
                )
            )
        )

        segment_time_s = float(
            np.clip(
                max_dq / speed,
                minimum_segment_s,
                maximum_segment_s,
            )
        )

        cumulative_time_s += (
            segment_time_s
        )

        point = JointTrajectoryPoint()
        point.positions = [
            float(value)
            for value in q_target
        ]

        total_nanoseconds = int(
            round(
                cumulative_time_s
                * 1_000_000_000
            )
        )

        seconds, nanoseconds = divmod(
            total_nanoseconds,
            1_000_000_000,
        )

        point.time_from_start.sec = int(
            seconds
        )

        point.time_from_start.nanosec = int(
            nanoseconds
        )

        goal.trajectory.points.append(
            point
        )

        segment_reports.append(
            {
                "index": index,
                "max_abs_dq_rad": max_dq,
                "segment_time_s": (
                    segment_time_s
                ),
                "time_from_start_s": (
                    cumulative_time_s
                ),
            }
        )

        previous_q = q_target.copy()

    if not node.client.wait_for_server(
        timeout_sec=5.0
    ):
        raise EpisodeStop(
            f"{label}: servidor "
            "FollowJointTrajectory "
            "no disponible"
        )

    print(
        f"label={label} "
        f"points={len(compact_targets)} "
        f"speed={speed:.3f}rad/s "
        f"nominal_duration="
        f"{cumulative_time_s:.3f}s",
        flush=True,
    )

    send_future = (
        node.client.send_goal_async(goal)
    )

    rclpy.spin_until_future_complete(
        node,
        send_future,
        timeout_sec=8.0,
    )

    if not send_future.done():
        raise EpisodeStop(
            f"{label}: timeout enviando "
            "trayectoria multipunto"
        )

    handle = send_future.result()

    if handle is None or not handle.accepted:
        raise EpisodeStop(
            f"{label}: trayectoria "
            "multipunto rechazada"
        )

    result_future = (
        handle.get_result_async()
    )

    result_timeout_s = max(
        30.0,
        cumulative_time_s
        * timeout_scale
        + 15.0,
    )

    rclpy.spin_until_future_complete(
        node,
        result_future,
        timeout_sec=result_timeout_s,
    )

    if not result_future.done():
        cancel_future = (
            handle.cancel_goal_async()
        )

        rclpy.spin_until_future_complete(
            node,
            cancel_future,
            timeout_sec=5.0,
        )

        raise EpisodeStop(
            f"{label}: timeout tras "
            f"{result_timeout_s:.1f}s; "
            "goal cancelado"
        )

    wrapped_result = (
        result_future.result()
    )

    status = int(
        wrapped_result.status
    )

    result = wrapped_result.result
    error_code = int(
        result.error_code
    )

    error_string = str(
        result.error_string
    )

    print(
        f"label={label} "
        f"status={status} "
        f"error_code={error_code} "
        f"message={error_string!r}",
        flush=True,
    )

    if (
        status
        != GoalStatus.STATUS_SUCCEEDED
        or error_code != 0
    ):
        raise EpisodeStop(
            f"{label}: trayectoria no "
            "terminó correctamente: "
            f"status={status}, "
            f"error_code={error_code}, "
            f"message={error_string!r}"
        )

    result_q, result_tcp = (
        current_robot_state(node)
    )

    result_error = float(
        np.max(
            np.abs(
                result_q
                - compact_targets[-1]
            )
        )
    )

    if result_error > 0.05:
        raise EpisodeStop(
            f"{label}: error articular "
            f"result={result_error:.4f} rad"
        )

    return {
        "label": label,
        "execution_mode": (
            "single_multipoint_trajectory"
        ),
        "number_of_points": (
            len(compact_targets)
        ),
        "joint_speed_rad_s": speed,
        "nominal_duration_s": (
            cumulative_time_s
        ),
        "result_timeout_s": (
            result_timeout_s
        ),
        "segments": segment_reports,
        "result_q_rad": result_q.tolist(),
        "result_tcp_m": result_tcp.tolist(),
        "result_joint_error_rad": (
            result_error
        ),
        "resultado": status,
        "result_error_code": error_code,
        "result_error_string": (
            error_string
        ),
        "requested_duration_s": duration_s,
        "requested_duration_used": False,
    }





def interpolate_cartesian_segment(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    max_step_m: float,
    label: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    start = finite_vector(
        start_xyz,
        3,
        f"{label}.start",
    )

    end = finite_vector(
        end_xyz,
        3,
        f"{label}.end",
    )

    if (
        not math.isfinite(max_step_m)
        or max_step_m <= 0.0
    ):
        raise EpisodeStop(
            f"{label}: max_step_m inválido: "
            f"{max_step_m}"
        )

    distance_m = float(
        np.linalg.norm(
            end - start
        )
    )

    number_of_steps = max(
        1,
        int(
            math.ceil(
                distance_m
                / max_step_m
            )
        ),
    )

    targets = [
        start
        + (
            float(index)
            / float(number_of_steps)
        )
        * (end - start)
        for index in range(
            1,
            number_of_steps + 1,
        )
    ]

    previous = start
    step_lengths_m: list[float] = []

    for target in targets:
        step_length_m = float(
            np.linalg.norm(
                target - previous
            )
        )

        if (
            step_length_m
            > max_step_m + 1e-9
        ):
            raise EpisodeStop(
                f"{label}: segmentación inválida: "
                f"{step_length_m:.6f} m > "
                f"{max_step_m:.6f} m"
            )

        step_lengths_m.append(
            step_length_m
        )

        previous = target

    report = {
        "label": label,
        "start_xyz_m": start.tolist(),
        "end_xyz_m": end.tolist(),
        "distance_m": distance_m,
        "max_step_m": max_step_m,
        "number_of_steps": (
            number_of_steps
        ),
        "step_lengths_m": (
            step_lengths_m
        ),
        "targets_xyz_m": [
            target.tolist()
            for target in targets
        ],
    }

    return targets, report




def plan_transport_handoff(
    q_start: np.ndarray,
    tcp_start: np.ndarray,
    entities: dict[str, Any],
    scene: dict[str, Any],
    config: Config,
) -> tuple[
    list[np.ndarray],
    np.ndarray,
    dict[str, Any],
]:
    """
    Busca una transición alcanzable entre el lift local
        y el plano global de transporte.
    
        La búsqueda conserva:
    
        - verticalidad del TCP;
        - continuidad articular;
        - límites del workspace;
        - separación XY conservadora respecto a otros
          objetos detectados.
    
        El handoff es una operación de seguridad cinemática.
        La acción VLA de transporte comienza después de él.
    """

    q_start = finite_vector(
        q_start,
        6,
        "transport_handoff.q_start",
    )

    tcp_start = finite_vector(
        tcp_start,
        3,
        "transport_handoff.tcp_start",
    )

    selected_object = str(
        entities.get(
            "selected_object",
            "",
        )
    )

    objects = scene.get(
        "objects",
        {},
    )

    if not isinstance(
        objects,
        dict,
    ):
        raise EpisodeStop(
            "transport_handoff: scene.objects "
            "no es un diccionario"
        )

    local_z = float(
        tcp_start[2]
    )

    transit_z = float(
        config.transit_z_m
    )

    if (
        not math.isfinite(local_z)
        or not math.isfinite(transit_z)
        or transit_z <= local_z + 0.020
    ):
        raise EpisodeStop(
            "transport_handoff: alturas "
            f"inválidas local={local_z:.4f}, "
            f"transit={transit_z:.4f}"
        )

    retreat_step_m = float(
        os.environ.get(
            "VLA_HANDOFF_RETREAT_STEP_M",
            "0.020",
        )
    )

    maximum_retreat_m = float(
        os.environ.get(
            "VLA_HANDOFF_MAX_RETREAT_M",
            "0.180",
        )
    )

    cartesian_step_m = float(
        os.environ.get(
            "VLA_HANDOFF_CARTESIAN_STEP_M",
            "0.012",
        )
    )

    held_envelope_radius_m = float(
        os.environ.get(
            "VLA_HANDOFF_HELD_ENVELOPE_RADIUS_M",
            "0.055",
        )
    )

    obstacle_margin_m = float(
        os.environ.get(
            "VLA_HANDOFF_OBSTACLE_MARGIN_M",
            "0.015",
        )
    )

    maximum_candidate_tests = int(os.environ.get(
        "VLA_HANDOFF_MAX_CANDIDATE_TESTS", "18"
    ))

    if not (
        0.010
        <= retreat_step_m
        <= 0.040
    ):
        raise EpisodeStop(
            "VLA_HANDOFF_RETREAT_STEP_M "
            "debe estar entre 0.010 y 0.040 m"
        )

    if not (
        0.040
        <= maximum_retreat_m
        <= 0.250
    ):
        raise EpisodeStop(
            "VLA_HANDOFF_MAX_RETREAT_M "
            "debe estar entre 0.040 y 0.250 m"
        )

    if not (
        0.005
        <= cartesian_step_m
        <= 0.020
    ):
        raise EpisodeStop(
            "VLA_HANDOFF_CARTESIAN_STEP_M "
            "debe estar entre 0.005 y 0.020 m"
        )

    if not (
        0.030
        <= held_envelope_radius_m
        <= 0.100
    ):
        raise EpisodeStop(
            "VLA_HANDOFF_HELD_ENVELOPE_RADIUS_M "
            "debe estar entre 0.030 y 0.100 m"
        )

    if not (
        0.005
        <= obstacle_margin_m
        <= 0.060
    ):
        raise EpisodeStop(
            "VLA_HANDOFF_OBSTACLE_MARGIN_M "
            "debe estar entre 0.005 y 0.060 m"
        )

    if not 4 <= maximum_candidate_tests <= 30:
        raise EpisodeStop("VLA_HANDOFF_MAX_CANDIDATE_TESTS debe estar entre 4 y 30")

    def record_radius(
        record: dict[str, Any],
    ) -> float:
        candidates = []

        for key in (
            "radius_m",
            "radius_p90_m",
            "radius_median_m",
        ):
            value = record.get(key)

            if isinstance(
                value,
                (int, float),
            ):
                value = float(value)

                if (
                    math.isfinite(value)
                    and value > 0.0
                ):
                    candidates.append(
                        value
                    )

        return max(
            candidates,
            default=0.030,
        )

    def segment_point_distance_xy(
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        direction = end - start

        denominator = float(
            np.dot(
                direction,
                direction,
            )
        )

        if denominator <= 1e-12:
            return float(
                np.linalg.norm(
                    point - start
                )
            )

        fraction = float(
            np.clip(
                np.dot(
                    point - start,
                    direction,
                )
                / denominator,
                0.0,
                1.0,
            )
        )

        closest = (
            start
            + fraction * direction
        )

        return float(
            np.linalg.norm(
                point - closest
            )
        )

    selected_record = objects.get(
        selected_object,
        {},
    )

    if not isinstance(
        selected_record,
        dict,
    ):
        selected_record = {}

    held_object_radius_m = record_radius(
        selected_record
    )

    held_clearance_radius_m = max(
        held_envelope_radius_m,
        held_object_radius_m,
    )

    obstacles = []

    for key, record in objects.items():
        if not isinstance(record, dict):
            continue

        if str(key) == selected_object:
            continue

        try:
            obstacle_xyz = xyz_from_record(
                record,
                f"transport_handoff.objects.{key}",
            )
        except Exception as error:
            raise EpisodeStop(
                "transport_handoff: objeto "
                f"{key!r} sin geometría válida: "
                f"{error}"
            ) from error

        if (
            float(
                np.linalg.norm(
                    obstacle_xyz[:2]
                    - tcp_start[:2]
                )
            )
            <= 0.020
        ):
            # Alias geométrico del objeto seleccionado.
            continue

        obstacles.append(
            {
                "key": str(key),
                "xyz_m": (
                    obstacle_xyz.copy()
                ),
                "radius_m": record_radius(
                    record
                ),
            }
        )

    radial_norm = float(
        np.linalg.norm(
            tcp_start[:2]
        )
    )

    if radial_norm > 1e-9:
        direction_to_base = (
            -tcp_start[:2]
            / radial_norm
        )
    else:
        target_direction = (
            finite_vector(
                entities[
                    "target_safe_xyz_m"
                ],
                3,
                "transport_handoff.target",
            )[:2]
            - tcp_start[:2]
        )

        target_norm = float(
            np.linalg.norm(
                target_direction
            )
        )

        if target_norm <= 1e-9:
            raise EpisodeStop(
                "transport_handoff: no se puede "
                "determinar dirección de retirada"
            )

        direction_to_base = (
            target_direction
            / target_norm
        )

    angle_offsets_deg = [
        0.0,
        25.0,
        -25.0,
        45.0,
        -45.0,
        65.0,
        -65.0,
    ]

    offset_values = [
        0.0
    ]

    number_of_offsets = int(
        math.floor(
            maximum_retreat_m
            / retreat_step_m
        )
    )

    for index in range(
        1,
        number_of_offsets + 1,
    ):
        offset_values.append(
            index * retreat_step_m
        )

    candidates = []

    for offset_m in offset_values:
        angles = (
            [0.0]
            if offset_m <= 1e-9
            else angle_offsets_deg
        )

        for angle_deg in angles:
            angle_rad = math.radians(
                angle_deg
            )

            cosine = math.cos(
                angle_rad
            )

            sine = math.sin(
                angle_rad
            )

            direction = np.asarray(
                [
                    cosine
                    * direction_to_base[0]
                    - sine
                    * direction_to_base[1],
                    sine
                    * direction_to_base[0]
                    + cosine
                    * direction_to_base[1],
                ],
                dtype=float,
            )

            local_target = (
                tcp_start.copy()
            )

            local_target[:2] = (
                tcp_start[:2]
                + offset_m * direction
            )

            rise_fraction = 0.0 if offset_m <= 1e-9 else min(
                0.88, 0.30 + 0.58 * offset_m / maximum_retreat_m
            )
            local_target[2] = local_z + rise_fraction * (transit_z - local_z)

            handoff_target = (
                local_target.copy()
            )

            handoff_target[2] = (
                transit_z
            )

            candidates.append(
                {
                    "offset_m": float(
                        offset_m
                    ),
                    "angle_deg": float(
                        angle_deg
                    ),
                    "rise_fraction": float(rise_fraction),
                    "local_target": (
                        local_target
                    ),
                    "handoff_target": (
                        handoff_target
                    ),
                }
            )

    rejected = []

    for candidate_index, candidate in enumerate(
        candidates,
        start=1,
    ):
        if candidate_index > maximum_candidate_tests:
            break
        local_target = candidate[
            "local_target"
        ]

        handoff_target = candidate[
            "handoff_target"
        ]

        if not workspace_ok(
            local_target
        ):
            rejected.append(
                {
                    "offset_m": candidate[
                        "offset_m"
                    ],
                    "angle_deg": candidate[
                        "angle_deg"
                    ],
                    "reason": (
                        "local_target_outside_workspace"
                    ),
                }
            )
            continue

        if not workspace_ok(
            handoff_target
        ):
            rejected.append(
                {
                    "offset_m": candidate[
                        "offset_m"
                    ],
                    "angle_deg": candidate[
                        "angle_deg"
                    ],
                    "reason": (
                        "handoff_target_outside_workspace"
                    ),
                }
            )
            continue

        minimum_obstacle_clearance_m = (
            float("inf")
        )

        limiting_obstacle = None
        collision_rejected = False

        for obstacle in obstacles:
            distance_m = (
                segment_point_distance_xy(
                    obstacle[
                        "xyz_m"
                    ][:2],
                    tcp_start[:2],
                    local_target[:2],
                )
            )

            required_m = (
                held_clearance_radius_m
                + float(
                    obstacle[
                        "radius_m"
                    ]
                )
                + obstacle_margin_m
            )

            minimum_obstacle_clearance_m = min(
                minimum_obstacle_clearance_m,
                distance_m,
            )

            if distance_m < required_m:
                limiting_obstacle = {
                    "key": obstacle["key"],
                    "distance_m": (
                        distance_m
                    ),
                    "required_m": (
                        required_m
                    ),
                }

                collision_rejected = True
                break

        if collision_rejected:
            rejected.append(
                {
                    "offset_m": candidate[
                        "offset_m"
                    ],
                    "angle_deg": candidate[
                        "angle_deg"
                    ],
                    "reason": (
                        "detected_object_clearance"
                    ),
                    "limiting_obstacle": (
                        limiting_obstacle
                    ),
                }
            )
            continue

        horizontal_targets = (
            _phase_segment(
                tcp_start,
                local_target,
                cartesian_step_m,
                (
                    "transport_handoff_"
                    f"candidate_{candidate_index}_"
                    "horizontal"
                ),
            )
        )

        vertical_targets = (
            _phase_segment(
                local_target,
                handoff_target,
                cartesian_step_m,
                (
                    "transport_handoff_"
                    f"candidate_{candidate_index}_"
                    "vertical"
                ),
            )
        )

        targets = (
            horizontal_targets
            + vertical_targets
        )

        if not targets:
            targets = [
                handoff_target.copy()
            ]

        try:
            (
                q_targets,
                ik_report,
            ) = solve_cartesian_targets(
                q_start=q_start,
                tcp_start=tcp_start,
                targets=targets,
                config=config,
                label=(
                    "transport_handoff_"
                    f"candidate_{candidate_index}"
                ),
                max_joint_step_rad=(
                    config.
                    max_chunk_joint_step_rad
                ),
                contact=False,
            )

        except EpisodeStop as error:
            rejected.append(
                {
                    "offset_m": candidate[
                        "offset_m"
                    ],
                    "angle_deg": candidate[
                        "angle_deg"
                    ],
                    "reason": "ik",
                    "detail": str(error),
                }
            )
            continue

        if not q_targets:
            rejected.append(
                {
                    "offset_m": candidate[
                        "offset_m"
                    ],
                    "angle_deg": candidate[
                        "angle_deg"
                    ],
                    "reason": (
                        "empty_joint_chain"
                    ),
                }
            )
            continue

        handoff_tcp = finite_vector(
            ik_report[
                "waypoints"
            ][-1][
                "fk_real_base_m"
            ],
            3,
            "transport_handoff.fk_end",
        )

        endpoint_error_m = float(
            np.linalg.norm(
                handoff_tcp
                - handoff_target
            )
        )

        if (
            endpoint_error_m
            > config.max_fk_error_m
        ):
            rejected.append(
                {
                    "offset_m": candidate[
                        "offset_m"
                    ],
                    "angle_deg": candidate[
                        "angle_deg"
                    ],
                    "reason": (
                        "handoff_endpoint_error"
                    ),
                    "endpoint_error_m": (
                        endpoint_error_m
                    ),
                }
            )
            continue

        report = {
            "policy": (
                "bounded_diagonal_then_"
                "vertical_transit"
            ),
            "selected_object": (
                selected_object
            ),
            "local_lift_xyz_m": (
                tcp_start.tolist()
            ),
            "selected_local_retreat_xyz_m": (
                local_target.tolist()
            ),
            "selected_handoff_xyz_m": (
                handoff_target.tolist()
            ),
            "handoff_fk_xyz_m": (
                handoff_tcp.tolist()
            ),
            "handoff_endpoint_error_m": (
                endpoint_error_m
            ),
            "selected_offset_m": (
                candidate["offset_m"]
            ),
            "selected_angle_deg": (
                candidate["angle_deg"]
            ),
            "minimum_obstacle_clearance_m": (
                minimum_obstacle_clearance_m
                if math.isfinite(
                    minimum_obstacle_clearance_m
                )
                else None
            ),
            "held_clearance_radius_m": (
                held_clearance_radius_m
            ),
            "obstacle_margin_m": (
                obstacle_margin_m
            ),
            "detected_obstacles": [
                {
                    "key": obstacle["key"],
                    "xyz_m": (
                        obstacle[
                            "xyz_m"
                        ].tolist()
                    ),
                    "radius_m": (
                        obstacle[
                            "radius_m"
                        ]
                    ),
                }
                for obstacle in obstacles
            ],
            "rejected_candidates": (
                rejected
            ),
            "ik": ik_report,
            "all_steps_ok": True,
        }

        print(
            f"offset="
            f"{candidate['offset_m']:.3f}m "
            f"angle="
            f"{candidate['angle_deg']:.1f}deg "
            f"rise={candidate['rise_fraction']:.3f} "
            f"tested={candidate_index}/{maximum_candidate_tests} "
            f"local="
            f"{local_target.tolist()} "
            f"handoff="
            f"{handoff_target.tolist()} "
            f"obstacles={len(obstacles)} "
            f"rejected={len(rejected)}",
            flush=True,
        )

        return (
            [
                value.copy()
                for value in q_targets
            ],
            handoff_tcp.copy(),
            report,
        )

    raise EpisodeStop(
        "transport_handoff: no existe una "
        "retirada local que permita alcanzar "
        "el plano de transporte; "
        f"tested={min(len(candidates), maximum_candidate_tests)}, "
        f"generated={len(candidates)}, "
        f"rejected={rejected[:20]}"
    )




def plan_contact_sequence(
    q_start: np.ndarray,
    tcp_start: np.ndarray,
    entities: dict[str, Any],
    config: Config,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """
    Entrada cartesiana determinista:
    
        tcp actual
          -> object_safe
          -> pre_pick
          -> pick
          -> lift
    
        No se aplica orientación VLA durante contacto.
    """

    q_start = finite_vector(
        q_start,
        6,
        "contact.q_start",
    )

    tcp_start = finite_vector(
        tcp_start,
        3,
        "contact.tcp_start",
    )

    object_xyz = finite_vector(
        entities["object_xyz_m"],
        3,
        "contact.object_xyz",
    )

    object_safe_xyz = finite_vector(
        entities["object_safe_xyz_m"],
        3,
        "contact.object_safe",
    )

    pre_pick_xyz = finite_vector(
        entities["pre_pick_xyz_m"],
        3,
        "contact.pre_pick",
    )

    pick_xyz = finite_vector(
        entities["pick_xyz_m"],
        3,
        "contact.pick",
    )

    lift_xyz = finite_vector(
        entities["lift_xyz_m"],
        3,
        "contact.lift",
    )

    entry_xy_error_m = float(
        np.linalg.norm(
            tcp_start[:2]
            - object_safe_xyz[:2]
        )
    )

    if entry_xy_error_m > 0.100:
        raise EpisodeStop(
            "contact: approach demasiado lejos: "
            f"{entry_xy_error_m:.4f}m"
        )

    def segment_nonempty(
        start_xyz,
        end_xyz,
        max_step_m,
        label,
    ):
        start_xyz = finite_vector(
            start_xyz,
            3,
            label + ".start",
        )

        end_xyz = finite_vector(
            end_xyz,
            3,
            label + ".end",
        )

        distance_m = float(
            np.linalg.norm(
                end_xyz - start_xyz
            )
        )

        if distance_m <= 1e-9:
            return [
                end_xyz.copy()
            ], {
                "label": label,
                "distance_m": 0.0,
                "max_step_m": max_step_m,
                "number_of_steps": 1,
                "targets_xyz_m": [
                    end_xyz.tolist()
                ],
            }

        return interpolate_cartesian_segment(
            start_xyz=start_xyz,
            end_xyz=end_xyz,
            max_step_m=max_step_m,
            label=label,
        )

    (
        safe_targets,
        safe_report,
    ) = segment_nonempty(
        start_xyz=tcp_start,
        end_xyz=object_safe_xyz,
        max_step_m=0.010,
        label="contact_to_object_safe",
    )

    (
        pre_pick_targets,
        pre_pick_report,
    ) = segment_nonempty(
        start_xyz=object_safe_xyz,
        end_xyz=pre_pick_xyz,
        max_step_m=0.010,
        label="contact_to_pre_pick",
    )

    (
        pick_targets,
        pick_report,
    ) = segment_nonempty(
        start_xyz=pre_pick_xyz,
        end_xyz=pick_xyz,
        max_step_m=0.010,
        label="contact_to_pick",
    )

    all_targets = (
        safe_targets
        + pre_pick_targets
        + pick_targets
    )

    object_safe_index = (
        len(safe_targets) - 1
    )

    pre_pick_index = (
        len(safe_targets)
        + len(pre_pick_targets)
        - 1
    )

    pick_index = (
        len(all_targets) - 1
    )

    (
        q_to_pick_steps,
        to_pick_report,
    ) = solve_cartesian_targets(
        q_start=q_start,
        tcp_start=tcp_start,
        targets=all_targets,
        config=config,
        label="contact_simple_pick",
        max_joint_step_rad=(
            config.max_contact_joint_step_rad
        ),
        contact=True,
    )

    q_object_safe = (
        q_to_pick_steps[
            object_safe_index
        ].copy()
    )

    q_pre_pick = (
        q_to_pick_steps[
            pre_pick_index
        ].copy()
    )

    q_pick = (
        q_to_pick_steps[
            pick_index
        ].copy()
    )

    (
        lift_targets,
        lift_segmentation_report,
    ) = segment_nonempty(
        start_xyz=pick_xyz,
        end_xyz=lift_xyz,
        max_step_m=0.012,
        label="contact_simple_lift",
    )

    (
        q_lift_steps,
        lift_report,
    ) = solve_cartesian_targets(
        q_start=q_pick,
        tcp_start=pick_xyz,
        targets=lift_targets,
        config=config,
        label="contact_simple_lift",
        max_joint_step_rad=(
            config.max_contact_joint_step_rad
        ),
        contact=True,
    )

    print(
        f"entry_xy_error="
        f"{entry_xy_error_m:.4f}m "
        f"pick_points="
        f"{len(q_to_pick_steps)} "
        f"lift_points="
        f"{len(q_lift_steps)}",
        flush=True,
    )

    result = {
        "q_to_pick_steps": (
            q_to_pick_steps
        ),
        "q_object_safe": (
            q_object_safe
        ),
        "q_pre_pick": q_pre_pick,
        "q_pick": q_pick,
        "q_lift_steps": (
            q_lift_steps
        ),
        "q_lift": (
            q_lift_steps[-1].copy()
        ),
    }

    report = {
        "architecture": (
            "supervised_vla_"
            "geometric_contact"
        ),
        "route_type": (
            "direct_segmented_contact"
        ),
        "vla_rotation_applied": False,
        "contact_entry_xy_error_m": (
            entry_xy_error_m
        ),
        "to_pick": to_pick_report,
        "lift": lift_report,
        "segmentation": {
            "to_object_safe": safe_report,
            "to_pre_pick": pre_pick_report,
            "to_pick": pick_report,
            "to_lift": (
                lift_segmentation_report
            ),
        },
        "endpoint_indices": {
            "object_safe": (
                object_safe_index
            ),
            "pre_pick": pre_pick_index,
            "pick": pick_index,
        },
        "named_targets": {
            "tcp_start_xyz_m": (
                tcp_start.tolist()
            ),
            "object_xyz_m": (
                object_xyz.tolist()
            ),
            "object_safe_xyz_m": (
                object_safe_xyz.tolist()
            ),
            "pre_pick_xyz_m": (
                pre_pick_xyz.tolist()
            ),
            "pick_xyz_m": (
                pick_xyz.tolist()
            ),
            "lift_xyz_m": (
                lift_xyz.tolist()
            ),
        },
    }

    return result, report






def _plan_place_sequence_fixed(
    q_start: np.ndarray,
    tcp_start: np.ndarray,
    entities: dict[str, Any],
    config: Config,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    q_start = finite_vector(
        q_start,
        6,
        "place.q_start",
    )

    tcp_start = finite_vector(
        tcp_start,
        3,
        "place.tcp_start",
    )

    pre_place_xyz = finite_vector(
        entities["pre_place_xyz_m"],
        3,
        "place.pre_place",
    )

    place_xyz = finite_vector(
        entities["place_xyz_m"],
        3,
        "place.place",
    )

    transit_z_m = float(
        entities.get(
            "place_transit_z_m",
            max(
                float(config.transit_z_m),
                float(tcp_start[2]),
                float(pre_place_xyz[2]),
            ),
        )
    )

    def segment(
        start_xyz: np.ndarray,
        end_xyz: np.ndarray,
        max_step_m: float,
        label: str,
    ) -> tuple[
        list[np.ndarray],
        dict[str, Any],
    ]:
        start_xyz = finite_vector(
            start_xyz,
            3,
            f"{label}.start",
        )

        end_xyz = finite_vector(
            end_xyz,
            3,
            f"{label}.end",
        )

        distance_m = float(
            np.linalg.norm(
                end_xyz - start_xyz
            )
        )

        if distance_m <= 1e-9:
            return [], {
                "label": label,
                "distance_m": 0.0,
                "max_step_m": max_step_m,
                "number_of_steps": 0,
                "targets_xyz_m": [],
            }

        return interpolate_cartesian_segment(
            start_xyz=start_xyz,
            end_xyz=end_xyz,
            max_step_m=max_step_m,
            label=label,
        )

    start_transit_xyz = (
        tcp_start.copy()
    )

    start_transit_xyz[2] = (
        transit_z_m
    )

    target_transit_xyz = (
        pre_place_xyz.copy()
    )

    target_transit_xyz[2] = (
        transit_z_m
    )

    (
        raise_targets,
        raise_report,
    ) = segment(
        start_xyz=tcp_start,
        end_xyz=start_transit_xyz,
        max_step_m=0.015,
        label="place_raise_to_transit",
    )

    (
        horizontal_targets,
        horizontal_report,
    ) = segment(
        start_xyz=start_transit_xyz,
        end_xyz=target_transit_xyz,
        max_step_m=0.015,
        label="place_horizontal_to_zone",
    )

    (
        pre_place_targets,
        pre_place_report,
    ) = segment(
        start_xyz=target_transit_xyz,
        end_xyz=pre_place_xyz,
        max_step_m=0.015,
        label="place_descend_to_pre_place",
    )

    (
        release_targets,
        release_report,
    ) = segment(
        start_xyz=pre_place_xyz,
        end_xyz=place_xyz,
        max_step_m=0.012,
        label="place_descend_to_release",
    )

    to_place_targets = (
        raise_targets
        + horizontal_targets
        + pre_place_targets
        + release_targets
    )

    if not to_place_targets:
        raise EpisodeStop(
            "place: no se generaron waypoints"
        )

    pre_place_index = (
        len(raise_targets)
        + len(horizontal_targets)
        + len(pre_place_targets)
        - 1
    )

    if not (
        0 <= pre_place_index
        < len(to_place_targets)
    ):
        raise EpisodeStop(
            "place: índice pre_place inválido"
        )

    (
        q_to_place_steps,
        to_place_report,
    ) = solve_cartesian_targets(
        q_start=q_start,
        tcp_start=tcp_start,
        targets=to_place_targets,
        config=config,
        label="place_segmented_descent",
        max_joint_step_rad=(
            config.max_contact_joint_step_rad
        ),
        contact=True,
    )

    q_pre_place = (
        q_to_place_steps[
            pre_place_index
        ]
    )

    q_place = (
        q_to_place_steps[-1]
    )

    place_radius_xy_m = float(
        np.linalg.norm(
            place_xyz[:2]
        )
    )

    if place_radius_xy_m <= 1e-6:
        raise EpisodeStop(
            "place: radio XY inválido"
        )

    retreat_inset_m = float(
        os.environ.get(
            "VLA_PLACE_RETREAT_INSET_M",
            "0.080",
        )
    )

    if (
        not math.isfinite(
            retreat_inset_m
        )
        or retreat_inset_m < 0.040
        or retreat_inset_m > 0.120
    ):
        raise EpisodeStop(
            "VLA_PLACE_RETREAT_INSET_M debe "
            "estar entre 0.040 y 0.120 m"
        )

    clearance_z_m = float(
        min(
            transit_z_m - 0.030,
            max(
                place_xyz[2] + 0.180,
                0.240,
            ),
        )
    )

    if clearance_z_m <= place_xyz[2]:
        raise EpisodeStop(
            "place: clearance_z inválida"
        )

    edge_clearance_xyz = (
        place_xyz.copy()
    )

    edge_clearance_xyz[2] = (
        clearance_z_m
    )

    radial_direction_xy = (
        place_xyz[:2]
        / place_radius_xy_m
    )

    inward_clearance_xyz = (
        edge_clearance_xyz.copy()
    )

    inward_clearance_xyz[:2] = (
        place_xyz[:2]
        - retreat_inset_m
        * radial_direction_xy
    )

    inward_transit_xyz = (
        inward_clearance_xyz.copy()
    )

    inward_transit_xyz[2] = (
        transit_z_m
    )

    (
        clearance_targets,
        clearance_report,
    ) = segment(
        start_xyz=place_xyz,
        end_xyz=edge_clearance_xyz,
        max_step_m=0.012,
        label=(
            "place_retreat_vertical_clearance"
        ),
    )

    (
        inward_targets,
        inward_report,
    ) = segment(
        start_xyz=edge_clearance_xyz,
        end_xyz=inward_clearance_xyz,
        max_step_m=0.012,
        label=(
            "place_retreat_radial_inward"
        ),
    )

    (
        transit_targets,
        transit_report,
    ) = segment(
        start_xyz=inward_clearance_xyz,
        end_xyz=inward_transit_xyz,
        max_step_m=0.012,
        label=(
            "place_retreat_rise_to_transit"
        ),
    )

    retreat_targets = (
        clearance_targets
        + inward_targets
        + transit_targets
    )

    if not retreat_targets:
        raise EpisodeStop(
            "place: no se generó retirada dogleg"
        )

    (
        q_retreat_steps,
        retreat_report,
    ) = solve_cartesian_targets(
        q_start=q_place,
        tcp_start=place_xyz,
        targets=retreat_targets,
        config=config,
        label="place_dogleg_retreat",
        max_joint_step_rad=(
            config.max_contact_joint_step_rad
        ),
        contact=True,
    )

    result = {
        "q_to_place_steps": (
            q_to_place_steps
        ),
        "q_pre_place": q_pre_place,
        "q_place": q_place,
        "q_retreat_steps": (
            q_retreat_steps
        ),
        "q_retreat": (
            q_retreat_steps[-1]
        ),
    }

    report = {
        "route_type": (
            "high_transit_place_"
            "vertical_inward_rise_retreat"
        ),
        "transit_z_m": transit_z_m,
        "retreat_inset_m": (
            retreat_inset_m
        ),
        "to_place": to_place_report,
        "retreat": retreat_report,
        "segmentation": {
            "raise_to_transit": (
                raise_report
            ),
            "horizontal_to_zone": (
                horizontal_report
            ),
            "descend_to_pre_place": (
                pre_place_report
            ),
            "descend_to_release": (
                release_report
            ),
            "retreat_vertical_clearance": (
                clearance_report
            ),
            "retreat_radial_inward": (
                inward_report
            ),
            "retreat_rise_to_transit": (
                transit_report
            ),
        },
        "named_targets": {
            "pre_place_xyz_m": (
                pre_place_xyz.tolist()
            ),
            "place_xyz_m": (
                place_xyz.tolist()
            ),
            "edge_clearance_xyz_m": (
                edge_clearance_xyz.tolist()
            ),
            "inward_clearance_xyz_m": (
                inward_clearance_xyz.tolist()
            ),
            "inward_transit_xyz_m": (
                inward_transit_xyz.tolist()
            ),
        },
    }

    return result, report


def lock_safe_place_target(
    entities: dict[str, Any],
    target_xyz: np.ndarray,
    config: Config,
    *,
    projection_report: dict[str, Any],
    preferred_xy: np.ndarray,
) -> dict[str, Any]:
    target = finite_vector(
        target_xyz,
        3,
        "safe_place.target_xyz",
    )

    preferred = finite_vector(
        preferred_xy,
        2,
        "safe_place.preferred_xy",
    )

    result = dict(entities)

    target_safe = target.copy()
    target_safe[2] = float(
        config.transit_z_m
    )

    place = target.copy()
    place[2] = float(
        config.place_z_m
    )

    result["target_xyz_m"] = target.copy()
    result["target_safe_xyz_m"] = (
        target_safe.copy()
    )
    result["pre_place_xyz_m"] = (
        target_safe.copy()
    )
    result["place_xyz_m"] = place.copy()
    result["retreat_xyz_m"] = (
        target_safe.copy()
    )

    result["vla_preferred_xy_m"] = (
        preferred.copy()
    )

    result["safe_place_projection"] = dict(
        projection_report
    )

    result["safe_place_target_locked"] = True

    return result


def project_vla_preference_to_safe_target(
    entities: dict[str, Any],
    preferred_xy: np.ndarray,
    config: Config,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    preferred = finite_vector(
        preferred_xy,
        2,
        "safe_place.preferred_xy",
    )

    center = finite_vector(
        entities[
            "target_detected_center_xyz_m"
        ],
        3,
        "safe_place.target_center",
    )

    shape_force_target_center = (
        os.environ.get(
            "VLA_TASK_FAMILY",
            "",
        ).strip()
        == "shape_classification"
        and os.environ.get(
            "VLA_SHAPE_FORCE_TARGET_CENTER",
            "NO",
        ).strip().upper()
        in {"1", "YES", "TRUE", "ON"}
    )

    if shape_force_target_center:
        preferred = center[:2].copy()

        print(
            f"preferred_xy={preferred.tolist()}",
            flush=True,
        )

    original_target = finite_vector(
        entities["target_xyz_m"],
        3,
        "safe_place.original_target",
    )

    selected_radius_m = float(
        entities.get(
            "selected_object_radius_m",
            0.030,
        )
    )

    zone_radius_m = float(
        os.environ.get(
            "VLA_SAFE_PLACE_ZONE_RADIUS_M",
            "0.110",
        )
    )

    border_margin_m = float(
        os.environ.get(
            "VLA_SAFE_PLACE_BORDER_MARGIN_M",
            "0.010",
        )
    )

    occupancy_margin_m = float(
        os.environ.get(
            "VLA_SAFE_PLACE_OCCUPANCY_MARGIN_M",
            "0.018",
        )
    )

    empirical_uncertainty_m = float(
        os.environ.get(
            "VLA_SAFE_PLACE_EMPIRICAL_UNCERTAINTY_M",
            "0.015",
        )
    )

    grid_step_m = float(
        os.environ.get(
            "VLA_SAFE_PLACE_GRID_STEP_M",
            "0.005",
        )
    )

    values = {
        "selected_radius_m":
            selected_radius_m,
        "zone_radius_m":
            zone_radius_m,
        "border_margin_m":
            border_margin_m,
        "occupancy_margin_m":
            occupancy_margin_m,
        "empirical_uncertainty_m":
            empirical_uncertainty_m,
        "grid_step_m":
            grid_step_m,
    }

    for name, value in values.items():
        if (
            not math.isfinite(value)
            or value <= 0.0
        ):
            raise EpisodeStop(
                f"safe_place: {name} inválido: "
                f"{value}"
            )

    required_surface_clearance_m = (
        occupancy_margin_m
        + empirical_uncertainty_m
    )

    usable_radius_m = (
        zone_radius_m
        - selected_radius_m
        - border_margin_m
        - empirical_uncertainty_m
    )

    if usable_radius_m <= 0.0:
        raise EpisodeStop(
            "safe_place: la zona no tiene "
            "radio útil para el objeto"
        )

    obstacles = entities.get(
        "place_obstacles",
        [],
    )

    if not isinstance(obstacles, list):
        raise EpisodeStop(
            "safe_place: place_obstacles "
            "no es una lista"
        )

    if shape_force_target_center:
        self_obstacle_radius_m = float(
            os.environ.get(
                "VLA_SHAPE_SELF_OBSTACLE_RADIUS_M",
                "0.060",
            )
        )

        filtered_obstacles = []
        removed_obstacles = []

        for obstacle in obstacles:
            obstacle_key = str(
                obstacle.get(
                    "key",
                    "unknown",
                )
            )

            try:
                obstacle_xyz = finite_vector(
                    obstacle["center_xyz_m"],
                    3,
                    (
                        "shape_safe_place.obstacle."
                        f"{obstacle_key}"
                    ),
                )

                target_distance_xy_m = float(
                    np.linalg.norm(
                        obstacle_xyz[:2]
                        - center[:2]
                    )
                )
            except Exception:
                filtered_obstacles.append(obstacle)
                continue

            belongs_to_selected_container = (
                obstacle_key.startswith(
                    "local_ply_obstacle_"
                )
                and target_distance_xy_m
                <= self_obstacle_radius_m
            )

            if belongs_to_selected_container:
                removed_obstacles.append(
                    {
                        "key": obstacle_key,
                        "target_distance_xy_m":
                            target_distance_xy_m,
                    }
                )
                continue

            filtered_obstacles.append(obstacle)

        obstacles = filtered_obstacles

        print(
            f"removed={removed_obstacles} "
            f"remaining={len(obstacles)} "
            f"radius_m={self_obstacle_radius_m:.3f}",
            flush=True,
        )

    x_values = np.arange(
        center[0] - usable_radius_m,
        center[0] + usable_radius_m
        + 0.5 * grid_step_m,
        grid_step_m,
        dtype=float,
    )

    y_values = np.arange(
        center[1] - usable_radius_m,
        center[1] + usable_radius_m
        + 0.5 * grid_step_m,
        grid_step_m,
        dtype=float,
    )

    safe_candidates = []
    interior_candidate_count = 0

    for x_value in x_values:
        for y_value in y_values:
            candidate_xy = np.array(
                [
                    float(x_value),
                    float(y_value),
                ],
                dtype=float,
            )

            center_offset = float(
                np.linalg.norm(
                    candidate_xy
                    - center[:2]
                )
            )

            if (
                center_offset
                > usable_radius_m + 1e-9
            ):
                continue

            interior_candidate_count += 1

            minimum_surface_clearance = (
                math.inf
            )

            nearest_obstacle = None
            safe = True

            for obstacle in obstacles:
                obstacle_xyz = finite_vector(
                    obstacle[
                        "center_xyz_m"
                    ],
                    3,
                    (
                        "safe_place.obstacle."
                        f"{obstacle.get('key')}"
                    ),
                )

                obstacle_radius_m = float(
                    obstacle.get(
                        "radius_m",
                        0.030,
                    )
                )

                center_distance = float(
                    np.linalg.norm(
                        candidate_xy
                        - obstacle_xyz[:2]
                    )
                )

                surface_clearance = (
                    center_distance
                    - selected_radius_m
                    - obstacle_radius_m
                )

                if (
                    surface_clearance
                    < minimum_surface_clearance
                ):
                    minimum_surface_clearance = (
                        surface_clearance
                    )
                    nearest_obstacle = str(
                        obstacle.get(
                            "key",
                            "unknown",
                        )
                    )

                if (
                    surface_clearance
                    + 1e-9
                    < required_surface_clearance_m
                ):
                    safe = False
                    break

            if not safe:
                continue

            preference_distance = float(
                np.linalg.norm(
                    candidate_xy
                    - preferred
                )
            )

            safe_candidates.append(
                {
                    "xy_m":
                        candidate_xy,
                    "preference_distance_m":
                        preference_distance,
                    "center_offset_m":
                        center_offset,
                    "minimum_surface_clearance_m":
                        (
                            None
                            if not math.isfinite(
                                minimum_surface_clearance
                            )
                            else float(
                                minimum_surface_clearance
                            )
                        ),
                    "nearest_obstacle":
                        nearest_obstacle,
                }
            )

    if not safe_candidates:
        raise EpisodeStop(
            "safe_place: no existe ninguna "
            "colocación con 33 mm de "
            "separación superficial"
        )

    selected = min(
        safe_candidates,
        key=lambda item: (
            item[
                "preference_distance_m"
            ],
            item[
                "center_offset_m"
            ],
        ),
    )

    selected_target = (
        original_target.copy()
    )

    selected_target[:2] = selected[
        "xy_m"
    ]

    report = {
        "schema":
            "vla_safe_place_projection",
        "policy":
            "nearest_safe_grid_point_to_"
            "vla_preference",
        "target_center_xyz_m":
            center.tolist(),
        "vla_preferred_xy_m":
            preferred.tolist(),
        "selected_target_xyz_m":
            selected_target.tolist(),
        "selected_object_radius_m":
            selected_radius_m,
        "zone_radius_m":
            zone_radius_m,
        "usable_radius_m":
            usable_radius_m,
        "border_margin_m":
            border_margin_m,
        "occupancy_margin_m":
            occupancy_margin_m,
        "empirical_uncertainty_m":
            empirical_uncertainty_m,
        "required_surface_clearance_m":
            required_surface_clearance_m,
        "grid_step_m":
            grid_step_m,
        "interior_candidate_count":
            interior_candidate_count,
        "safe_candidate_count":
            len(safe_candidates),
        "selected_preference_distance_m":
            selected[
                "preference_distance_m"
            ],
        "selected_center_offset_m":
            selected[
                "center_offset_m"
            ],
        "selected_minimum_surface_clearance_m":
            selected[
                "minimum_surface_clearance_m"
            ],
        "selected_nearest_obstacle":
            selected[
                "nearest_obstacle"
            ],
        "obstacles": [
            {
                "key":
                    str(
                        obstacle.get(
                            "key",
                            "unknown",
                        )
                    ),
                "center_xyz_m":
                    list(
                        obstacle[
                            "center_xyz_m"
                        ]
                    ),
                "radius_m":
                    float(
                        obstacle.get(
                            "radius_m",
                            0.030,
                        )
                    ),
            }
            for obstacle in obstacles
        ],
        "robot_motion": False,
        "ik_applied": False,
    }

    locked_entities = lock_safe_place_target(
        entities,
        selected_target,
        config,
        projection_report=report,
        preferred_xy=preferred,
    )

    print(
        f"preferred={preferred.tolist()} "
        f"selected="
        f"{selected_target[:2].tolist()} "
        f"safe={len(safe_candidates)}/"
        f"{interior_candidate_count} "
        f"clearance="
        f"{report['selected_minimum_surface_clearance_m']}",
        flush=True,
    )

    return (
        locked_entities,
        report,
    )

def plan_place_sequence(
    q_start: np.ndarray,
    tcp_start: np.ndarray,
    entities: dict[str, Any],
    config: Config,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    q_start = finite_vector(
        q_start,
        6,
        "place_search.q_start",
    )

    tcp_start = finite_vector(
        tcp_start,
        3,
        "place_search.tcp_start",
    )

    initial_target = finite_vector(
        entities["target_xyz_m"],
        3,
        "place_search.initial_target",
    )

    center_value = entities.get(
        "target_detected_center_xyz_m"
    )

    if center_value is not None:
        center = finite_vector(
            center_value,
            3,
            "place_search.center",
        )
    else:
        configured_inset_m = float(
            os.environ.get(
                "VLA_TARGET_INTERIOR_INSET_M",
                "0.070",
            )
        )

        selected_radius_m = float(
            np.linalg.norm(
                initial_target[:2]
            )
        )

        center = initial_target.copy()

        # El criterio de selección desplaza el centro radialmente
        # hacia la base. Se reconstruye aquí el centro
        # detectado para buscar otros puntos interiores.
        if (
            configured_inset_m > 0.0
            and selected_radius_m > 1e-9
        ):
            center[:2] = (
                initial_target[:2]
                / selected_radius_m
                * (
                    selected_radius_m
                    + configured_inset_m
                )
            )

    center_radius_m = float(
        np.linalg.norm(
            center[:2]
        )
    )

    if center_radius_m <= 1e-9:
        raise EpisodeStop(
            "place_search: centro de destino inválido"
        )

    toward_base = (
        -center[:2]
        / center_radius_m
    )

    toward_start = (
        tcp_start[:2]
        - center[:2]
    )

    toward_start_norm = float(
        np.linalg.norm(
            toward_start
        )
    )

    if toward_start_norm <= 1e-9:
        toward_start = (
            toward_base.copy()
        )
    else:
        toward_start = (
            toward_start
            / toward_start_norm
        )

    blended = (
        toward_base
        + toward_start
    )

    blended_norm = float(
        np.linalg.norm(
            blended
        )
    )

    if blended_norm <= 1e-9:
        blended = toward_base.copy()
    else:
        blended = (
            blended
            / blended_norm
        )

    current_offset_m = float(
        np.linalg.norm(
            initial_target[:2]
            - center[:2]
        )
    )

    candidates: list[
        tuple[
            str,
            np.ndarray,
            float,
        ]
    ] = [
        (
            "current_selected",
            initial_target.copy(),
            current_offset_m,
        ),
    ]

    search_specs = [
        (
            "toward_base_0.075",
            0.075,
            toward_base,
        ),
        (
            "toward_base_0.078",
            0.078,
            toward_base,
        ),
        (
            "toward_base_0.080",
            0.080,
            toward_base,
        ),
        (
            "blended_0.078",
            0.078,
            blended,
        ),
        (
            "toward_start_0.078",
            0.078,
            toward_start,
        ),
        (
            "toward_start_0.080",
            0.080,
            toward_start,
        ),
    ]

    for (
        candidate_name,
        offset_m,
        direction_xy,
    ) in search_specs:
        candidate = center.copy()

        candidate[:2] = (
            center[:2]
            + offset_m
            * direction_xy
        )

        candidates.append(
            (
                candidate_name,
                candidate,
                offset_m,
            )
        )

    unique_candidates: list[
        tuple[
            str,
            np.ndarray,
            float,
        ]
    ] = []

    for (
        candidate_name,
        candidate,
        offset_m,
    ) in candidates:
        duplicate = any(
            float(
                np.linalg.norm(
                    candidate[:2]
                    - existing[:2]
                )
            )
            <= 0.001
            for _, existing, _
            in unique_candidates
        )

        if duplicate:
            continue

        unique_candidates.append(
            (
                candidate_name,
                candidate,
                offset_m,
            )
        )

    if bool(
        entities.get(
            "safe_place_target_locked",
            False,
        )
    ):
        unique_candidates = [
            (
                "vla_safe_projected",
                initial_target.copy(),
                current_offset_m,
            ),
        ]

    rejected: list[
        dict[str, Any]
    ] = []

    valid: list[
        dict[str, Any]
    ] = []

    for (
        candidate_index,
        (
            candidate_name,
            candidate_target,
            center_offset_m,
        ),
    ) in enumerate(
        unique_candidates
    ):
        local_entities = dict(
            entities
        )

        candidate_target = (
            candidate_target.copy()
        )

        candidate_target[2] = (
            initial_target[2]
        )

        candidate_safe = (
            candidate_target.copy()
        )

        candidate_safe[2] = float(
            entities.get(
                "place_transit_z_m",
                config.transit_z_m,
            )
        )

        candidate_place = (
            candidate_target.copy()
        )

        candidate_place[2] = (
            config.place_z_m
        )

        local_entities[
            "target_xyz_m"
        ] = candidate_target

        local_entities[
            "target_safe_xyz_m"
        ] = candidate_safe

        local_entities[
            "pre_place_xyz_m"
        ] = candidate_safe.copy()

        local_entities[
            "place_xyz_m"
        ] = candidate_place

        local_entities[
            "retreat_xyz_m"
        ] = candidate_safe.copy()

        try:
            (
                candidate_plan,
                candidate_report,
            ) = _plan_place_sequence_fixed(
                q_start=q_start,
                tcp_start=tcp_start,
                entities=local_entities,
                config=config,
            )
        except EpisodeStop as exc:
            rejected_item = {
                "index": candidate_index,
                "name": candidate_name,
                "target_xyz_m": (
                    candidate_target.tolist()
                ),
                "center_offset_xy_m": (
                    center_offset_m
                ),
                "reason": str(exc),
            }

            rejected.append(
                rejected_item
            )

            print(
                f"name={candidate_name} "
                f"target="
                f"{candidate_target.tolist()} "
                f"reason={exc}",
                flush=True,
            )

            continue

        to_place_report = (
            candidate_report.get(
                "to_place",
                {},
            )
        )

        retreat_report = (
            candidate_report.get(
                "retreat",
                {},
            )
        )

        max_joint_step_rad = max(
            float(
                to_place_report.get(
                    "max_abs_joint_delta_rad",
                    0.0,
                )
            ),
            float(
                retreat_report.get(
                    "max_abs_joint_delta_rad",
                    0.0,
                )
            ),
        )

        max_fk_error_m = max(
            float(
                to_place_report.get(
                    "max_fk_error_m",
                    0.0,
                )
            ),
            float(
                retreat_report.get(
                    "max_fk_error_m",
                    0.0,
                )
            ),
        )

        valid_item = {
            "index": candidate_index,
            "name": candidate_name,
            "target_xyz_m": (
                candidate_target.tolist()
            ),
            "center_offset_xy_m": (
                center_offset_m
            ),
            "max_joint_step_rad": (
                max_joint_step_rad
            ),
            "max_fk_error_m": (
                max_fk_error_m
            ),
            "plan": candidate_plan,
            "report": candidate_report,
        }

        valid.append(
            valid_item
        )

        print(
            f"name={candidate_name} "
            f"target="
            f"{candidate_target.tolist()} "
            f"max_dq="
            f"{max_joint_step_rad:.4f}rad "
            f"max_fk="
            f"{max_fk_error_m:.4f}m",
            flush=True,
        )

    if not valid:
        raise EpisodeStop(
            "place_search: ningún punto interior "
            "permite place + retreat continuos. "
            f"rejected={rejected}"
        )

    # Se prioriza continuidad articular y después
    # proximidad al centro de la zona.
    selected = min(
        valid,
        key=lambda item: (
            item[
                "max_joint_step_rad"
            ],
            item[
                "center_offset_xy_m"
            ],
        ),
    )

    selected_plan = selected[
        "plan"
    ]

    selected_report = selected[
        "report"
    ]

    projection_for_report = dict(
        entities.get(
            "safe_place_projection",
            {},
        )
    )

    projection_for_report[
        "ik_applied"
    ] = True

    projection_for_report[
        "ik_validation_scope"
    ] = (
        "pre_place_place_retreat_"
        "cartesian_sequence"
    )

    projection_for_report[
        "ik_all_steps_ok"
    ] = True

    selected_report[
        "vla_safe_place_projection"
    ] = projection_for_report

    selected_report[
        "reachability_target_search"
    ] = {
        "policy": (
            "full_place_and_retreat_ik_"
            "inside_target_region"
        ),
        "detected_center_xyz_m": (
            center.tolist()
        ),
        "initial_target_xyz_m": (
            initial_target.tolist()
        ),
        "maximum_center_offset_xy_m": (
            0.080
        ),
        "rejected_candidates": (
            rejected
        ),
        "valid_candidates": [
            {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "plan",
                    "report",
                }
            }
            for item in valid
        ],
        "selected": {
            key: value
            for key, value
            in selected.items()
            if key
            not in {
                "plan",
                "report",
            }
        },
    }

    selected_plan[
        "selected_target_xyz_m"
    ] = selected[
        "target_xyz_m"
    ]

    selected_plan[
        "target_search_policy"
    ] = (
        "reachability_aware_region_search"
    )

    print(
        f"name={selected['name']} "
        f"target="
        f"{selected['target_xyz_m']} "
        f"max_dq="
        f"{selected['max_joint_step_rad']:.4f}rad",
        flush=True,
    )

    return (
        selected_plan,
        selected_report,
    )

def reverse_unique(
    values: list[np.ndarray],
) -> list[np.ndarray]:
    output: list[np.ndarray] = []

    for value in reversed(values):
        vector = finite_vector(
            value,
            6,
            "reverse_path",
        )

        if (
            output
            and float(
                np.max(
                    np.abs(
                        vector - output[-1]
                    )
                )
            )
            < 1e-6
        ):
            continue

        output.append(vector.copy())

    return output


def expected_global_home_q() -> np.ndarray:
    raw = os.environ.get(
        "VLA_GLOBAL_HOME_Q_RAD",
        "",
    ).strip()

    if not raw:
        return DEFAULT_GLOBAL_HOME_Q.copy()

    parts = [
        item.strip()
        for item in raw.split(",")
        if item.strip()
    ]

    return finite_vector(
        [float(item) for item in parts],
        6,
        "VLA_GLOBAL_HOME_Q_RAD",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Supervisor temporal π0.5-LIBERO para un episodio "
            "completo UR5e + Photoneo + 3FG15."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["check", "real"],
        default="check",
    )

    parser.add_argument(
        "--instruction",
        default=(
            "put the orange cube in the yellow zone"
        ),
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    config = Config(
        instruction=args.instruction.strip(),
        mode=args.mode,
        open_width_m=float(
            os.environ.get(
                "OPEN_WIDTH_M",
                "0.090",
            )
        ),
        close_width_m=float(
            os.environ.get(
                "CLOSE_WIDTH_M",
                "0.020",
            )
        ),
        pick_z_m=float(
            os.environ.get(
                "PICK_Z_OVERRIDE_M",
                "0.040",
            )
        ),
        pre_pick_z_m=float(
            os.environ.get(
                "VLA_PRE_PICK_Z_M",
                "0.200",
            )
        ),
        transit_z_m=float(
            os.environ.get(
                "VLA_SCENE_TRANSIT_Z_M",
                "0.300",
            )
        ),
        place_z_m=float(
            os.environ.get(
                "PLACE_Z_OVERRIDE_M",
                "0.060",
            )
        ),
        chunk_steps=5,
        max_cycles_approach=int(
            os.environ.get(
                "VLA_PI0_MAX_APPROACH_CYCLES",
                "8",
            )
        ),
        max_cycles_transport=int(
            os.environ.get(
                "VLA_PI0_MAX_TRANSPORT_CYCLES",
                "8",
            )
        ),
        approach_goal_tolerance_m=float(
            os.environ.get(
                "VLA_PI0_APPROACH_TOL_M",
                "0.050",
            )
        ),
        transport_goal_tolerance_m=float(
            os.environ.get(
                "VLA_PI0_TRANSPORT_TOL_M",
                "0.055",
            )
        ),
        maximum_step_distance_increase_m=float(
            os.environ.get(
                "VLA_PI0_MAX_DISTANCE_INCREASE_M",
                "0.002",
            )
        ),
        minimum_chunk_progress_m=float(
            os.environ.get(
                "VLA_PI0_MIN_CHUNK_PROGRESS_M",
                "0.003",
            )
        ),
        max_chunk_joint_step_rad=float(
            os.environ.get(
                "VLA_PI0_MAX_CHUNK_DQ_RAD",
                "0.150",
            )
        ),
        max_contact_joint_step_rad=float(
            os.environ.get(
                "VLA_MAX_DQ_PRE_PICK",
                "0.450",
            )
        ),
        max_fk_error_m=float(
            os.environ.get(
                "VLA_PI0_MAX_FK_ERROR_M",
                "0.012",
            )
        ),
        max_tilt_deg=float(
            os.environ.get(
                "VLA_PI0_MAX_TILT_DEG",
                "2.0",
            )
        ),
        pick_actual_error_m=float(
            os.environ.get(
                "PICK_MAX_ERROR_M",
                "0.030",
            )
        ),
        place_actual_error_m=float(
            os.environ.get(
                "PLACE_MAX_ERROR_M",
                "0.025",
            )
        ),
        waypoint_duration_s=float(
            os.environ.get(
                "VLA_PI0_WAYPOINT_DURATION_S",
                "2.5",
            )
        ),
        gripper_hold_s=float(
            os.environ.get(
                "VLA_PI0_GRIPPER_HOLD_S",
                "3.0",
            )
        ),
    )

    if not config.instruction:
        raise EpisodeStop("La instrucción está vacía")

    if not (
        0.0
        < config.close_width_m
        < config.open_width_m
        <= 0.150
    ):
        raise EpisodeStop(
            "Anchuras de garra inválidas: "
            f"open={config.open_width_m}, "
            f"close={config.close_width_m}"
        )

    if not (
        0.030
        <= config.pick_z_m
        <= 0.080
    ):
        raise EpisodeStop(
            f"pick_z_m fuera de rango: {config.pick_z_m}"
        )

    if not (
        config.pick_z_m
        < config.pre_pick_z_m
        < config.transit_z_m
        <= 0.40
    ):
        raise EpisodeStop(
            "Contrato Z inválido: "
            f"pick={config.pick_z_m}, "
            f"pre={config.pre_pick_z_m}, "
            f"transit={config.transit_z_m}"
        )

    return config


def preflight_real(config: Config) -> None:
    if config.mode != "real":
        return

    required = {
        "CONFIRM_PI0_FULL_EPISODE": "YES",
        "CONFIRM_PHYSICAL_EXECUTION": "YES",
        "CONFIRM_EXECUTION_RISK": "YES",
    }

    missing = [
        key
        for key, expected in required.items()
        if os.environ.get(key) != expected
    ]

    if missing:
        raise EpisodeStop(
            "Faltan confirmaciones de ejecución real: "
            + ", ".join(missing)
        )

    if os.environ.get(
        "VLA_PICK_Z_MODE",
        "",
    ) != "override":
        raise EpisodeStop(
            "VLA_PICK_Z_MODE debe ser override"
        )

    env_pick = float(
        os.environ.get(
            "VLA_PICK_Z_OVERRIDE_M",
            "nan",
        )
    )

    if not math.isfinite(env_pick):
        raise EpisodeStop(
            "Falta VLA_PICK_Z_OVERRIDE_M"
        )

    if abs(
        env_pick - config.pick_z_m
    ) > 1e-9:
        raise EpisodeStop(
            "PICK_Z_OVERRIDE_M y "
            "VLA_PICK_Z_OVERRIDE_M no coinciden"
        )


def _one_shot_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _one_shot_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _one_shot_jsonable(item)
            for item in value
        ]
    return value

def _one_shot_validate_joint_chain(
    q_start: np.ndarray,
    q_targets: list[np.ndarray],
    label: str,
) -> dict[str, Any]:
    maximum_step_rad = float(
        os.environ.get(
            "VLA_ONE_SHOT_MAX_CHAIN_STEP_RAD",
            "0.45",
        )
    )

    if not 0.15 <= maximum_step_rad <= 0.50:
        raise EpisodeStop(
            "VLA_ONE_SHOT_MAX_CHAIN_STEP_RAD "
            "debe estar entre 0.15 y 0.50 rad"
        )

    previous = finite_vector(
        q_start,
        6,
        f"{label}.q_start",
    )

    maximum_observed = 0.0
    steps = []

    for index, value in enumerate(q_targets, start=1):
        current = finite_vector(
            value,
            6,
            f"{label}.q[{index}]",
        )

        step = float(
            np.max(np.abs(current - previous))
        )

        if step > maximum_step_rad:
            raise EpisodeStop(
                f"{label}: discontinuidad articular "
                f"{step:.4f} rad > "
                f"{maximum_step_rad:.4f} rad "
                f"en el punto {index}"
            )

        maximum_observed = max(
            maximum_observed,
            step,
        )

        steps.append(step)
        previous = current.copy()

    if not q_targets:
        raise EpisodeStop(
            f"{label}: cadena articular vacía"
        )

    return {
        "label": label,
        "number_of_targets": len(q_targets),
        "max_allowed_step_rad": maximum_step_rad,
        "max_observed_step_rad": maximum_observed,
        "steps_rad": steps,
        "valid": True,
    }


def _predicted_tcp_quaternion_xyzw(
    q_value: np.ndarray,
) -> np.ndarray:
    q = finite_vector(
        q_value,
        6,
        "predicted.q",
    )

    (
        _,
        model,
        data,
        _,
        tcp_id,
        qpos_ids,
        _,
    ) = robot_base.load_scene()

    robot_base.set_robot_q(
        model,
        data,
        qpos_ids,
        q,
    )

    rotation = np.asarray(
        data.site_xmat[tcp_id],
        dtype=float,
    ).reshape(3, 3)

    quaternion_wxyz = np.zeros(
        4,
        dtype=float,
    )

    robot_base.mujoco.mju_mat2Quat(
        quaternion_wxyz,
        np.ascontiguousarray(
            rotation.reshape(-1)
        ),
    )

    quaternion_xyzw = np.array(
        [
            quaternion_wxyz[1],
            quaternion_wxyz[2],
            quaternion_wxyz[3],
            quaternion_wxyz[0],
        ],
        dtype=float,
    )

    return action_controller.normalize_quaternion_xyzw(
        quaternion_xyzw
    )


def _infer_pi0_from_predicted_state(
    run_dir: Path,
    config: Config,
    index: int,
    label: str,
    image_path: Path,
    scene: dict[str, Any],
    reference_snapshot: dict[str, Any],
    q_predicted: np.ndarray,
    tcp_predicted: np.ndarray,
    gripper_width_m: float,
) -> dict[str, Any]:
    global _PI0_COMMANDED_GRIPPER_WIDTH_M

    q_value = finite_vector(
        q_predicted,
        6,
        f"{label}.q",
    )

    tcp_value = finite_vector(
        tcp_predicted,
        3,
        f"{label}.tcp",
    )

    quaternion = (
        _predicted_tcp_quaternion_xyzw(
            q_value
        )
    )

    snapshot = dict(
        reference_snapshot
    )

    snapshot[
        "tcp_position_base_m"
    ] = tcp_value.tolist()

    snapshot[
        "tcp_position_m"
    ] = tcp_value.tolist()

    snapshot[
        "tcp_quaternion_xyzw"
    ] = quaternion.tolist()

    snapshot[
        "tcp_orientation_xyzw"
    ] = quaternion.tolist()

    snapshot["gripper_width_m"] = float(
        gripper_width_m
    )

    _PI0_COMMANDED_GRIPPER_WIDTH_M = float(
        gripper_width_m
    )

    state_document = (
        action_controller.build_pi0_virtual_state(
            real_snapshot=snapshot,
            gripper_open_width_m=(
                config.open_width_m
            ),
            gripper_close_width_m=(
                config.close_width_m
            ),
        )
    )

    prefix = f"{index:03d}_{label}"

    state_path = (
        run_dir
        / f"{prefix}_pi0_state.json"
    )

    json_write(
        state_path,
        state_document,
    )

    inference_started_ns = time.time_ns()

    pi0_document = action_controller.execute_pi0_inference(
        instruction=config.instruction,
        image_path=Path(image_path),
        state_path=state_path,
    )

    raw_actions = np.asarray(
        pi0_document["output"]["actions"],
        dtype=float,
    )

    if (
        raw_actions.shape != (10, 7)
        or not np.all(
            np.isfinite(raw_actions)
        )
    ):
        raise EpisodeStop(
            f"{label}: salida pi0 inválida: "
            f"{raw_actions.shape}"
        )

    raw_source = Path(
        action_controller.PI0_LATEST_OUTPUT
    )

    if not raw_source.is_file():
        raise EpisodeStop(
            f"{label}: no existe salida raw"
        )

    if (
        raw_source.stat().st_mtime_ns
        < inference_started_ns
        - 2_000_000_000
    ):
        raise EpisodeStop(
            f"{label}: salida raw antigua"
        )

    raw_copy = (
        run_dir
        / f"{prefix}_pi0_raw.json"
    )

    raw_copy.write_text(
        raw_source.read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    record = {
        "scene_path": str(
            _PI0_INITIAL_VISUAL_CONTEXT[
                "scene_path"
            ]
        ),
        "scene": copy.deepcopy(scene),
        "image_path": str(image_path),
        "measured_snapshot": snapshot,
        "model_snapshot": snapshot,
        "measured_gripper_width_m": float(
            gripper_width_m
        ),
        "commanded_gripper_width_m": float(
            gripper_width_m
        ),
        "state_path": str(state_path),
        "raw_path": str(raw_copy),
        "raw_sha256": sha256_file(
            raw_copy
        ),
        "raw_actions": raw_actions,
        "raw_gripper": (
            raw_actions[:, 6].tolist()
        ),
        "predicted_state": True,
        "predicted_q_rad": (
            q_value.tolist()
        ),
        "predicted_tcp_base_m": (
            tcp_value.tolist()
        ),
        "predicted_quaternion_xyzw": (
            quaternion.tolist()
        ),
    }

    json_write(
        run_dir
        / f"{prefix}_observation.json",
        _one_shot_jsonable(record),
    )

    print(
        f"label={label} "
        f"tcp={tcp_value.tolist()} "
        f"gripper={gripper_width_m:.4f}m",
        flush=True,
    )

    return record



def _geometric_rotation_lock_report(
    phase: str,
    q_values: list[np.ndarray],
) -> dict[str, Any]:
    return {
        "phase": phase,
        "architecture": (
            "supervised_vla_geometric_orientation"
        ),
        "vla_rotation_requested": True,
        "vla_rotation_applied": False,
        "executed_rotation_retention": 0.0,
        "reason": (
            "orientacion geometrica bloqueada "
            "en fases sensibles"
        ),
        "number_of_waypoints": len(q_values),
        "valid": True,
    }


def _build_one_shot_preplan(
    run_dir: Path,
    config: Config,
) -> dict[str, Any]:
    """
    Una captura Photoneo y cuatro inferencias pi0.5 se realizan antes
        del primer movimiento: approach, grasp, transport y release.
        Cada una usa el estado 8D previsto para su fase.
    """
    observation = infer_fresh_pi0(
        run_dir=run_dir,
        config=config,
        index=1,
        label="approach_cycle_1",
        commanded_gripper_width_m=(
            config.open_width_m
        ),
    )

    entities = resolve_entities(
        observation["scene"],
        config,
    )

    raw_actions = np.asarray(
        observation["raw_actions"],
        dtype=float,
    )

    if (
        raw_actions.shape != (10, 7)
        or not np.all(np.isfinite(raw_actions))
    ):
        raise EpisodeStop(
            "ONE_SHOT: salida pi0 invalida"
        )

    with robot_node() as node:
        q_home, tcp_home = (
            current_robot_state(node)
        )

    home_reference = expected_global_home_q()

    home_error = float(
        np.max(
            np.abs(q_home - home_reference)
        )
    )

    if home_error > 0.08:
        raise EpisodeStop(
            "ONE_SHOT debe comenzar en home: "
            f"error={home_error:.4f} rad"
        )

    snapshot_tcp = finite_vector(
        observation["measured_snapshot"][
            "tcp_position_base_m"
        ],
        3,
        "one_shot.snapshot_tcp",
    )

    snapshot_error = float(
        np.linalg.norm(tcp_home - snapshot_tcp)
    )

    if snapshot_error > 0.005:
        raise EpisodeStop(
            "ONE_SHOT: el robot cambio durante "
            f"la inferencia: {snapshot_error:.4f} m"
        )

    # Compara Full7D y geometría dentro del mismo
    # contexto local. No modifica el solver IK.
    def retain_full7d_position(
        vla_chain,
        geometric_chain,
        label,
        target_xyz=None,
    ):
        vla_values = [
            finite_vector(
                value,
                6,
                f"{label}.vla[{index}]",
            ).copy()
            for index, value in enumerate(vla_chain)
        ]

        geometric_values = [
            finite_vector(
                value,
                6,
                f"{label}.geometric[{index}]",
            ).copy()
            for index, value in enumerate(geometric_chain)
        ]

        if (
            not vla_values
            or len(vla_values) != len(geometric_values)
        ):
            raise EpisodeStop(
                f"{label}: cadenas Full7D incompatibles"
            )

        target_value = (
            finite_vector(
                target_xyz,
                3,
                label + ".target",
            )
            if target_xyz is not None
            else np.zeros(3, dtype=float)
        )

        context = build_ik_context(
            q_reference=geometric_values[-1],
            tcp_real_reference=target_value,
        )

        def raw_fk(q_value):
            robot_base.set_robot_q(
                context["model"],
                context["data"],
                context["qpos_ids"],
                q_value,
            )

            position, _, _ = (
                robot_base.get_site_pos_axis(
                    context["data"],
                    context["tcp_id"],
                )
            )

            return np.asarray(position, dtype=float)

        allowed_m = min(
            0.004,
            float(config.max_fk_error_m),
        )

        waypoint_drifts = []

        for q_vla, q_geometric in zip(
            vla_values,
            geometric_values,
        ):
            waypoint_drifts.append(
                float(
                    np.linalg.norm(
                        raw_fk(q_vla)
                        - raw_fk(q_geometric)
                    )
                )
            )

        maximum_drift_m = max(waypoint_drifts)

        vla_retained = bool(
            maximum_drift_m <= allowed_m
        )

        selected = (
            vla_values
            if vla_retained
            else geometric_values
        )

        endpoint_error_m = None
        endpoint_xyz = None

        if target_xyz is not None:
            endpoint_xyz = (
                raw_fk(selected[-1])
                + context["offset"]
            )

            endpoint_error_m = float(
                np.linalg.norm(
                    endpoint_xyz - target_value
                )
            )

            if endpoint_error_m > allowed_m:
                raise EpisodeStop(
                    f"{label}: endpoint cartesiano "
                    f"{endpoint_error_m:.4f}m > "
                    f"{allowed_m:.4f}m"
                )

        report = {
            "policy": (
                "position_primary_rotation_secondary"
            ),
            "vla_rotation_retained": vla_retained,
            "executed_rotation_retention": (
                1.0 if vla_retained else 0.0
            ),
            "maximum_local_drift_m": (
                maximum_drift_m
            ),
            "allowed_m": allowed_m,
            "waypoint_drifts_m": waypoint_drifts,
            "endpoint_fk_xyz_m": (
                endpoint_xyz.tolist()
                if endpoint_xyz is not None
                else None
            ),
            "target_error_m": endpoint_error_m,
            "valid": True,
        }

        print(
            f"phase={label} "
            f"vla_retained="
            f"{str(vla_retained).lower()} "
            f"max_drift="
            f"{maximum_drift_m:.4f}m "
            f"target_error="
            f"{endpoint_error_m}",
            flush=True,
        )

        return selected, report

    (
        approach_targets_xyz,
        approach_translation_report,
    ) = make_vla_chunk_targets(
        raw_actions=raw_actions,
        tcp_start=tcp_home,
        goal_xyz=entities[
            "object_safe_xyz_m"
        ],
        config=config,
        phase="approach_object",
    )

    (
        approach_reference_q,
        approach_translation_ik,
    ) = solve_cartesian_targets(
        q_start=q_home,
        tcp_start=tcp_home,
        targets=approach_targets_xyz,
        config=config,
        label="one_shot_approach_translation",
        max_joint_step_rad=(
            config.max_chunk_joint_step_rad
        ),
    )

    approach_q = [
        value.copy()
        for value in approach_reference_q
    ]

    approach_rotation_report = (
        _geometric_rotation_lock_report(
            "one_shot_approach",
            approach_q,
        )
    )

    (
        approach_q,
        approach_position_contract,
    ) = retain_full7d_position(
        vla_chain=approach_q,
        geometric_chain=approach_reference_q,
        label="approach",
        target_xyz=approach_targets_xyz[-1],
    )

    # Un único marco FK anclado en el home real.
    # q y TCP usados para iniciar cada fase deben
    # pertenecer al mismo estado físico.
    episode_fk_context = build_ik_context(
        q_reference=q_home,
        tcp_real_reference=tcp_home,
    )

    def episode_global_fk(
        q_value,
        label,
    ):
        q_value = finite_vector(
            q_value,
            6,
            label + ".q",
        )

        robot_base.set_robot_q(
            episode_fk_context["model"],
            episode_fk_context["data"],
            episode_fk_context["qpos_ids"],
            q_value,
        )

        position, _, _ = (
            robot_base.get_site_pos_axis(
                episode_fk_context["data"],
                episode_fk_context["tcp_id"],
            )
        )

        return (
            np.asarray(position, dtype=float)
            + episode_fk_context["offset"]
        )

    def require_global_endpoint(
        label,
        q_value,
        target_xyz,
    ):
        target_xyz = finite_vector(
            target_xyz,
            3,
            label + ".target",
        )

        actual_xyz = episode_global_fk(
            q_value,
            label,
        )

        error_m = float(
            np.linalg.norm(
                actual_xyz - target_xyz
            )
        )

        allowed_m = min(
            0.004,
            float(config.max_fk_error_m),
        )

        if error_m > allowed_m:
            raise EpisodeStop(
                f"{label}: error FK global "
                f"{error_m:.4f}m > "
                f"{allowed_m:.4f}m; "
                f"target={target_xyz.tolist()} "
                f"fk={actual_xyz.tolist()}"
            )

        print(
            f"label={label} "
            f"error={error_m:.4f}m",
            flush=True,
        )

        return {
            "target_xyz_m": target_xyz.tolist(),
            "fk_xyz_m": actual_xyz.tolist(),
            "error_m": error_m,
            "allowed_m": allowed_m,
            "valid": True,
        }

    approach_tcp_end = episode_global_fk(
        approach_q[-1],
        "approach_tcp_end",
    )

    (
        contact_plan,
        contact_report,
    ) = plan_contact_sequence(
        q_start=approach_q[-1],
        tcp_start=approach_tcp_end,
        entities=entities,
        config=config,
    )

    (
        q_to_pre_pick,
        _,
        _,
    ) = split_chain_at_target(
        q_targets=contact_plan[
            "q_to_pick_steps"
        ],
        q_target=contact_plan["q_pre_pick"],
        label="one_shot_pre_pick_split",
    )


    grasp_observation = _infer_pi0_from_predicted_state(
        run_dir=run_dir,
        config=config,
        index=2,
        label="grasp_predicted",
        image_path=Path(
            observation["image_path"]
        ),
        scene=observation["scene"],
        reference_snapshot=observation[
            "measured_snapshot"
        ],
        q_predicted=contact_plan[
            "q_pre_pick"
        ],
        tcp_predicted=entities[
            "pre_pick_xyz_m"
        ],
        gripper_width_m=(
            config.open_width_m
        ),
    )

    q_pick = [
        contact_plan["q_pick"].copy()
    ]

    pick_rotation_report = (
        _geometric_rotation_lock_report(
            "preinferred_pick",
            q_pick,
        )
    )

    (
        q_pick,
        pick_position_contract,
    ) = retain_full7d_position(
        vla_chain=q_pick,
        geometric_chain=[
            contact_plan["q_pick"]
        ],
        label="pick",
        target_xyz=entities["pick_xyz_m"],
    )

    q_lift = [
        value.copy()
        for value in contact_plan[
            "q_lift_steps"
        ]
    ]

    lift_rotation_report = (
        _geometric_rotation_lock_report(
            "preinferred_lift",
            q_lift,
        )
    )

    (
        q_lift,
        lift_position_contract,
    ) = retain_full7d_position(
        vla_chain=q_lift,
        geometric_chain=contact_plan[
            "q_lift_steps"
        ],
        label="lift",
        target_xyz=entities["lift_xyz_m"],
    )

    close_event = decode_full7d_gripper(
        raw_actions=grasp_observation["raw_actions"],
        expected_event="close",
        config=config,
        phase="preinferred_grasp",
    )

    local_lift_tcp = finite_vector(
        entities["lift_xyz_m"],
        3,
        "one_shot.local_lift_tcp",
    )

    (
        q_handoff,
        transport_start_tcp,
        transport_handoff_report,
    ) = plan_transport_handoff(
        q_start=q_lift[-1],
        tcp_start=local_lift_tcp,
        entities=entities,
        scene=observation["scene"],
        config=config,
    )

    return_object_safe_targets = (
        _phase_segment(
            local_lift_tcp,
            finite_vector(
                entities[
                    "object_safe_xyz_m"
                ],
                3,
                "return.object_safe_xyz",
            ),
            0.012,
            "return_local_lift_to_object_safe",
        )
    )

    if not return_object_safe_targets:
        return_object_safe_targets = [
            finite_vector(
                entities[
                    "object_safe_xyz_m"
                ],
                3,
                "return.object_safe_target",
            )
        ]

    (
        return_object_safe_q,
        return_object_safe_report,
    ) = solve_cartesian_targets(
        q_start=q_lift[-1],
        tcp_start=local_lift_tcp,
        targets=return_object_safe_targets,
        config=config,
        label=(
            "return_local_lift_to_object_safe"
        ),
        max_joint_step_rad=(
            config.max_contact_joint_step_rad
        ),
        contact=True,
    )

    transport_observation = _infer_pi0_from_predicted_state(
        run_dir=run_dir,
        config=config,
        index=3,
        label="transport_predicted",
        image_path=Path(
            observation["image_path"]
        ),
        scene=observation["scene"],
        reference_snapshot=observation[
            "measured_snapshot"
        ],
        q_predicted=q_handoff[-1],
        tcp_predicted=transport_start_tcp,
        gripper_width_m=(
            close_event["width_m"]
        ),
    )

    (
        transport_targets_xyz,
        transport_translation_report,
    ) = _make_vla_chunk_targets_chunk_only(
        raw_actions=transport_observation["raw_actions"],
        tcp_start=transport_start_tcp,
        goal_xyz=entities[
            "target_safe_xyz_m"
        ],
        config=config,
        phase="transport_target",
    )

    (
        transport_reference_q,
        transport_translation_ik,
    ) = solve_cartesian_targets(
        q_start=q_handoff[-1],
        tcp_start=transport_start_tcp,
        targets=transport_targets_xyz,
        config=config,
        label="transport_target_one_shot_translation",
        max_joint_step_rad=(
            config.max_chunk_joint_step_rad
        ),
    )

    transport_q = [
        value.copy()
        for value in transport_reference_q
    ]

    transport_rotation_report = (
        _geometric_rotation_lock_report(
            "preinferred_transport",
            transport_q,
        )
    )

    (
        transport_q,
        transport_position_contract,
    ) = retain_full7d_position(
        vla_chain=transport_q,
        geometric_chain=transport_reference_q,
        label="transport",
        target_xyz=None,
    )

    transport_tcp_end = finite_vector(
        transport_translation_ik[
            "waypoints"
        ][-1]["fk_real_base_m"],
        3,
        "one_shot.transport_tcp_end",
    )

    # El chunk de transporte puede terminar por debajo
    # del plano global y en una posición radial desde la
    # que la subida vertical a transit_z no sea alcanzable.
    # Se reutiliza el mismo planificador de handoff usado
    # después del lift. Primero intenta ascender sin retirada;
    # si no es posible, busca la retirada mínima segura hacia
    # una región interior y asciende allí.
    transit_entry_tolerance_m = 0.005

    if (
        float(transport_tcp_end[2])
        >= float(config.transit_z_m)
        - transit_entry_tolerance_m
    ):
        q_place_entry = []

        place_start_q = finite_vector(
            transport_q[-1],
            6,
            "place_entry.transport_q_end",
        ).copy()

        place_start_tcp = (
            transport_tcp_end.copy()
        )

        place_entry_report = {
            "policy": (
                "already_at_global_"
                "transit_plane"
            ),
            "requested_transit_z_m": (
                float(
                    config.transit_z_m
                )
            ),
            "transport_tcp_end_m": (
                transport_tcp_end.tolist()
            ),
            "place_start_tcp_m": (
                place_start_tcp.tolist()
            ),
            "joint_points": 0,
            "all_steps_ok": True,
        }

        print(
            "policy=already_at_transit "
            f"start={transport_tcp_end.tolist()} "
            f"place_start={place_start_tcp.tolist()}",
            flush=True,
        )

    else:
        (
            q_place_entry,
            place_start_tcp,
            place_entry_report,
        ) = plan_transport_handoff(
            q_start=transport_q[-1],
            tcp_start=transport_tcp_end,
            entities=entities,
            scene=observation["scene"],
            config=config,
        )

        if not q_place_entry:
            raise EpisodeStop(
                "place_entry: el handoff no "
                "generó soluciones articulares"
            )

        place_start_q = finite_vector(
            q_place_entry[-1],
            6,
            "place_entry.q_end",
        ).copy()

        print(
            "policy=nearest_reachable_"
            "retreat_then_vertical_transit "
            f"transport_end="
            f"{transport_tcp_end.tolist()} "
            f"place_start="
            f"{place_start_tcp.tolist()} "
            f"points={len(q_place_entry)}",
            flush=True,
        )

    place_entry_z_error_m = abs(
        float(place_start_tcp[2])
        - float(config.transit_z_m)
    )

    if (
        place_entry_z_error_m
        > config.max_fk_error_m
    ):
        raise EpisodeStop(
            "place_entry: el handoff no terminó "
            "en el plano global: "
            f"z_error={place_entry_z_error_m:.4f}m"
        )

    # El valor cartesiano alcanzado en el handoff pasa a ser
    # el contrato de entrada a place. No se reconstruye
    # el valor nominal config.transit_z_m.
    place_entities = dict(
        entities
    )

    place_entities[
        "place_transit_z_m"
    ] = float(
        place_start_tcp[2]
    )

    print(
        f"configured="
        f"{float(config.transit_z_m):.6f}m "
        f"validated="
        f"{float(place_start_tcp[2]):.6f}m "
        f"error="
        f"{abs(float(place_start_tcp[2]) - float(config.transit_z_m)):.6f}m "
        "source=validated_handoff_fk",
        flush=True,
    )

    (
        place_plan,
        place_report,
    ) = plan_place_sequence(
        q_start=place_start_q,
        tcp_start=place_start_tcp,
        entities=place_entities,
        config=config,
    )

    (
        q_to_pre_place,
        _,
        _,
    ) = split_chain_at_target(
        q_targets=place_plan[
            "q_to_place_steps"
        ],
        q_target=place_plan["q_pre_place"],
        label="one_shot_pre_place_split",
    )

    # El planificador geométrico ya genera un descenso
    # segmentado y validado desde q_pre_place hasta q_place.
    # El Full7D debe aplicar la orientación de pi0 sobre esa
    # cadena completa, no saltar directamente al endpoint.
    place_reference_chain = [
        finite_vector(
            value,
            6,
            f"place_reference_chain[{index}]",
        ).copy()
        for index, value in enumerate(
            place_plan["q_to_place_steps"]
        )
    ]

    q_pre_place_reference = finite_vector(
        place_plan["q_pre_place"],
        6,
        "place_plan.q_pre_place",
    )

    pre_place_distances = [
        float(
            np.max(
                np.abs(
                    value
                    - q_pre_place_reference
                )
            )
        )
        for value in place_reference_chain
    ]

    if not pre_place_distances:
        raise EpisodeStop(
            "La cadena de place está vacía"
        )

    pre_place_index = int(
        np.argmin(
            np.asarray(
                pre_place_distances,
                dtype=float,
            )
        )
    )

    pre_place_match_error = float(
        pre_place_distances[
            pre_place_index
        ]
    )

    if pre_place_match_error > 1e-5:
        raise EpisodeStop(
            "q_pre_place no aparece en "
            "q_to_place_steps: "
            f"error={pre_place_match_error:.8f}rad"
        )

    place_descent_reference_q = [
        value.copy()
        for value in place_reference_chain[
            pre_place_index + 1:
        ]
    ]

    if not place_descent_reference_q:
        raise EpisodeStop(
            "No existen waypoints segmentados "
            "entre pre-place y place"
        )

    q_place_reference = finite_vector(
        place_plan["q_place"],
        6,
        "place_plan.q_place",
    )

    place_endpoint_error = float(
        np.max(
            np.abs(
                place_descent_reference_q[-1]
                - q_place_reference
            )
        )
    )

    if place_endpoint_error > 1e-5:
        raise EpisodeStop(
            "El descenso segmentado no termina "
            "en q_place: "
            f"error={place_endpoint_error:.8f}rad"
        )

    geometric_descent_max_dq = 0.0
    previous_reference_q = (
        q_pre_place_reference.copy()
    )

    for value in place_descent_reference_q:
        geometric_descent_max_dq = max(
            geometric_descent_max_dq,
            float(
                np.max(
                    np.abs(
                        value
                        - previous_reference_q
                    )
                )
            ),
        )

        previous_reference_q = value.copy()

    if (
        geometric_descent_max_dq
        > config.max_contact_joint_step_rad
    ):
        raise EpisodeStop(
            "El descenso geométrico segmentado "
            "ya es discontinuo antes del Full7D: "
            f"{geometric_descent_max_dq:.4f}rad"
        )

    print(
        f"points={len(place_descent_reference_q)} "
        f"geometric_max_dq="
        f"{geometric_descent_max_dq:.4f}rad",
        flush=True,
    )

    selected_target = finite_vector(
        place_plan["selected_target_xyz_m"],
        3,
        "one_shot.selected_target",
    )

    pre_place_xyz = selected_target.copy()
    pre_place_xyz[2] = float(
        place_start_tcp[2]
    )

    place_xyz = selected_target.copy()
    place_xyz[2] = float(
        config.place_z_m
    )


    # Esta inferencia se realiza en pre-place.
    # Se usa para orientar y segmentar el
    # descenso, pero no para decidir la apertura.
    preplace_observation = _infer_pi0_from_predicted_state(
        run_dir=run_dir,
        config=config,
        index=4,
        label="preplace_predicted",
        image_path=Path(
            observation["image_path"]
        ),
        scene=observation["scene"],
        reference_snapshot=observation[
            "measured_snapshot"
        ],
        q_predicted=place_plan[
            "q_pre_place"
        ],
        tcp_predicted=pre_place_xyz,
        gripper_width_m=(
            close_event["width_m"]
        ),
    )

    q_place = [
        place_plan["q_place"].copy()
    ]

    place_rotation_report = (
        _geometric_rotation_lock_report(
            "preinferred_place",
            q_place,
        )
    )

    (
        q_place,
        place_position_contract,
    ) = retain_full7d_position(
        vla_chain=q_place,
        geometric_chain=(
            place_descent_reference_q
        ),
        label="place",
        target_xyz=place_xyz,
    )

    # La apertura se consulta desde el estado
    # previsto en el punto real de colocación:
    # objeto abajo y garra todavía cerrada.
    release_observation = _infer_pi0_from_predicted_state(
        run_dir=run_dir,
        config=config,
        index=5,
        label="release_at_place_predicted",
        image_path=Path(
            observation["image_path"]
        ),
        scene=observation["scene"],
        reference_snapshot=observation[
            "measured_snapshot"
        ],
        q_predicted=q_place[-1],
        tcp_predicted=place_xyz,
        gripper_width_m=(
            close_event["width_m"]
        ),
    )

    print(
        f"tcp={place_xyz.tolist()} "
        f"gripper_closed="
        f"{close_event['width_m']:.4f}m",
        flush=True,
    )

    q_retreat = [
        value.copy()
        for value in place_plan[
            "q_retreat_steps"
        ]
    ]

    retreat_rotation_report = (
        _geometric_rotation_lock_report(
            "preinferred_retreat",
            q_retreat,
        )
    )

    # La inferencia de apertura se guarda como referencia.
    # En las pruebas realizadas con la imagen inicial reutilizada,
    # pi0.5 mantiene el cierre incluso estando en place_xyz.
    # La apertura se ordena después de comprobar que
    # el robot ha alcanzado el punto de colocación.
    release_raw_actions = np.asarray(
        release_observation["raw_actions"],
        dtype=float,
    )

    if (
        release_raw_actions.shape != (10, 7)
        or not np.all(
            np.isfinite(
                release_raw_actions
            )
        )
    ):
        raise EpisodeStop(
            "Salida release de pi0 inválida: "
            f"shape={release_raw_actions.shape}"
        )

    release_window_count = min(
        int(config.chunk_steps),
        int(release_raw_actions.shape[0]),
    )

    if release_window_count < 1:
        raise EpisodeStop(
            "La ventana release de pi0 está vacía"
        )

    release_raw_window = (
        release_raw_actions[
            :release_window_count,
            6,
        ]
    )

    release_raw_median = float(
        np.median(
            release_raw_window
        )
    )

    release_hold_s = float(
        os.environ.get(
            "VLA_ONE_SHOT_RELEASE_HOLD_S",
            "2.5",
        )
    )

    if not (
        math.isfinite(release_hold_s)
        and 0.5 <= release_hold_s <= 6.0
    ):
        raise EpisodeStop(
            "VLA_ONE_SHOT_RELEASE_HOLD_S "
            "debe estar entre 0.5 y 6.0 s"
        )

    release_event = {
        "phase": "release_at_place",
        "requested_event": "open",
        "event": "open",
        "decoded_model_event": (
            "close"
            if release_raw_median >= 0.0
            else "open"
        ),
        "width_m": float(
            config.open_width_m
        ),
        "hold_s": release_hold_s,
        "source": (
            "supervisor_scheduled_after_"
            "validated_place"
        ),
        "model_open_executed": False,
        "supervised_open_executed": True,
        "vla_release_probe_recorded": True,
        "vla_release_raw_median": (
            release_raw_median
        ),
        "vla_release_raw_window": (
            release_raw_window.tolist()
        ),
        "reason": (
            "pi0.5_libero_did_not_emit_open_"
            "with_static_initial_visual_context"
        ),
    }

    print(
        f"model_raw_median={release_raw_median:.6f} "
        f"executed_width={config.open_width_m:.4f}m "
        "source=supervisor_after_validated_place",
        flush=True,
    )



    global_endpoint_validation = {
        "pre_pick": require_global_endpoint(
            "preplan.pre_pick",
            contact_plan["q_pre_pick"],
            entities["pre_pick_xyz_m"],
        ),
        "pick": require_global_endpoint(
            "preplan.pick",
            q_pick[-1],
            entities["pick_xyz_m"],
        ),
        "lift": require_global_endpoint(
            "preplan.lift",
            q_lift[-1],
            entities["lift_xyz_m"],
        ),
        "place": require_global_endpoint(
            "preplan.place",
            q_place[-1],
            place_xyz,
        ),
    }

    preclose_q = (
        [value.copy() for value in approach_q]
        + [
            value.copy()
            for value in q_to_pre_pick
        ]
        + [value.copy() for value in q_pick]
    )

    held_q = (
        [value.copy() for value in q_lift]
        + [
            value.copy()
            for value in q_handoff
        ]
        + [
            value.copy()
            for value in transport_q
        ]
        + [
            value.copy()
            for value in q_place_entry
        ]
        + [
            value.copy()
            for value in q_to_pre_place
        ]
        + [value.copy() for value in q_place]
    )

    # Después de abrir, no se enlaza el retreat Full7D
    # con una rama geométrica diferente. Se deshace
    # exactamente el descenso q_place ya validado.
    # held_q termina en q_place[-1]. La vuelta comienza
    # por q_place[-2], q_place[-3], etc., hasta recuperar
    # el q_pre_place geométrico desde el que comenzó
    # apply_full7d_rotation.
    if not q_place:
        raise EpisodeStop(
            "q_place está vacío; no se puede "
            "construir el retorno post-release"
        )

    q_pre_place_geometric = finite_vector(
        place_plan["q_pre_place"],
        6,
        "post_release.q_pre_place_geometric",
    )

    post_release_retrace_q = [
        finite_vector(
            value,
            6,
            f"post_release.q_place_reverse[{index}]",
        ).copy()
        for index, value in enumerate(
            reversed(q_place[:-1])
        )
    ]

    post_release_retrace_q.append(
        q_pre_place_geometric.copy()
    )

    retrace_previous = finite_vector(
        q_place[-1],
        6,
        "post_release.q_place_result",
    )

    retrace_max_dq = 0.0

    for index, value in enumerate(
        post_release_retrace_q
    ):
        step = float(
            np.max(
                np.abs(
                    value
                    - retrace_previous
                )
            )
        )

        retrace_max_dq = max(
            retrace_max_dq,
            step,
        )

        if (
            step
            > config.max_contact_joint_step_rad
        ):
            raise EpisodeStop(
                "Retrazado del descenso discontinuo "
                f"en punto {index}: "
                f"{step:.4f}rad > "
                f"{config.max_contact_joint_step_rad:.4f}rad"
            )

        retrace_previous = value.copy()

    return_q = (
        [
            value.copy()
            for value in post_release_retrace_q
        ]
        + reverse_unique(q_to_pre_place)
        + reverse_unique(q_place_entry)
        + reverse_unique(transport_q)
        + [
            value.copy()
            for value in reversed(
                q_handoff
            )
        ]
        + [q_lift[-1].copy()]
        + [
            value.copy()
            for value in return_object_safe_q
        ]
        + [contact_plan["q_object_safe"].copy()]
        + reverse_unique(approach_q)
        + [q_home.copy()]
    )

    return_policy = {
        "policy": (
            "reverse_exact_validated_"
            "vla_place_descent"
        ),
        "q_retreat_full7d_executed": False,
        "q_retreat_full7d_retained_for_metrics": True,
        "post_release_retrace_points": len(
            post_release_retrace_q
        ),
        "post_release_retrace_max_dq_rad": (
            retrace_max_dq
        ),
        "reason": (
            "avoid_joining_independently_"
            "rotated_joint_branches"
        ),
    }

    print(
        f"points={len(post_release_retrace_q)} "
        f"max_dq={retrace_max_dq:.4f}rad "
        "q_retreat_full7d_executed=false",
        flush=True,
    )

    preclose_validation = (
        _one_shot_validate_joint_chain(
            q_start=q_home,
            q_targets=preclose_q,
            label="one_shot_preclose",
        )
    )

    held_validation = (
        _one_shot_validate_joint_chain(
            q_start=preclose_q[-1],
            q_targets=held_q,
            label="one_shot_held_object",
        )
    )

    return_validation = (
        _one_shot_validate_joint_chain(
            q_start=held_q[-1],
            q_targets=return_q,
            label="one_shot_return",
        )
    )

    preplan = {
        "schema": "pi0_preplan",
        "timestamp": datetime.now().isoformat(),
        "instruction": config.instruction,
        "architecture": (
            "five_phase_preinferences_"
            "single_visual_context_"
            "continuous_execution"
        ),
        "remote_inference_count": 5,
        "inference_during_motion": False,
        "inference_while_holding_object": False,
        "image_path": observation["image_path"],
        "scene_path": observation["scene_path"],
        "raw_path": observation["raw_path"],
        "home_q": q_home,
        "home_tcp": tcp_home,
        "home_error_rad": home_error,
        "snapshot_error_m": snapshot_error,
        "entities": entities,
        "raw_actions": raw_actions,
        "approach_q": approach_q,
        "q_to_pre_pick": q_to_pre_pick,
        "q_pick": q_pick,
        "q_lift": q_lift,
        "q_handoff": q_handoff,
        "local_lift_tcp": local_lift_tcp,
        "transport_start_tcp": (
            transport_start_tcp
        ),
        "transport_q": transport_q,
        "q_place_entry": q_place_entry,
        "place_start_q": place_start_q,
        "place_start_tcp": place_start_tcp,
        "q_to_pre_place": q_to_pre_place,
        "q_place": q_place,
        "q_retreat": q_retreat,
        "preclose_q": preclose_q,
        "held_q": held_q,
        "return_q": return_q,
        "return_policy": return_policy,
        "selected_target_xyz_m": selected_target,
        "pre_place_xyz_m": pre_place_xyz,
        "place_xyz_m": place_xyz,
        "close_event": close_event,
        "release_event": release_event,
        "reports": {
            "approach_translation": (
                approach_translation_report
            ),
            "approach_translation_ik": (
                approach_translation_ik
            ),
            "approach_rotation": (
                approach_rotation_report
            ),
            "contact": contact_report,
            "pick_rotation": (
                pick_rotation_report
            ),
            "lift_rotation": (
                lift_rotation_report
            ),
            "transport_handoff": (
                transport_handoff_report
            ),
            "return_object_safe_bridge": (
                return_object_safe_report
            ),
            "transport_translation": (
                transport_translation_report
            ),
            "transport_translation_ik": (
                transport_translation_ik
            ),
            "transport_rotation": (
                transport_rotation_report
            ),
            "place_entry_handoff": (
                place_entry_report
            ),
            "place": place_report,
            "place_rotation": (
                place_rotation_report
            ),
            "retreat_rotation": (
                retreat_rotation_report
            ),
            "position_contracts": {
                "approach": (
                    approach_position_contract
                ),
                "pick": (
                    pick_position_contract
                ),
                "lift": (
                    lift_position_contract
                ),
                "transport": (
                    transport_position_contract
                ),
                "place": (
                    place_position_contract
                ),
            },
            "chain_validation": {
                "preclose": preclose_validation,
                "held": held_validation,
                "return": return_validation,
            },
        },
    }

    json_write(
        run_dir / "episode_preplan.json",
        _one_shot_jsonable(preplan),
    )

    print(
        "remote_inferences=5 "
        f"preclose_points={len(preclose_q)} "
        f"held_points={len(held_q)} "
        f"return_points={len(return_q)} "
        f"target={selected_target.tolist()}",
        flush=True,
    )

    return preplan










def check_mode(
    run_dir: Path,
    config: Config,
) -> dict[str, Any]:
    preplan = _build_one_shot_preplan(
        run_dir=run_dir,
        config=config,
    )

    report = {
        "schema": (
            "pi0_precheck"
        ),
        "timestamp": (
            datetime.now().isoformat()
        ),
        "mode": "check",
        "instruction": config.instruction,
        "architecture": preplan[
            "architecture"
        ],
        "remote_inference_count": 5,
        "inference_during_motion": False,
        "full_episode_prevalidated": True,
        "selected_target_xyz_m": (
            preplan[
                "selected_target_xyz_m"
            ].tolist()
        ),
        "chain_validation": (
            preplan["reports"][
                "chain_validation"
            ]
        ),
        "close_event": preplan[
            "close_event"
        ],
        "release_event": preplan[
            "release_event"
        ],
        "robot_motion": False,
        "gripper_motion": False,
        "ready_for_real_episode": True,
    }

    json_write(
        run_dir / "episode_check.json",
        _one_shot_jsonable(report),
    )

    return report



FULL7D_INTEGRATION_MARKER = "PI0_7D_EXECUTION"


def require_full7d_real_enabled() -> None:
    value = os.environ.get(
        "VLA_PI0_FULL7D_ENABLE",
        "",
    ).strip().upper()

    if value != "YES":
        raise EpisodeStop(
            "Falta VLA_PI0_FULL7D_ENABLE=YES para ejecutar "
            "el episodio real full-7D"
        )


def full7d_rotation_limit_deg() -> float:
    value = float(
        os.environ.get(
            "VLA_PI0_MAX_ROTATION_STEP_DEG",
            "3.0",
        )
    )

    if not math.isfinite(value) or not (0.1 <= value <= 5.0):
        raise EpisodeStop(
            "VLA_PI0_MAX_ROTATION_STEP_DEG debe estar "
            "entre 0.1 y 5.0"
        )

    return value


def full7d_gripper_threshold() -> float:
    value = float(
        os.environ.get(
            "VLA_PI0_GRIPPER_EVENT_THRESHOLD",
            "0.50",
        )
    )

    if not math.isfinite(value) or not (0.20 <= value <= 0.95):
        raise EpisodeStop(
            "VLA_PI0_GRIPPER_EVENT_THRESHOLD debe estar "
            "entre 0.20 y 0.95"
        )

    return value


def split_chain_at_target(
    q_targets: list[np.ndarray],
    q_target: np.ndarray,
    label: str,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    target = finite_vector(
        q_target,
        6,
        f"{label}.target",
    )

    vectors = [
        finite_vector(
            value,
            6,
            f"{label}.q[{index}]",
        )
        for index, value in enumerate(q_targets)
    ]

    matches = [
        index
        for index, value in enumerate(vectors)
        if float(
            np.max(
                np.abs(
                    value - target
                )
            )
        ) <= 1e-7
    ]

    if len(matches) != 1:
        raise EpisodeStop(
            f"{label}: target no es único en cadena: "
            f"matches={matches}"
        )

    index = matches[0]

    return (
        [
            value.copy()
            for value in vectors[: index + 1]
        ],
        [
            value.copy()
            for value in vectors[index + 1 :]
        ],
        index,
    )


def verify_observation_snapshot(
    observation: dict[str, Any],
    tcp_actual: np.ndarray,
    label: str,
) -> float:
    snapshot_tcp = finite_vector(
        observation[
            "measured_snapshot"
        ][
            "tcp_position_base_m"
        ],
        3,
        f"{label}.snapshot_tcp",
    )

    error = float(
        np.linalg.norm(
            finite_vector(
                tcp_actual,
                3,
                f"{label}.tcp_actual",
            )
            - snapshot_tcp
        )
    )

    if error > 0.005:
        raise EpisodeStop(
            f"{label}: robot cambió entre inferencia y ejecución: "
            f"{error:.4f} m"
        )

    return error


def apply_full7d_rotation(
    q_start: np.ndarray,
    tcp_start: np.ndarray,
    q_reference_targets: list[np.ndarray],
    raw_actions: np.ndarray,
    config: Config,
    phase: str,
    max_joint_step_rad: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    try:
        return full7d.reorient_chain(
            q_start=q_start,
            tcp_start=tcp_start,
            q_reference_targets=q_reference_targets,
            raw_actions=raw_actions,
            chunk_steps=config.chunk_steps,
            model_to_base=MODEL_TO_BASE,
            max_rotation_step_deg=(
                full7d_rotation_limit_deg()
            ),
            max_tilt_deg=config.max_tilt_deg,
            max_fk_error_m=config.max_fk_error_m,
            max_joint_step_rad=max_joint_step_rad,
            target_axis=TARGET_AXIS,
            phase=phase,
        )
    except full7d.Full7DValidationError as error:
        raise EpisodeStop(
            f"{phase}: validación de rotación π0 falló: {error}"
        ) from error


def decode_full7d_gripper(
    raw_actions: np.ndarray,
    expected_event: str,
    config: Config,
    phase: str,
) -> dict[str, Any]:
    try:
        return full7d.decode_gripper_event(
            raw_actions=raw_actions,
            chunk_steps=config.chunk_steps,
            expected_event=expected_event,
            threshold=full7d_gripper_threshold(),
            close_width_m=config.close_width_m,
            open_width_m=config.open_width_m,
            close_hold_s=config.gripper_hold_s,
        )
    except full7d.Full7DValidationError as error:
        raise EpisodeStop(
            f"{phase}: validación de garra π0 falló: {error}"
        ) from error


def _load_checked_preplan(config: Config) -> dict[str, Any]:
    """Carga y comprueba el preplan antes de ejecutarlo."""
    import hashlib, json

    raw=os.environ.get(
        "VLA_EXECUTE_PREPLAN_PATH",""
    ).strip()

    expected=os.environ.get(
        "VLA_EXECUTE_PREPLAN_SHA256",""
    ).strip().lower()

    if not raw or not expected:
        raise EpisodeStop(
            "REAL requiere ruta y SHA256 "
            "del preplan validado"
        )

    path=Path(raw).resolve()

    if not path.is_file():
        raise EpisodeStop(
            f"Preplan inexistente: {path}"
        )

    payload=path.read_bytes()
    observed=hashlib.sha256(
        payload
    ).hexdigest()

    if observed!=expected:
        raise EpisodeStop(
            "SHA256 distinto: "
            f"expected={expected}, "
            f"observed={observed}"
        )

    check_path=(
        path.parent/"episode_check.json"
    )

    if not check_path.is_file():
        raise EpisodeStop(
            f"Falta {check_path}"
        )

    plan=json.loads(
        payload.decode("utf-8")
    )

    check=json.loads(
        check_path.read_text(
            encoding="utf-8"
        )
    )

    if plan.get("schema") != (
        "pi0_preplan"
    ):
        raise EpisodeStop(
            "Schema de preplan no autorizado"
        )

    if plan.get("instruction") != (
        config.instruction
    ):
        raise EpisodeStop(
            "La instruccion no coincide "
            "con el preplan"
        )

    required={
        "mode":"check",
        "instruction":config.instruction,
        "full_episode_prevalidated":True,
        "ready_for_real_episode":True,
        "robot_motion":False,
        "gripper_motion":False,
    }

    if any(
        check.get(key)!=value
        for key,value in required.items()
    ):
        raise EpisodeStop(
            "episode_check.json "
            "no autoriza ejecucion"
        )

    for key,size in (
        ("home_q",6),
        ("home_tcp",3),
        ("place_xyz_m",3),
    ):
        plan[key]=finite_vector(
            plan[key],
            size,
            f"loaded.{key}",
        )

    for key in (
        "preclose_q",
        "held_q",
        "return_q",
    ):
        values=plan.get(key)

        if (
            not isinstance(values,list)
            or not values
        ):
            raise EpisodeStop(
                f"{key}: cadena ausente "
                "o vacia"
            )

        plan[key]=[
            finite_vector(
                value,
                6,
                f"loaded.{key}[{index}]",
            )
            for index,value
            in enumerate(values)
        ]

    _one_shot_validate_joint_chain(
        plan["home_q"],
        plan["preclose_q"],
        "loaded_preclose",
    )

    _one_shot_validate_joint_chain(
        plan["preclose_q"][-1],
        plan["held_q"],
        "loaded_held",
    )

    _one_shot_validate_joint_chain(
        plan["held_q"][-1],
        plan["return_q"],
        "loaded_return",
    )

    for name in (
        "close_event",
        "release_event",
    ):
        event=plan.get(name)

        if not isinstance(event,dict):
            raise EpisodeStop(
                f"{name}: ausente"
            )

        width=float(event["width_m"])
        hold=float(event["hold_s"])

        if (
            not 0.0 < width <= 0.12
            or not 0.5 <= hold <= 8.0
        ):
            raise EpisodeStop(
                f"{name}: invalido"
            )

    print(
        f"path={path} "
        f"sha256={observed} "
        f"preclose="
        f"{len(plan['preclose_q'])} "
        f"held={len(plan['held_q'])} "
        f"return={len(plan['return_q'])}",
        flush=True,
    )

    return plan


def execute_real_episode(
    run_dir: Path,
    config: Config,
) -> dict[str, Any]:
    require_full7d_real_enabled()
    preflight_real(config)

    gripper_closed = False
    object_released = False
    place_verified = False
    events: list[dict[str, Any]] = []

    preplan = _load_checked_preplan(config)

    try:
        with robot_node() as node:
            require_single_gripper_subscriber(node)

            q_start, tcp_start = (
                current_robot_state(node)
            )

            q_start_error = float(
                np.max(
                    np.abs(
                        q_start
                        - preplan["home_q"]
                    )
                )
            )

            tcp_start_error = float(
                np.linalg.norm(
                    tcp_start
                    - preplan["home_tcp"]
                )
            )

            if (
                q_start_error > 0.02
                or tcp_start_error > 0.005
            ):
                raise EpisodeStop(
                    "El robot cambio despues del "
                    "preplan: "
                    f"q_error={q_start_error:.4f} rad, "
                    f"tcp_error={tcp_start_error:.4f} m"
                )

            node.set_gripper_width(
                config.open_width_m,
                wait_sec=2.5,
            )

            print(
                "No habra inferencias durante el movimiento.",
                flush=True,
            )

            preclose_execution = (
                send_joint_sequence(
                    node=node,
                    q_start_expected=q_start,
                    q_targets=preplan[
                        "preclose_q"
                    ],
                    config=config,
                    label=(
                        "contact_continuous_preclose"
                    ),
                )
            )

            q_pick_actual, tcp_pick_actual = (
                current_robot_state(node)
            )

            pick_error = float(
                np.linalg.norm(
                    tcp_pick_actual
                    - finite_vector(
                        preplan["entities"][
                            "pick_xyz_m"
                        ],
                        3,
                        "one_shot.pick_xyz",
                    )
                )
            )

            if (
                pick_error
                > config.pick_actual_error_m
            ):
                raise EpisodeStop(
                    "ONE_SHOT: error real de pick "
                    f"{pick_error:.4f} m"
                )

            node.set_gripper_width(
                preplan["close_event"][
                    "width_m"
                ],
                wait_sec=preplan[
                    "close_event"
                ]["hold_s"],
            )

            gripper_closed = True

            held_execution = (
                send_joint_sequence(
                    node=node,
                    q_start_expected=(
                        q_pick_actual
                    ),
                    q_targets=preplan["held_q"],
                    config=config,
                    label=(
                        "contact_continuous_held_object"
                    ),
                )
            )

            q_place_actual, tcp_place_actual = (
                current_robot_state(node)
            )

            place_error = float(
                np.linalg.norm(
                    tcp_place_actual
                    - preplan["place_xyz_m"]
                )
            )

            if (
                place_error
                > config.place_actual_error_m
            ):
                raise EpisodeStop(
                    "ONE_SHOT: error real de place "
                    f"{place_error:.4f} m"
                )

            node.set_gripper_width(
                preplan["release_event"][
                    "width_m"
                ],
                wait_sec=preplan[
                    "release_event"
                ]["hold_s"],
            )

            object_released = True
            gripper_closed = False

            return_execution = (
                send_joint_sequence(
                    node=node,
                    q_start_expected=(
                        q_place_actual
                    ),
                    q_targets=preplan["return_q"],
                    config=config,
                    label=(
                        "return_continuous_after_release"
                    ),
                )
            )

            result_q, result_tcp = (
                current_robot_state(node)
            )

        result_home_error = float(
            np.max(
                np.abs(
                    result_q
                    - preplan["home_q"]
                )
            )
        )

        if result_home_error > 0.05:
            raise EpisodeStop(
                "ONE_SHOT: no se volvio a home: "
                f"{result_home_error:.4f} rad"
            )

        (
            verify_scene_path,
            verify_scene,
            verify_image_path,
            _,
        ) = request_fresh_scene(
            run_dir=run_dir,
            instruction=config.instruction,
            index=2,
            label="place_verification",
        )

        verify_entities = resolve_entities(
            verify_scene,
            config,
        )

        observed_object = finite_vector(
            verify_entities["object_xyz_m"],
            3,
            "one_shot.verify_object",
        )

        observed_target = finite_vector(
            verify_entities["target_xyz_m"],
            3,
            "one_shot.verify_target",
        )

        verify_xy_error = float(
            np.linalg.norm(
                observed_object[:2]
                - observed_target[:2]
            )
        )

        verify_z_error = float(
            abs(
                observed_object[2]
                - observed_target[2]
            )
        )

        place_verified = bool(
            verify_xy_error <= 0.080
            and verify_z_error <= 0.080
        )

        events.extend(
            [
                {
                    "phase": "preclose_continuous",
                    "execution": preclose_execution,
                    "pick_actual_xyz_m": (
                        tcp_pick_actual.tolist()
                    ),
                    "pick_error_m": pick_error,
                },
                {
                    "phase": "grasp",
                    "event": preplan[
                        "close_event"
                    ],
                },
                {
                    "phase": (
                        "held_object_continuous"
                    ),
                    "execution": held_execution,
                    "place_actual_xyz_m": (
                        tcp_place_actual.tolist()
                    ),
                    "place_error_m": place_error,
                },
                {
                    "phase": "release",
                    "event": preplan[
                        "release_event"
                    ],
                },
                {
                    "phase": "return_home",
                    "execution": return_execution,
                    "result_q_rad": result_q.tolist(),
                    "result_tcp_m": result_tcp.tolist(),
                    "result_home_error_rad": (
                        result_home_error
                    ),
                },
                {
                    "phase": "place_verification",
                    "scene_path": str(
                        verify_scene_path
                    ),
                    "image_path": str(
                        verify_image_path
                    ),
                    "object_xyz_m": (
                        observed_object.tolist()
                    ),
                    "target_xyz_m": (
                        observed_target.tolist()
                    ),
                    "xy_error_m": verify_xy_error,
                    "z_error_m": verify_z_error,
                    "verified": place_verified,
                },
            ]
        )

        if not place_verified:
            raise EpisodeStop(
                "ONE_SHOT termino y volvio a home, "
                "pero Photoneo no confirmo el place"
            )

        report = {
            "schema": (
                "pi0_preinferred"
            ),
            "timestamp": (
                datetime.now().isoformat()
            ),
            "mode": "real",
            "instruction": config.instruction,
            "result": "SUCCESS",
            "architecture": preplan[
                "architecture"
            ],
            "remote_inference_count": 5,
            "inference_during_motion": False,
            "inference_while_holding_object": False,
            "pi0_translation_applied": True,
            "pi0_rotation_applied": True,
            "pi0_close_event_applied": True,
            "release_source": preplan[
                "release_event"
            ]["source"],
            "object_released": object_released,
            "place_verified": place_verified,
            "events": events,
        }

        json_write(
            run_dir / "episode_result.json",
            _one_shot_jsonable(report),
        )

        return report

    except Exception as error:
        failure = {
            "schema": (
                "pi0_preinferred_failure"
            ),
            "timestamp": (
                datetime.now().isoformat()
            ),
            "mode": config.mode,
            "instruction": config.instruction,
            "error_type": (
                type(error).__name__
            ),
            "error": str(error),
            "traceback": traceback.format_exc(),
            "events": events,
            "gripper_closed": gripper_closed,
            "object_released": object_released,
            "place_verified": place_verified,
            "safety_action": (
                "hold_position_with_object"
                if gripper_closed
                and not object_released
                else "no_object_held"
            ),
        }

        json_write(
            run_dir / "episode_failure.json",
            _one_shot_jsonable(failure),
        )

        raise




def main() -> int:
    args = parse_arguments()
    config = build_config(args)
    run_dir = make_run_dir()

    json_write(
        run_dir / "episode_config.json",
        {
            **config.__dict__,
            "mapping_model_to_base": (
                MODEL_TO_BASE.tolist()
            ),
            "rotation_mapping_resolved": True,
            "pi0_rotation_applied_in_real": (
                config.mode == "real"
            ),
            "pi0_gripper_event_applied_in_real": (
                config.mode == "real"
            ),
            "full7d_marker": (
                FULL7D_INTEGRATION_MARKER
            ),
        },
    )

    print("Supervisor de episodio")
    print("mode:", config.mode)
    print("instruction:", config.instruction)
    print("run_dir:", run_dir)
    print(
        "architecture: "
        "pi0 temporal full-7D + "
        "geometric safety supervisor"
    )
    print("rotation_mapping_resolved: true")
    print(
        "pi0_rotation_applied_in_real:",
        config.mode == "real",
    )
    print(
        "pi0_gripper_event_applied_in_real:",
        config.mode == "real",
    )

    if config.mode == "check":
        report = check_mode(
            run_dir,
            config,
        )

        print("Comprobación completada")
        print(
            "Captura, inferencia, "
            "chunk, IK de contacto y place "
            "validados."
        )
        print("robot_motion=false")
        print("gripper_motion=false")
        print(
            "Informe:",
            run_dir / "episode_check.json",
        )
        return 0

    report = execute_real_episode(
        run_dir,
        config,
    )

    print("Episodio completado")
    print("pick=true")
    print("place=true")
    print("release=true")
    print("returned_global_home=true")
    print(
        "Informe:",
        run_dir / "episode_result.json",
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EpisodeStop as error:
        print(
            f"{error}",
            file=sys.stderr,
        )
        raise SystemExit(2)
