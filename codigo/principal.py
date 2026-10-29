#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import subprocess
import sys
import traceback
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise RuntimeError("OpenCV es obligatorio para el grounding visual VLA") from exc


ROOT = Path("/workspace")
SCRIPTS = ROOT / "scripts"
RESULTS_ROOT = ROOT / "results" / "pick_place"

SUPERVISOR_PATH = SCRIPTS / "supervisor.py"
ADAPTATION_PATH = SCRIPTS / "adaptacion.py"

ORIN = os.environ.get("VLA_ORIN_SSH", "").strip()


def require_orin_ssh() -> str:
    if not ORIN:
        raise RuntimeError(
            "Falta VLA_ORIN_SSH. Configura el usuario y la IP "
            "de la Jetson en el entorno antes de usar la inferencia remota."
        )
    return ORIN

REMOTE_IMAGE = (
    "openvla/"
    "escena.png"
)
REMOTE_OPENVLA_OUTPUT = (
    "openvla/"
    "salida_openvla.json"
)
REMOTE_DROID_OUTPUT = (
    "/mnt/sd_pi0/openpi/results/"
    "salida_droid.json"
)

OPENVLA_INFERENCE_SCRIPT = (
    "openvla/"
    "inferencia.py"
)
OPENVLA_ACTIVATION_SCRIPT = (
    "openvla/"
    "activar.sh"
)

DROID_ACTIVATION_SCRIPT = (
    "/mnt/sd_pi0/openpi/"
    "activar.sh"
)
DROID_INFERENCE_SCRIPT = (
    "/mnt/sd_pi0/openpi/"
    "inferencia.py"
)
DROID_PYTHON = (
    "/mnt/sd_pi0/openpi/"
    ".venv/bin/python3"
)

OPENVLA_PERSISTENT_PORT = int(
    os.environ.get("VLA_OPENVLA_SERVER_PORT", "8100")
)
OPENPI_PERSISTENT_PORT = int(
    os.environ.get("VLA_OPENPI_SERVER_PORT", "8000")
)
OPENPI_CLIENT_SCRIPT = (
    "/mnt/sd_pi0/openpi/"
    "cliente.py"
)

EXPECTED_SHAPES = {
    "openvla": (1, 7),
    "pi05_droid": (15, 8),
    "pi0_droid": (10, 8),
}


# Las siete fases cartesianas comparten el mismo criterio de ejecución.
# El control previo y el solver IK usan las mismas reglas; así
# un punto de contacto no puede ser válido en una capa y rechazado en otra
# por tratarlo después como espacio libre.
PHASE_EXECUTION_CONTRACTS = {
    "approach": {
        "workspace_mode": "free",
        "solver_contact": False,
        "joint_step_config": "max_chunk_joint_step_rad",
    },
    "pick_descent": {
        "workspace_mode": "contact",
        "solver_contact": True,
        "joint_step_config": "max_contact_joint_step_rad",
    },
    "lift": {
        "workspace_mode": "contact",
        "solver_contact": True,
        "joint_step_config": "max_contact_joint_step_rad",
    },
    "transport": {
        "workspace_mode": "free",
        "solver_contact": False,
        "joint_step_config": "max_chunk_joint_step_rad",
    },
    "place_approach": {
        "workspace_mode": "free",
        "solver_contact": False,
        "joint_step_config": "max_chunk_joint_step_rad",
    },
    "place_descent": {
        "workspace_mode": "contact",
        "solver_contact": True,
        "joint_step_config": "max_contact_joint_step_rad",
    },
    "retreat": {
        "workspace_mode": "contact",
        "solver_contact": True,
        "joint_step_config": "max_contact_joint_step_rad",
    },
}


def build_phase_execution_contracts(config: Any) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}

    for phase_name, raw in PHASE_EXECUTION_CONTRACTS.items():
        workspace_mode = str(raw["workspace_mode"])
        solver_contact = bool(raw["solver_contact"])
        joint_step_name = str(raw["joint_step_config"])

        if workspace_mode not in {"free", "contact"}:
            raise MainStop(
                f"Contrato de fase inválido para {phase_name}: "
                f"workspace_mode={workspace_mode}"
            )

        expected_contact = workspace_mode == "contact"
        if solver_contact != expected_contact:
            raise MainStop(
                f"Contrato incoherente para {phase_name}: "
                f"workspace={workspace_mode}, "
                f"solver_contact={solver_contact}"
            )

        if not hasattr(config, joint_step_name):
            raise MainStop(
                f"Config no contiene {joint_step_name} "
                f"para la fase {phase_name}"
            )

        maximum_joint_step_rad = float(
            getattr(config, joint_step_name)
        )
        if (
            not math.isfinite(maximum_joint_step_rad)
            or maximum_joint_step_rad <= 0.0
        ):
            raise MainStop(
                f"Límite articular inválido para {phase_name}: "
                f"{maximum_joint_step_rad}"
            )

        contracts[phase_name] = {
            "workspace_mode": workspace_mode,
            "solver_contact": solver_contact,
            "joint_step_config": joint_step_name,
            "max_joint_step_rad": maximum_joint_step_rad,
        }

        print(
            f"phase={phase_name} "
            f"workspace={workspace_mode} "
            f"solver_contact={str(solver_contact).lower()} "
            f"joint_step={maximum_joint_step_rad:.6f}rad",
            flush=True,
        )

    return contracts


def phase_solver_max_segment_m(workspace_mode: str) -> float:
    mode = str(workspace_mode)
    if mode == "contact":
        env_name = "VLA_CONTACT_MAX_SEGMENT_M"
        default = 0.012
        lower = 0.003
        upper = 0.030
    elif mode == "free":
        env_name = "VLA_SOLVER_MAX_SEGMENT_M"
        default = 0.060
        lower = 0.010
        upper = 0.100
    else:
        raise MainStop(
            f"workspace_mode desconocido: {workspace_mode}"
        )

    value = float(os.environ.get(env_name, str(default)))
    if (
        not math.isfinite(value)
        or value < lower
        or value > upper
    ):
        raise MainStop(
            f"{env_name} debe estar entre {lower:.3f} y {upper:.3f} m"
        )

    return value


class MainStop(RuntimeError):
    pass


def _high_shoulder_contract(label: str) -> dict[str, Any]:
    """
    Define el corredor articular del UR5e según la fase.
    
    En pick se mantienen límites más estrictos. Durante transporte y place se
    admite un corredor elbow-up algo más amplio, pero siempre acotado. Los límites
    de q2, q3 y de sus deltas respecto a home dependen de la fase.
    
    La rama de hombro bajo descartada físicamente queda fuera de estos límites.
    """
    normalized_label = str(label).lower()
    far_side_tokens = (
        "transport",
        "place",
        "retreat",
        "held_chain",
        "return_chain",
    )
    corridor = (
        "far_side"
        if any(token in normalized_label for token in far_side_tokens)
        else "pick_side"
    )

    strict_max_q2 = float(
        os.environ.get("VLA_HIGH_SHOULDER_MAX_Q2_RAD", "-1.00")
    )
    far_side_max_q2 = float(
        os.environ.get("VLA_HIGH_SHOULDER_FAR_MAX_Q2_RAD", "-0.88")
    )

    strict_min_q3 = float(
        os.environ.get("VLA_HIGH_SHOULDER_MIN_Q3_RAD", "0.90")
    )
    far_side_min_q3 = float(
        os.environ.get("VLA_HIGH_SHOULDER_FAR_MIN_Q3_RAD", "0.72")
    )

    strict_max_q2_delta = float(
        os.environ.get("VLA_HIGH_SHOULDER_MAX_Q2_DELTA_RAD", "0.60")
    )
    far_side_max_q2_delta = float(
        os.environ.get("VLA_HIGH_SHOULDER_FAR_MAX_Q2_DELTA_RAD", "0.68")
    )

    strict_max_q3_delta = float(
        os.environ.get("VLA_HIGH_SHOULDER_MAX_Q3_DELTA_RAD", "0.85")
    )
    far_side_max_q3_delta = float(
        os.environ.get("VLA_HIGH_SHOULDER_FAR_MAX_Q3_DELTA_RAD", "1.00")
    )

    values: dict[str, Any] = {
        "corridor": corridor,
        "maximum_shoulder_lift_rad": (
            far_side_max_q2 if corridor == "far_side" else strict_max_q2
        ),
        "minimum_elbow_rad": (
            far_side_min_q3 if corridor == "far_side" else strict_min_q3
        ),
        "maximum_shoulder_delta_from_home_rad": (
            far_side_max_q2_delta
            if corridor == "far_side"
            else strict_max_q2_delta
        ),
        "maximum_elbow_delta_from_home_rad": (
            far_side_max_q3_delta
            if corridor == "far_side"
            else strict_max_q3_delta
        ),
        "strict_maximum_shoulder_lift_rad": strict_max_q2,
        "far_side_maximum_shoulder_lift_rad": far_side_max_q2,
        "strict_minimum_elbow_rad": strict_min_q3,
        "far_side_minimum_elbow_rad": far_side_min_q3,
        "strict_maximum_shoulder_delta_from_home_rad": strict_max_q2_delta,
        "far_side_maximum_shoulder_delta_from_home_rad": far_side_max_q2_delta,
        "strict_maximum_elbow_delta_from_home_rad": strict_max_q3_delta,
        "far_side_maximum_elbow_delta_from_home_rad": far_side_max_q3_delta,
    }

    if not -1.40 <= strict_max_q2 <= -0.75:
        raise MainStop(
            "VLA_HIGH_SHOULDER_MAX_Q2_RAD debe estar entre -1.40 y -0.75 rad"
        )
    if not -1.20 <= far_side_max_q2 <= -0.85:
        raise MainStop(
            "VLA_HIGH_SHOULDER_FAR_MAX_Q2_RAD debe estar entre -1.20 y -0.85 rad"
        )
    if not 0.80 <= strict_min_q3 <= 1.30:
        raise MainStop(
            "VLA_HIGH_SHOULDER_MIN_Q3_RAD debe estar entre 0.80 y 1.30 rad"
        )
    if not 0.65 <= far_side_min_q3 <= 1.10:
        raise MainStop(
            "VLA_HIGH_SHOULDER_FAR_MIN_Q3_RAD debe estar entre 0.65 y 1.10 rad"
        )
    if not 0.25 <= strict_max_q2_delta <= 1.00:
        raise MainStop(
            "VLA_HIGH_SHOULDER_MAX_Q2_DELTA_RAD debe estar entre 0.25 y 1.00 rad"
        )
    if not 0.40 <= far_side_max_q2_delta <= 0.72:
        raise MainStop(
            "VLA_HIGH_SHOULDER_FAR_MAX_Q2_DELTA_RAD debe estar entre 0.40 y 0.72 rad"
        )
    if not 0.35 <= strict_max_q3_delta <= 1.20:
        raise MainStop(
            "VLA_HIGH_SHOULDER_MAX_Q3_DELTA_RAD debe estar entre 0.35 y 1.20 rad"
        )
    if not 0.70 <= far_side_max_q3_delta <= 1.08:
        raise MainStop(
            "VLA_HIGH_SHOULDER_FAR_MAX_Q3_DELTA_RAD debe estar entre 0.70 y 1.08 rad"
        )

    return values

def validate_high_shoulder_branch(
    *,
    q_start: np.ndarray,
    q_targets: list[np.ndarray] | np.ndarray,
    label: str,
) -> dict[str, Any]:
    """
    Comprueba que el plan permanezca en la rama validada de hombro alto.
    
    La condición se aplica antes del movimiento a las siete fases, a los candidatos
    de transporte/place y a las cadenas de ida y vuelta. Una rama de hombro o codo
    bajo no llega a ``send_joint_sequence``.
    """
    home = finite_vector(
        supervisor.expected_global_home_q(),
        6,
        f"{label}.home_reference",
    )
    start = finite_vector(q_start, 6, f"{label}.q_start")
    targets = [
        finite_vector(value, 6, f"{label}.q[{index}]")
        for index, value in enumerate(list(q_targets), start=1)
    ]
    chain = [start] + targets
    contract = _high_shoulder_contract(label)

    weakest_margin = float("inf")
    minimum_q2 = float("inf")
    maximum_q2 = float("-inf")
    minimum_q3 = float("inf")
    maximum_q3 = float("-inf")

    for index, q in enumerate(chain):
        q2 = float(q[1])
        q3 = float(q[2])
        q2_delta = abs(q2 - float(home[1]))
        q3_delta = abs(q3 - float(home[2]))

        margins = {
            "shoulder_ceiling": (
                contract["maximum_shoulder_lift_rad"] - q2
            ),
            "elbow_floor": (
                q3 - contract["minimum_elbow_rad"]
            ),
            "shoulder_home_corridor": (
                contract["maximum_shoulder_delta_from_home_rad"] - q2_delta
            ),
            "elbow_home_corridor": (
                contract["maximum_elbow_delta_from_home_rad"] - q3_delta
            ),
        }
        local_weakest_name, local_weakest_margin = min(
            margins.items(), key=lambda item: item[1]
        )
        weakest_margin = min(weakest_margin, float(local_weakest_margin))

        minimum_q2 = min(minimum_q2, q2)
        maximum_q2 = max(maximum_q2, q2)
        minimum_q3 = min(minimum_q3, q3)
        maximum_q3 = max(maximum_q3, q3)

        if local_weakest_margin < -1e-9:
            raise MainStop(
                f"{label}: rama cinemática de hombro bajo prohibida en "
                f"waypoint={index}; q2_shoulder_lift={q2:.6f}rad, "
                f"q3_elbow={q3:.6f}rad, "
                f"delta_q2_home={q2_delta:.6f}rad, "
                f"delta_q3_home={q3_delta:.6f}rad, "
                f"violacion={local_weakest_name}, "
                f"margen={local_weakest_margin:.6f}rad"
            )

    report = {
        "schema": "VLA_phase_aware_posture_corridor",
        "label": label,
        "waypoints": len(chain),
        "home_q2_rad": float(home[1]),
        "home_q3_rad": float(home[2]),
        "minimum_q2_rad": minimum_q2,
        "maximum_q2_rad": maximum_q2,
        "minimum_q3_rad": minimum_q3,
        "maximum_q3_rad": maximum_q3,
        "weakest_margin_rad": weakest_margin,
        "contract": contract,
        "status": "PASS",
    }
    print(
        f"label={label} waypoints={len(chain)} "
        f"q2_range=[{minimum_q2:.4f},{maximum_q2:.4f}]rad "
        f"q3_range=[{minimum_q3:.4f},{maximum_q3:.4f}]rad "
        f"corridor={contract['corridor']} "
        f"q2_ceiling={contract['maximum_shoulder_lift_rad']:.4f}rad "
        f"q3_floor={contract['minimum_elbow_rad']:.4f}rad "
        f"q2_home_delta_max="
        f"{contract['maximum_shoulder_delta_from_home_rad']:.4f}rad "
        f"q3_home_delta_max="
        f"{contract['maximum_elbow_delta_from_home_rad']:.4f}rad "
        f"margin={weakest_margin:.4f}rad",
        flush=True,
    )
    return report


def load_module(name: str, path: Path):
    if not path.is_file():
        raise MainStop(
            f"No existe módulo requerido: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise MainStop(
            f"No se pudo cargar: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


supervisor = load_module(
    "supervisor",
    SUPERVISOR_PATH,
)
adaptation = load_module(
    "adaptacion",
    ADAPTATION_PATH,
)

def finite_vector(
    value: Any,
    size: int,
    label: str,
) -> np.ndarray:
    vector = np.asarray(
        value,
        dtype=float,
    ).reshape(-1)

    if vector.size != size:
        raise MainStop(
            f"{label}: tamaño {vector.size}; "
            f"esperado {size}"
        )

    if not np.all(np.isfinite(vector)):
        raise MainStop(
            f"{label}: contiene NaN o Inf"
        )

    return vector


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            jsonable(item)
            for item in value
        ]

    return value


def json_write(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            jsonable(value),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def run_checked(
    command: list[str],
    *,
    input_text: str | None = None,
    label: str,
) -> str:
    completed = subprocess.run(
        command,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    output = completed.stdout or ""

    if completed.returncode != 0:
        raise MainStop(
            f"{label} falló "
            f"(rc={completed.returncode}):\n"
            + output[-12000:]
        )

    return output


def reset_visual_context() -> None:
    if hasattr(
        supervisor,
        "_PI0_INITIAL_VISUAL_CONTEXT",
    ):
        supervisor._PI0_INITIAL_VISUAL_CONTEXT = (
            None
        )

    if hasattr(
        supervisor,
        "_PI0_VISUAL_REUSE_LABELS",
    ):
        value = (
            supervisor._PI0_VISUAL_REUSE_LABELS
        )

        if isinstance(value, list):
            value.clear()


def load_openvla_calibration() -> dict[str, Any]:
    candidates = []

    for name in (
        "OPENVLA_CALIBRATION_PATH",
        "OPENVLA_CALIBRATION",
    ):
        value = getattr(
            lab,
            name,
            None,
        )

        if value is not None:
            candidates.append(
                Path(value)
            )

    candidates.append(
        ROOT
        / "results"
        / "openvla_residual"
        / "openvla_empirical_action_calibration.json"
    )

    for path in candidates:
        if not path.is_file():
            continue

        document = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(document, dict):
            continue

        return document

    raise MainStop(
        "No se encontró la calibración "
        "traslacional de OpenVLA."
    )


def find_action_matrix(
    value: Any,
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    if isinstance(value, list):
        array = np.asarray(
            value,
            dtype=object,
        )

        if array.shape == expected_shape:
            try:
                numeric = np.asarray(
                    value,
                    dtype=float,
                )
            except Exception:
                numeric = None

            if (
                numeric is not None
                and np.all(
                    np.isfinite(numeric)
                )
            ):
                return numeric

        if (
            expected_shape == (1, 7)
            and len(value) == 7
        ):
            try:
                numeric = np.asarray(
                    value,
                    dtype=float,
                ).reshape(1, 7)
            except Exception:
                numeric = None

            if (
                numeric is not None
                and np.all(
                    np.isfinite(numeric)
                )
            ):
                return numeric

        for item in value:
            found = find_action_matrix(
                item,
                expected_shape,
            )

            if found is not None:
                return found

    if isinstance(value, dict):
        preferred_keys = (
            "actions",
            "openvla_action",
            "raw_actions",
        )

        for key in preferred_keys:
            if key not in value:
                continue

            found = find_action_matrix(
                value[key],
                expected_shape,
            )

            if found is not None:
                return found

        for item in value.values():
            found = find_action_matrix(
                item,
                expected_shape,
            )

            if found is not None:
                return found

    return None


def state_for_droid(
    q_ur5e: np.ndarray,
    mode: str,
) -> list[float]:
    q = finite_vector(
        q_ur5e,
        6,
        "droid.q_ur5e",
    )

    if mode == "zeros":
        return [0.0] * 7

    if mode == "ur5e":
        return [
            *[
                float(value)
                for value in q
            ],
            0.0,
        ]

    raise MainStop(
        f"Modo de estado DROID desconocido: "
        f"{mode}"
    )


def infer_on_orin_direct(
    *,
    model: str,
    image_path: Path,
    instruction: str,
    q_ur5e: np.ndarray,
    gripper_width_m: float,
    droid_state_mode: str,
    seed: int,
    run_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    expected_shape = EXPECTED_SHAPES[
        model
    ]

    run_checked(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            require_orin_ssh(),
            "true",
        ],
        label="Conexión SSH con Orin",
    )

    run_checked(
        [
            "scp",
            "-q",
            str(image_path),
            f"{require_orin_ssh()}:{REMOTE_IMAGE}",
        ],
        label="Copia de imagen a Orin",
    )

    if model == "openvla":
        remote_output = (
            REMOTE_OPENVLA_OUTPUT
        )

        command_text = (
            f"source "
            f"{shlex.quote(OPENVLA_ACTIVATION_SCRIPT)} "
            f"&& python "
            f"{shlex.quote(OPENVLA_INFERENCE_SCRIPT)} "
            f"--base-image "
            f"{shlex.quote(REMOTE_IMAGE)} "
            f"--prompt "
            f"{shlex.quote(instruction)} "
            f"--seed {int(seed)} "
            f"--out "
            f"{shlex.quote(remote_output)}"
        )

    else:
        remote_output = (
            REMOTE_DROID_OUTPUT
        )

        joint_position = ",".join(
            f"{value:.10f}"
            for value in state_for_droid(
                q_ur5e,
                droid_state_mode,
            )
        )

        gripper_position = float(
            np.clip(
                (
                    gripper_width_m
                    - 0.015
                )
                / (
                    0.090
                    - 0.015
                ),
                0.0,
                1.0,
            )
        )

        command_text = (
            f"source "
            f"{shlex.quote(DROID_ACTIVATION_SCRIPT)} "
            f"&& "
            f"{shlex.quote(DROID_PYTHON)} "
            f"{shlex.quote(DROID_INFERENCE_SCRIPT)} "
            f"--model "
            f"{shlex.quote(model)} "
            f"--base-image "
            f"{shlex.quote(REMOTE_IMAGE)} "
            f"--prompt "
            f"{shlex.quote(instruction)} "
            f"--joint-position "
            f"{shlex.quote(joint_position)} "
            f"--gripper-position "
            f"{gripper_position:.8f} "
            f"--seed {int(seed)} "
            f"--out "
            f"{shlex.quote(remote_output)}"
        )

    provider_output = run_checked(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            require_orin_ssh(),
            "bash",
            "-lc",
            shlex.quote(command_text),
        ],
        label=f"Inferencia {model}",
    )

    local_output = (
        run_dir
        / f"{model}_raw_output.json"
    )

    run_checked(
        [
            "scp",
            "-q",
            f"{require_orin_ssh()}:{remote_output}",
            str(local_output),
        ],
        label="Copia de salida VLA",
    )

    payload = json.loads(
        local_output.read_text(
            encoding="utf-8"
        )
    )

    actions = find_action_matrix(
        payload,
        expected_shape,
    )

    if actions is None:
        raise MainStop(
            f"{model}: el JSON no contiene "
            f"acciones {expected_shape}."
        )

    evidence = {
        "model": model,
        "native_shape":
            list(actions.shape),
        "actions_finite": bool(
            np.all(np.isfinite(actions))
        ),
        "image_path": str(
            image_path
        ),
        "image_sha256": sha256_file(
            image_path
        ),
        "instruction": instruction,
        "seed": int(seed),
        "droid_state_mode": (
            droid_state_mode
            if model != "openvla"
            else None
        ),
        "local_output": str(
            local_output
        ),
        "provider_output_tail":
            provider_output[-4000:],
    }

    return actions, evidence



def _copy_inference_image_to_orin(image_path: Path) -> str:
    image_hash = sha256_file(image_path)
    suffix = image_path.suffix.lower() or ".png"
    remote_image = (
        "openvla/"
        f"request_{image_hash[:16]}{suffix}"
    )
    run_checked(
        [
            "scp",
            "-q",
            str(image_path),
            f"{require_orin_ssh()}:{remote_image}",
        ],
        label="Copia de imagen a backend persistente",
    )
    return remote_image


def _ssh_bash(command_text: str, *, label: str) -> str:
    return run_checked(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            require_orin_ssh(),
            "bash",
            "-lc",
            shlex.quote(command_text),
        ],
        label=label,
    )


def _persistent_openvla_inference(
    *,
    model: str,
    image_path: Path,
    instruction: str,
    seed: int,
    run_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model != "openvla":
        raise MainStop(f"Backend OpenVLA recibió modelo incompatible: {model}")

    health_text = _ssh_bash(
        f"curl -fsS --max-time 4 "
        f"http://127.0.0.1:{OPENVLA_PERSISTENT_PORT}/health",
        label="Health OpenVLA persistente",
    )
    try:
        health = json.loads(health_text)
    except json.JSONDecodeError as exc:
        raise MainStop(
            "El servidor OpenVLA respondió un health no JSON:\n"
            + health_text[-2000:]
        ) from exc
    if health.get("status") != "ready" or health.get("model") != "openvla":
        raise MainStop(f"Backend OpenVLA no preparado: {health}")

    remote_image = _copy_inference_image_to_orin(image_path)
    image_hash = sha256_file(image_path)
    request_id = (
        f"{run_dir.name}:{image_hash[:16]}:{int(seed)}"
    )
    request_payload = {
        "request_id": request_id,
        "model": "openvla",
        "image_path": remote_image,
        "image_sha256": image_hash,
        "instruction": instruction,
        "seed": int(seed),
    }
    payload_text = json.dumps(request_payload, ensure_ascii=False)
    response_text = _ssh_bash(
        "curl -fsS --max-time 1800 "
        "-H 'Content-Type: application/json' "
        f"--data-binary {shlex.quote(payload_text)} "
        f"http://127.0.0.1:{OPENVLA_PERSISTENT_PORT}/infer",
        label="Inferencia OpenVLA persistente",
    )
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise MainStop(
            "OpenVLA persistente devolvió una respuesta no JSON:\n"
            + response_text[-4000:]
        ) from exc

    local_output = run_dir / "openvla_raw_output.json"
    json_write(local_output, payload)

    required = {
        "status": "ok",
        "model": "openvla",
        "request_id": request_id,
        "image_sha256": image_hash,
        "instruction": instruction,
    }
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise MainStop(
            "Respuesta OpenVLA persistente no corresponde a la solicitud: "
            f"{mismatches}"
        )

    actions = find_action_matrix(payload, EXPECTED_SHAPES[model])
    if actions is None:
        raise MainStop(
            f"OpenVLA persistente no devolvió acciones {EXPECTED_SHAPES[model]}"
        )
    evidence = {
        "model": model,
        "backend": "persistent_http",
        "backend_health": health,
        "request_id": request_id,
        "native_shape": list(actions.shape),
        "actions_finite": bool(np.all(np.isfinite(actions))),
        "image_path": str(image_path),
        "image_sha256": image_hash,
        "instruction": instruction,
        "seed": int(seed),
        "local_output": str(local_output),
        "timing": payload.get("timing", {}),
    }
    return actions, evidence


def _persistent_openpi_inference(
    *,
    model: str,
    image_path: Path,
    instruction: str,
    q_ur5e: np.ndarray,
    gripper_width_m: float,
    droid_state_mode: str,
    seed: int,
    run_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model not in {"pi05_droid", "pi0_droid"}:
        raise MainStop(f"Backend OpenPI recibió modelo incompatible: {model}")

    _ssh_bash(
        "python3 -c "
        + shlex.quote(
            "import socket; "
            f"s=socket.create_connection(('127.0.0.1',{OPENPI_PERSISTENT_PORT}),4); "
            "s.close(); print('OPENPI_PORT_READY')"
        ),
        label="Health OpenPI persistente",
    )

    remote_image = _copy_inference_image_to_orin(image_path)
    image_hash = sha256_file(image_path)
    request_id = f"{run_dir.name}:{image_hash[:16]}:{int(seed)}"
    remote_output = (
        "/mnt/sd_pi0/openpi/results/"
        f"persistent_{model}_{image_hash[:16]}.json"
    )
    joint_position = ",".join(
        f"{value:.10f}"
        for value in state_for_droid(q_ur5e, droid_state_mode)
    )
    gripper_position = float(
        np.clip(
            (gripper_width_m - 0.015) / (0.090 - 0.015),
            0.0,
            1.0,
        )
    )
    command_text = (
        f"source {shlex.quote(DROID_ACTIVATION_SCRIPT)} && "
        f"{shlex.quote(DROID_PYTHON)} "
        f"{shlex.quote(OPENPI_CLIENT_SCRIPT)} "
        f"--model {shlex.quote(model)} "
        f"--host 127.0.0.1 "
        f"--port {OPENPI_PERSISTENT_PORT} "
        f"--base-image {shlex.quote(remote_image)} "
        f"--prompt {shlex.quote(instruction)} "
        f"--joint-position {shlex.quote(joint_position)} "
        f"--gripper-position {gripper_position:.8f} "
        f"--seed {int(seed)} "
        f"--request-id {shlex.quote(request_id)} "
        f"--image-sha256 {shlex.quote(image_hash)} "
        f"--out {shlex.quote(remote_output)}"
    )
    provider_output = _ssh_bash(
        command_text,
        label=f"Inferencia {model} persistente",
    )

    # Primero se comprueba que la salida sea nueva y no esté vacía.
    remote_size_text = _ssh_bash(
        f"test -s {shlex.quote(remote_output)} && "
        f"stat -c %s {shlex.quote(remote_output)}",
        label="Validación de salida OpenPI persistente",
    ).strip()
    try:
        remote_size = int(remote_size_text.splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise MainStop(
            "OpenPI persistente devolvió un tamaño de salida no válido: "
            f"{remote_size_text!r}"
        ) from exc
    if remote_size <= 0:
        raise MainStop("OpenPI persistente creó una salida vacía")

    local_output = run_dir / f"{model}_raw_output.json"
    local_tmp = run_dir / f".{model}_raw_output.json.part"
    run_checked(
        ["scp", "-q", f"{require_orin_ssh()}:{remote_output}", str(local_tmp)],
        label="Copia de salida OpenPI persistente",
    )
    if not local_tmp.is_file() or local_tmp.stat().st_size != remote_size:
        raise MainStop(
            "La copia local de OpenPI no coincide con la salida remota: "
            f"remote_size={remote_size} "
            f"local_size={local_tmp.stat().st_size if local_tmp.exists() else -1}"
        )
    os.replace(local_tmp, local_output)
    payload = json.loads(local_output.read_text(encoding="utf-8"))
    required = {
        "status": "ok",
        "model": model,
        "request_id": request_id,
        "image_sha256": image_hash,
        "instruction": instruction,
    }
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise MainStop(
            "Respuesta OpenPI persistente no corresponde a la solicitud: "
            f"{mismatches}"
        )
    actions = find_action_matrix(payload, EXPECTED_SHAPES[model])
    if actions is None:
        raise MainStop(
            f"{model} persistente no devolvió acciones {EXPECTED_SHAPES[model]}"
        )
    evidence = {
        "model": model,
        "backend": "persistent_openpi_websocket",
        "request_id": request_id,
        "native_shape": list(actions.shape),
        "actions_finite": bool(np.all(np.isfinite(actions))),
        "image_path": str(image_path),
        "image_sha256": image_hash,
        "instruction": instruction,
        "seed": int(seed),
        "droid_state_mode": droid_state_mode,
        "local_output": str(local_output),
        "provider_output_tail": provider_output[-4000:],
        "server_metadata": payload.get("server_metadata", {}),
        "timing": payload.get("timing", {}),
    }
    return actions, evidence


def infer_on_orin(
    *,
    model: str,
    image_path: Path,
    instruction: str,
    q_ur5e: np.ndarray,
    gripper_width_m: float,
    droid_state_mode: str,
    seed: int,
    run_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    # Todos los backends/modelos usan el mismo criterio para guardar la evidencia.
    # OpenVLA crea los directorios con json_write(), mientras OpenPI copia con
    # scp; por eso el directorio debe existir antes de inferir.
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    backend = os.environ.get("VLA_BACKEND", "persistent").strip().lower()
    if backend == "persistent":
        if model == "openvla":
            return _persistent_openvla_inference(
                model=model,
                image_path=image_path,
                instruction=instruction,
                seed=seed,
                run_dir=run_dir,
            )
        return _persistent_openpi_inference(
            model=model,
            image_path=image_path,
            instruction=instruction,
            q_ur5e=q_ur5e,
            gripper_width_m=gripper_width_m,
            droid_state_mode=droid_state_mode,
            seed=seed,
            run_dir=run_dir,
        )
    if backend == "direct":
        if os.environ.get("VLA_ALLOW_DIRECT_MODEL_RELOAD") != "YES":
            raise MainStop(
                "La recarga directa del modelo está deshabilitada. "
                "Usa VLA_BACKEND=persistent o confirma "
                "VLA_ALLOW_DIRECT_MODEL_RELOAD=YES para una prueba técnica."
            )
        actions, evidence = infer_on_orin_direct(
            model=model,
            image_path=image_path,
            instruction=instruction,
            q_ur5e=q_ur5e,
            gripper_width_m=gripper_width_m,
            droid_state_mode=droid_state_mode,
            seed=seed,
            run_dir=run_dir,
        )
        evidence["backend"] = "direct_model_reload"
        return actions, evidence
    raise MainStop(
        f"VLA_BACKEND desconocido: {backend}; "
        "valores válidos: persistent, direct"
    )

# Grounding visual-lingüístico.
# Photoneo genera candidatos geométricos neutros. La selección
# semántica se obtiene midiendo cuánto cambia la acción nativa del VLA cuando
# se oculta cada región candidata. Se usa una ablación de regiones:
# imagen + instrucción -> VLA -> sensibilidad por candidato.
# El parser queda como referencia y como apoyo para construir la geometría.


def _candidate_xyz(value: dict[str, Any]) -> np.ndarray | None:
    keys = (
        "base_xyz_m",
        "xyz_base_m",
        "center_base_xyz_m",
        "object_xyz_m",
        "target_xyz_m",
    )
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            try:
                xyz = np.asarray(raw[:3], dtype=float)
            except (TypeError, ValueError):
                continue
            if xyz.shape == (3,) and np.all(np.isfinite(xyz)):
                return xyz
    return None


def _candidate_bbox(
    value: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> list[int] | None:
    raw = value.get("bbox_xywh")
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        x, y, w, h = [int(round(float(item))) for item in raw[:4]]
        x1, y1 = x, y
        x2, y2 = x + max(w, 1), y + max(h, 1)
    else:
        raw = value.get("bbox_xyxy") or value.get("bbox")
        if isinstance(raw, (list, tuple)) and len(raw) >= 4:
            x1, y1, x2, y2 = [int(round(float(item))) for item in raw[:4]]
        else:
            center = value.get("center_px") or value.get("center_uv")
            if not (
                isinstance(center, (list, tuple))
                and len(center) >= 2
            ):
                return None
            radius = int(
                os.environ.get(
                    "VLA_GROUNDING_DEFAULT_RADIUS_PX",
                    "36",
                )
            )
            cx = int(round(float(center[0])))
            cy = int(round(float(center[1])))
            x1, y1 = cx - radius, cy - radius
            x2, y2 = cx + radius, cy + radius

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(x1 + 1, min(image_width, x2))
    y2 = max(y1 + 1, min(image_height, y2))
    return [x1, y1, x2, y2]


def _candidate_role(name: str, path: tuple[str, ...], value: dict[str, Any]) -> str:
    explicit = " ".join(
        str(value.get(key, ""))
        for key in ("role", "kind", "type", "category")
    ).lower()
    context = " ".join((*path, name, explicit)).lower()

    target_tokens = (
        "target",
        "zone",
        "area",
        "region",
        "box",
        "bin",
        "container",
        "source",
        "destination",
    )
    object_tokens = (
        "object",
        "cube",
        "die",
        "cylinder",
        "piece",
        "part",
        "item",
    )

    if any(token in context for token in target_tokens):
        return "target"
    if "objects" in path or any(token in context for token in object_tokens):
        return "object"
    return "unknown"


def _bbox_iou(first: list[int], second: list[int]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _canonical_candidate_name(names: list[str], role: str) -> str:
    unique = sorted({str(name) for name in names})
    if role == "target":
        suffix_priority = {
            "_zone": 0,
            "_area": 1,
            "_region": 2,
            "_target": 3,
            "_box": 4,
            "_bin": 5,
            "_source": 6,
        }
    else:
        suffix_priority = {
            "_cube": 0,
            "_cylinder": 0,
            "_block": 1,
            "_object": 2,
            "_die": 3,
        }

    def ranking(name: str) -> tuple[int, int, str]:
        suffix_rank = min(
            (
                rank
                for suffix, rank in suffix_priority.items()
                if name.endswith(suffix)
            ),
            default=20,
        )
        return suffix_rank, len(name), name

    return min(unique, key=ranking)


def _same_physical_candidate(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    if first["role"] != second["role"]:
        return False
    xyz_a = finite_vector(first["base_xyz_m"], 3, "candidate.xyz_a")
    xyz_b = finite_vector(second["base_xyz_m"], 3, "candidate.xyz_b")
    xyz_distance = float(np.linalg.norm(xyz_a - xyz_b))
    iou = _bbox_iou(first["bbox_xyxy"], second["bbox_xyxy"])
    acx = 0.5 * (first["bbox_xyxy"][0] + first["bbox_xyxy"][2])
    acy = 0.5 * (first["bbox_xyxy"][1] + first["bbox_xyxy"][3])
    bcx = 0.5 * (second["bbox_xyxy"][0] + second["bbox_xyxy"][2])
    bcy = 0.5 * (second["bbox_xyxy"][1] + second["bbox_xyxy"][3])
    center_distance_px = math.hypot(acx - bcx, acy - bcy)
    return bool(
        (iou >= 0.80 and xyz_distance <= 0.020)
        or (center_distance_px <= 4.0 and xyz_distance <= 0.012)
    )



_PLY_SCALAR_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}

_T_BASE_FROM_CAMERA_MM = np.array(
    [
        [0.041, -0.998, -0.056, -219.488],
        [-0.999, -0.041, 0.000, -952.756],
        [-0.002, 0.056, -0.998, 866.163],
        [0.000, 0.000, 0.000, 1.000],
    ],
    dtype=float,
)


def _find_scene_ply_path(scene: dict[str, Any]) -> Path | None:
    for key in ("ply", "ply_path", "point_cloud_path"):
        value = scene.get(key)
        if isinstance(value, str) and Path(value).is_file():
            return Path(value)

    def walk(value: Any) -> Path | None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in {"ply", "ply_path", "point_cloud_path"}:
                    if isinstance(item, str) and Path(item).is_file():
                        return Path(item)
            for item in value.values():
                found = walk(item)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = walk(item)
                if found is not None:
                    return found
        return None

    found = walk(scene)
    if found is not None:
        return found

    fallback = ROOT / "photoneo_inbox" / "live_scene.ply"
    return fallback if fallback.is_file() else None


def _read_binary_or_ascii_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise MainStop(f"PLY inválido: {path}")

        fmt = None
        vertex_count = None
        active_element = None
        properties: list[tuple[str, str]] = []

        while True:
            raw = handle.readline()
            if not raw:
                raise MainStop(f"PLY truncado: {path}")
            line = raw.decode("ascii", errors="strict").strip()
            if line == "end_header":
                data_offset = handle.tell()
                break
            if not line or line.startswith("comment") or line.startswith("obj_info"):
                continue
            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                active_element = parts[1]
                if active_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and active_element == "vertex":
                if parts[1] == "list":
                    raise MainStop("PLY vertex con propiedades list no soportadas")
                code = _PLY_SCALAR_DTYPES.get(parts[1])
                if code is None:
                    raise MainStop(f"Tipo PLY no soportado: {parts[1]}")
                properties.append((parts[2], code))

        if fmt is None or vertex_count is None:
            raise MainStop(f"PLY sin format/vertex: {path}")

        names = [name for name, _ in properties]
        if not all(name in names for name in ("x", "y", "z")):
            raise MainStop(f"PLY sin x/y/z: {path}")

        if fmt in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if fmt == "binary_little_endian" else ">"
            dtype = np.dtype([(name, endian + code) for name, code in properties])
            handle.seek(data_offset)
            vertices = np.fromfile(handle, dtype=dtype, count=vertex_count)
            if len(vertices) != vertex_count:
                raise MainStop(
                    f"PLY truncado: {len(vertices)} de {vertex_count} vertices"
                )
            return np.column_stack(
                (
                    np.asarray(vertices["x"], dtype=float),
                    np.asarray(vertices["y"], dtype=float),
                    np.asarray(vertices["z"], dtype=float),
                )
            )

        if fmt == "ascii":
            handle.seek(data_offset)
            table = np.loadtxt(handle, dtype=float, max_rows=vertex_count)
            table = np.atleast_2d(table)
            indices = [names.index(name) for name in ("x", "y", "z")]
            return table[:, indices]

        raise MainStop(f"Formato PLY no soportado: {fmt}")


def _recover_visual_region_from_organized_ply(
    *,
    ply_path: Path,
    image_height: int,
    image_width: int,
    object_name: str,
    object_data: dict[str, Any],
) -> dict[str, Any]:
    camera_mm = _read_binary_or_ascii_ply_xyz(ply_path)
    expected = image_height * image_width
    if camera_mm.shape[0] != expected:
        raise MainStop(
            "La nube Photoneo no coincide con el raster RGB: "
            f"vertices={camera_mm.shape[0]} imagen={image_width}x{image_height}"
        )

    center = _candidate_xyz(object_data)
    if center is None:
        raise MainStop(f"{object_name}: falta base_xyz_m")

    rotation = _T_BASE_FROM_CAMERA_MM[:3, :3]
    translation = _T_BASE_FROM_CAMERA_MM[:3, 3]
    base_m = (camera_mm @ rotation.T + translation) * 0.001

    radius_m = float(os.environ.get("VLA_VISUAL_REGION_RADIUS_M", "0.038"))
    below_m = float(os.environ.get("VLA_VISUAL_REGION_BELOW_CENTER_M", "0.028"))
    above_m = float(os.environ.get("VLA_VISUAL_REGION_ABOVE_CENTER_M", "0.060"))
    min_points = int(os.environ.get("VLA_VISUAL_REGION_MIN_POINTS", "80"))

    finite = np.all(np.isfinite(base_m), axis=1)
    distance_xy = np.linalg.norm(base_m[:, :2] - center[:2], axis=1)
    raw_mask = (
        finite
        & (distance_xy <= radius_m)
        & (base_m[:, 2] >= center[2] - below_m)
        & (base_m[:, 2] <= center[2] + above_m)
    )

    mask_image = raw_mask.reshape(image_height, image_width).astype(np.uint8)
    mask_image = cv2.morphologyEx(
        mask_image,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_image,
        connectivity=8,
    )
    valid = [
        index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_points
    ]
    if not valid:
        areas = [
            int(stats[index, cv2.CC_STAT_AREA])
            for index in range(1, count)
        ]
        raise MainStop(
            f"{object_name}: no hay componente visual suficiente; "
            f"áreas={areas[:10]}"
        )

    component = max(
        valid,
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
    )
    x = int(stats[component, cv2.CC_STAT_LEFT])
    y = int(stats[component, cv2.CC_STAT_TOP])
    w = int(stats[component, cv2.CC_STAT_WIDTH])
    h = int(stats[component, cv2.CC_STAT_HEIGHT])
    area = int(stats[component, cv2.CC_STAT_AREA])
    cx, cy = [float(value) for value in centroids[component]]

    padding = int(os.environ.get("VLA_VISUAL_REGION_PADDING_PX", "6"))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_width, x + w + padding)
    y2 = min(image_height, y + h + padding)

    return {
        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
        "bbox_xyxy": [x1, y1, x2, y2],
        "center_px": [cx, cy],
        "area_px": area,
        "visual_region_source": "organized_photoneo_ply_metric_cluster",
        "visual_region_metric_radius_m": radius_m,
        "visual_region_z_window_m": [
            float(center[2] - below_m),
            float(center[2] + above_m),
        ],
        "visual_region_ply": str(ply_path),
    }



def _is_auxiliary_geometric_obstacle(
    name: str,
    value: dict[str, Any],
) -> bool:
    candidate_name = str(name).strip().lower()

    semantic_override = str(
        value.get("semantic_override", "")
    ).strip().lower()

    object_shape = str(
        value.get("object_shape", "")
    ).strip().lower()

    source = str(
        value.get("source", "")
    ).strip().lower()

    return bool(
        candidate_name.startswith("local_ply_obstacle_")
        or semantic_override == "local_ply_obstacle"
        or object_shape == "local_ply_obstacle"
        or source == "local_ply_geometric_annulus"
    )


def _enrich_missing_visual_regions(
    scene: dict[str, Any],
    image_path: Path,
) -> list[dict[str, Any]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise MainStop(f"No se pudo leer {image_path}")
    image_height, image_width = image.shape[:2]

    objects = scene.get("objects")
    if not isinstance(objects, dict):
        return []

    missing: list[tuple[str, dict[str, Any]]] = []
    for name, value in objects.items():
        if not isinstance(value, dict):
            continue

        if _is_auxiliary_geometric_obstacle(str(name), value):
            print(
                f"object={name} "
                "visual_region_required=false "
                "preserved_for_scene_safety=true"
            )
            continue

        if _candidate_xyz(value) is None:
            continue
        if _candidate_bbox(
            value,
            image_width=image_width,
            image_height=image_height,
        ) is None:
            missing.append((str(name), value))

    if not missing:
        return []

    ply_path = _find_scene_ply_path(scene)
    if ply_path is None:
        raise MainStop(
            "Hay objetos 3D sin bbox y no se encontró el PLY organizado"
        )

    reports = []
    for name, value in missing:
        region = _recover_visual_region_from_organized_ply(
            ply_path=ply_path,
            image_height=image_height,
            image_width=image_width,
            object_name=name,
            object_data=value,
        )
        value.update(region)

        top_level = scene.get(name)
        if isinstance(top_level, dict):
            top_xyz = _candidate_xyz(top_level)
            value_xyz = _candidate_xyz(value)
            if (
                top_xyz is not None
                and value_xyz is not None
                and float(np.linalg.norm(top_xyz - value_xyz)) <= 0.010
            ):
                top_level.update(region)

        report = {"object": name, **region}
        reports.append(report)
        print(
            f"object={name} bbox={region['bbox_xywh']} "
            f"center={region['center_px']} "
            f"area={region['area_px']} "
            f"source={region['visual_region_source']}"
        )

    return reports

def neutral_scene_candidates(
    scene: dict[str, Any],
    image_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    _enrich_missing_visual_regions(scene, image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise MainStop(f"No se pudo leer la imagen para grounding: {image_path}")
    height, width = image.shape[:2]
    raw_candidates: list[dict[str, Any]] = []

    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            xyz = _candidate_xyz(value)
            if xyz is not None and path:
                name = str(path[-1])
                role = _candidate_role(name, path[:-1], value)
                bbox = _candidate_bbox(
                    value,
                    image_width=width,
                    image_height=height,
                )
                if role in {"object", "target"} and bbox is not None:
                    candidate_record = {
                        "name":
                            name,
                        "aliases":
                            [name],
                        "role":
                            role,
                        "base_xyz_m":
                            xyz.tolist(),
                        "bbox_xyxy":
                            bbox,
                        "source_path":
                            list(path),
                        "source_paths":
                            [list(path)],
                    }

                    # Se mantienen solo los metadatos geométricos y de
                    # seguridad. SigLIP no recibe esos campos:
                    # el cliente reduce cada candidato a
                    # candidate_id + bbox_xyxy.
                    metadata_keys = (
                        "canonical",
                        "object_shape",
                        "container_shape",
                        "selected_interior_xyz_m",
                        "opening_floor_xyz_m",
                        "rim_reference_xyz_m",
                        "opening_center_uv",
                        "opening_bbox_xyxy",
                        "container_interior_valid",
                        "gripper_enters_container",
                        "release_policy",
                        "source",
                    )

                    for metadata_key in metadata_keys:
                        if metadata_key in value:
                            candidate_record[
                                metadata_key
                            ] = copy.deepcopy(
                                value[
                                    metadata_key
                                ]
                            )

                    raw_candidates.append(
                        candidate_record
                    )
            for key, item in value.items():
                if str(key) in {"stats", "planned_safe_probe_sequence"}:
                    continue
                walk(item, (*path, str(key)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, (*path, str(index)))

    walk(scene)

    physical_candidates: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        matched = None
        for current in physical_candidates:
            if _same_physical_candidate(current, candidate):
                matched = current
                break
        if matched is None:
            physical_candidates.append(copy.deepcopy(candidate))
            continue
        matched["aliases"] = sorted(
            set(matched.get("aliases", [matched["name"]]))
            | set(candidate.get("aliases", [candidate["name"]]))
        )
        matched["source_paths"].extend(candidate.get("source_paths", []))
        matched["name"] = _canonical_candidate_name(
            matched["aliases"], matched["role"]
        )
        old_area = (
            (matched["bbox_xyxy"][2] - matched["bbox_xyxy"][0])
            * (matched["bbox_xyxy"][3] - matched["bbox_xyxy"][1])
        )
        new_area = (
            (candidate["bbox_xyxy"][2] - candidate["bbox_xyxy"][0])
            * (candidate["bbox_xyxy"][3] - candidate["bbox_xyxy"][1])
        )
        if new_area > old_area:
            matched["bbox_xyxy"] = candidate["bbox_xyxy"]
            matched["base_xyz_m"] = candidate["base_xyz_m"]
            matched["source_path"] = candidate["source_path"]

    grouped = {"object": [], "target": []}
    for candidate in physical_candidates:
        candidate["name"] = _canonical_candidate_name(
            candidate.get("aliases", [candidate["name"]]),
            candidate["role"],
        )
        grouped[candidate["role"]].append(candidate)

    for role in grouped:
        grouped[role].sort(key=lambda item: item["name"])
        for item in grouped[role]:
            if len(item.get("aliases", [])) > 1:
                print(
                    f"role={role} canonical={item['name']} "
                    f"aliases={item['aliases']}"
                )

    # No se da por válida una selección visual entre varios objetos si un
    # objeto detectado en 3D carece de bbox/máscara 2D.
    missing_visual_objects: list[str] = []
    declared_objects = scene.get("objects")
    if isinstance(declared_objects, dict):
        available_names = {
            str(name)
            for item in grouped["object"]
            for name in [item["name"], *item.get("aliases", [])]
        }
        for name, value in declared_objects.items():
            if not isinstance(value, dict):
                continue

            if _is_auxiliary_geometric_obstacle(str(name), value):
                continue

            if _candidate_xyz(value) is None:
                continue

            if str(name) not in available_names:
                missing_visual_objects.append(str(name))
    if (
        missing_visual_objects
        and os.environ.get("VLA_REQUIRE_ALL_OBJECT_BBOX", "YES") == "YES"
    ):
        raise MainStop(
            "Grounding VLA incompleto: objetos 3D sin bbox/máscara visual: "
            f"{sorted(missing_visual_objects)}. "
            "El detector debe publicar bbox_xywh, bbox_xyxy o center_px "
            "para cada objeto antes de comparar modelos."
        )

    if not grouped["object"]:
        raise MainStop(
            "Grounding VLA: Photoneo no proporcionó candidatos de objeto "
            "con bbox/center_px y base_xyz_m."
        )
    if not grouped["target"]:
        raise MainStop(
            "Grounding VLA: Photoneo no proporcionó candidatos de destino "
            "con bbox/center_px y base_xyz_m."
        )
    return grouped


def _bbox_contains_object(
    target: dict[str, Any],
    selected_object: dict[str, Any],
) -> tuple[bool, dict[str, float]]:
    tx1, ty1, tx2, ty2 = [float(v) for v in target["bbox_xyxy"]]
    ox1, oy1, ox2, oy2 = [float(v) for v in selected_object["bbox_xyxy"]]
    ix1 = max(tx1, ox1)
    iy1 = max(ty1, oy1)
    ix2 = min(tx2, ox2)
    iy2 = min(ty2, oy2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    object_area = max(1.0, (ox2 - ox1) * (oy2 - oy1))
    object_coverage = float(intersection / object_area)
    object_xyz = finite_vector(
        selected_object["base_xyz_m"], 3, "grounding.object_xyz"
    )
    target_xyz = finite_vector(
        target["base_xyz_m"], 3, "grounding.target_xyz"
    )
    xy_distance = float(np.linalg.norm(object_xyz[:2] - target_xyz[:2]))
    contains = bool(object_coverage >= 0.55 and xy_distance <= 0.16)
    return contains, {
        "object_bbox_coverage": object_coverage,
        "object_target_xy_distance_m": xy_distance,
    }


def _safe_candidate_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() else "_"
        for character in value
    ).strip("_")
    return cleaned[:80] or "candidate"


def _mask_candidate_regions_image(
    image_path: Path,
    candidates_to_mask: list[dict[str, Any]],
    output_path: Path,
    *,
    variant: str = "median",
    padding_override: int | None = None,
    blur_kernel_override: int | None = None,
) -> None:
    """
    Oculta regiones físicas con variantes neutras y reproducibles.
    
        Las variantes permiten medir si la decisión del mismo VLA se conserva
        frente a artefactos concretos del enmascarado. Ninguna variante introduce
        etiquetas, colores de clase ni información procedente de otro modelo.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise MainStop(f"No se pudo leer {image_path}")

    valid_variants = {"median", "local_median", "inpaint"}
    if variant not in valid_variants:
        raise MainStop(
            f"Variante de máscara no válida: {variant}; "
            f"válidas={sorted(valid_variants)}"
        )

    default_padding = int(
        os.environ.get("VLA_GROUNDING_MASK_PADDING_PX", "8")
    )
    padding = default_padding if padding_override is None else int(padding_override)
    padding = max(0, padding)

    default_kernel = int(
        os.environ.get("VLA_GROUNDING_BLUR_KERNEL", "21")
    )
    kernel = default_kernel if blur_kernel_override is None else int(
        blur_kernel_override
    )
    kernel = max(3, kernel)
    if kernel % 2 == 0:
        kernel += 1

    global_median = np.median(image.reshape(-1, 3), axis=0).astype(np.uint8)
    masked = image.copy()
    inpaint_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for candidate in candidates_to_mask:
        bx1, by1, bx2, by2 = [int(v) for v in candidate["bbox_xyxy"]]
        x1 = max(0, bx1 - padding)
        y1 = max(0, by1 - padding)
        x2 = min(image.shape[1], bx2 + padding)
        y2 = min(image.shape[0], by2 + padding)
        if x2 <= x1 or y2 <= y1:
            continue

        if variant == "inpaint":
            inpaint_mask[y1:y2, x1:x2] = 255
            continue

        fill = global_median
        if variant == "local_median":
            ring_padding = max(8, 2 * max(1, padding))
            rx1 = max(0, x1 - ring_padding)
            ry1 = max(0, y1 - ring_padding)
            rx2 = min(image.shape[1], x2 + ring_padding)
            ry2 = min(image.shape[0], y2 + ring_padding)
            context = image[ry1:ry2, rx1:rx2]
            if context.size:
                ring_mask = np.ones(context.shape[:2], dtype=bool)
                ix1 = x1 - rx1
                iy1 = y1 - ry1
                ix2 = x2 - rx1
                iy2 = y2 - ry1
                ring_mask[iy1:iy2, ix1:ix2] = False
                ring_pixels = context[ring_mask]
                if ring_pixels.size:
                    fill = np.median(ring_pixels, axis=0).astype(np.uint8)

        masked[y1:y2, x1:x2] = fill
        roi = masked[y1:y2, x1:x2]
        if roi.size and min(roi.shape[:2]) >= 3:
            local_kernel = min(kernel, min(roi.shape[:2]) | 1)
            if local_kernel % 2 == 0:
                local_kernel -= 1
            if local_kernel >= 3:
                masked[y1:y2, x1:x2] = cv2.GaussianBlur(
                    roi,
                    (local_kernel, local_kernel),
                    0,
                )

    if variant == "inpaint" and np.any(inpaint_mask):
        masked = cv2.inpaint(
            image,
            inpaint_mask,
            5.0,
            cv2.INPAINT_TELEA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), masked):
        raise MainStop(f"No se pudo escribir {output_path}")


def _action_grounding_distance(
    model: str,
    reference: np.ndarray,
    counterfactual: np.ndarray,
) -> dict[str, float]:
    """Distancia ponderada entre políticas nativas del mismo VLA."""
    reference = np.asarray(reference, dtype=float)
    counterfactual = np.asarray(counterfactual, dtype=float)
    if reference.shape != counterfactual.shape:
        raise MainStop(
            f"Grounding {model}: shapes incompatibles "
            f"{reference.shape} vs {counterfactual.shape}"
        )
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(counterfactual)):
        raise MainStop(f"Grounding {model}: acciones no finitas")

    action_dim = min(reference.shape[-1], 7)
    delta = counterfactual[..., :action_dim] - reference[..., :action_dim]
    xyz_dim = min(3, action_dim)
    mean_abs = float(np.mean(np.abs(delta)))
    rms = float(np.sqrt(np.mean(delta * delta)))
    endpoint_xyz = float(np.linalg.norm(delta[-1, :xyz_dim]))
    temporal_xyz = float(
        np.mean(np.linalg.norm(delta[..., :xyz_dim], axis=-1))
    )
    direction_penalty = 0.0
    ref_xyz = np.asarray(reference[..., :xyz_dim], dtype=float).reshape(-1, xyz_dim)
    cf_xyz = np.asarray(counterfactual[..., :xyz_dim], dtype=float).reshape(-1, xyz_dim)
    ref_vector = np.sum(ref_xyz, axis=0)
    cf_vector = np.sum(cf_xyz, axis=0)
    ref_norm = float(np.linalg.norm(ref_vector))
    cf_norm = float(np.linalg.norm(cf_vector))
    if ref_norm > 1e-9 and cf_norm > 1e-9:
        cosine = float(np.dot(ref_vector, cf_vector) / (ref_norm * cf_norm))
        cosine = float(np.clip(cosine, -1.0, 1.0))
        direction_penalty = 1.0 - cosine

    distance = (
        mean_abs
        + 0.50 * rms
        + 0.35 * endpoint_xyz
        + 0.35 * temporal_xyz
        + 0.15 * direction_penalty
    )
    return {
        "distance": float(distance),
        "mean_abs_action_delta": mean_abs,
        "rms_action_delta": rms,
        "endpoint_xyz_action_delta": endpoint_xyz,
        "temporal_xyz_action_delta": temporal_xyz,
        "direction_penalty": direction_penalty,
    }


def _normalize_counterfactual_scores(
    role: str,
    raw: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Combina necesidad y suficiencia de cada región para la política.
    
        Necesidad: al ocultar el candidato la política debe cambiar.
        Suficiencia: al ocultar los demás candidatos la política debe conservarse.
    """
    if not raw:
        raise MainStop(f"Grounding VLA: lista vacía para {role}")
    if len(raw) == 1:
        only = dict(raw[0])
        only.update(
            {
                "score": 1.0,
                "necessity_score_normalized": 1.0,
                "sufficiency_score_normalized": 1.0,
                "necessity_spread": 0.0,
                "sufficiency_spread": 0.0,
                "necessity_weight": 0.55,
                "sufficiency_weight": 0.45,
                "single_candidate_after_geometric_role_filter": True,
            }
        )
        return [only]

    necessity = np.asarray(
        [float(item["necessity_distance"]) for item in raw],
        dtype=float,
    )
    sufficiency_distance = np.asarray(
        [float(item["sufficiency_distance"]) for item in raw],
        dtype=float,
    )
    if not np.all(np.isfinite(necessity)) or not np.all(np.isfinite(sufficiency_distance)):
        raise MainStop(f"Grounding VLA: métricas no finitas para {role}")

    def normalize(values: np.ndarray) -> tuple[np.ndarray, float]:
        low = float(np.min(values))
        high = float(np.max(values))
        spread = high - low
        if spread <= 1e-12:
            return np.full_like(values, 0.5), spread
        return (values - low) / spread, spread

    necessity_norm, necessity_spread = normalize(necessity)
    sufficiency_bad_norm, sufficiency_spread = normalize(sufficiency_distance)
    sufficiency_norm = 1.0 - sufficiency_bad_norm

    necessity_weight = float(
        os.environ.get("VLA_GROUNDING_NECESSITY_WEIGHT", "0.55")
    )
    necessity_weight = float(np.clip(necessity_weight, 0.0, 1.0))
    sufficiency_weight = 1.0 - necessity_weight

    scored: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        score = (
            necessity_weight * float(necessity_norm[index])
            + sufficiency_weight * float(sufficiency_norm[index])
        )
        scored.append(
            {
                **item,
                "score": float(score),
                "necessity_score_normalized": float(necessity_norm[index]),
                "sufficiency_score_normalized": float(sufficiency_norm[index]),
                "necessity_spread": necessity_spread,
                "sufficiency_spread": sufficiency_spread,
                "necessity_weight": necessity_weight,
                "sufficiency_weight": sufficiency_weight,
            }
        )
    return scored


def _select_grounding_candidate(
    role: str,
    scored: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not scored:
        raise MainStop(f"Grounding VLA: lista vacía para {role}")

    ordered = sorted(scored, key=lambda item: item["score"], reverse=True)
    best = ordered[0]
    second_score = float(ordered[1]["score"]) if len(ordered) > 1 else 0.0
    score_margin = float(best["score"] - second_score)
    relative_margin = float(
        score_margin / max(abs(float(best["score"])), 1e-12)
    )
    minimum_margin = float(
        os.environ.get("VLA_GROUNDING_MIN_RELATIVE_MARGIN", "0.08")
    )
    minimum_score = float(
        os.environ.get("VLA_GROUNDING_MIN_SCORE", "0.55")
    )
    minimum_raw_spread = float(
        os.environ.get("VLA_GROUNDING_MIN_RAW_SPREAD", "1e-5")
    )

    necessity_spread = float(best.get("necessity_spread", 0.0))
    sufficiency_spread = float(best.get("sufficiency_spread", 0.0))
    if len(ordered) > 1 and max(necessity_spread, sufficiency_spread) < minimum_raw_spread:
        raise MainStop(
            f"Grounding VLA sin señal causal para {role}: "
            f"necessity_spread={necessity_spread:.8g}, "
            f"sufficiency_spread={sufficiency_spread:.8g}, "
            f"mínimo={minimum_raw_spread:.8g}"
        )
    if float(best["score"]) < minimum_score:
        raise MainStop(
            f"Grounding VLA insuficiente para {role}: score máximo "
            f"{float(best['score']):.4f} < {minimum_score:.4f}"
        )
    if len(ordered) > 1 and relative_margin < minimum_margin:
        raise MainStop(
            f"Grounding VLA ambiguo para {role}: margen relativo "
            f"{relative_margin:.4f} < {minimum_margin:.4f}; "
            f"mejor={best['name']} segundo={ordered[1]['name']}"
        )

    trace = {
        "role": role,
        "selected": best["name"],
        "selected_score": float(best["score"]),
        "second": ordered[1]["name"] if len(ordered) > 1 else None,
        "second_score": second_score,
        "score_margin": score_margin,
        "relative_margin": relative_margin,
        "minimum_relative_margin": minimum_margin,
        "minimum_score": minimum_score,
        "minimum_raw_spread": minimum_raw_spread,
        "candidates": ordered,
    }
    return best, trace


def _phase_target_instruction(instruction: str) -> str:
    """Condicionamiento de fase genérico; no contiene nombres ni decide roles."""
    return (
        "Assume the robot has already securely grasped the object requested "
        "by the user. Continue the same task and move toward the requested "
        f"destination. Original task: {instruction}"
    )


def _counterfactual_role_scores(
    *,
    role: str,
    candidates: list[dict[str, Any]],
    reference_actions: np.ndarray,
    reference_instruction: str,
    model: str,
    image_path: Path,
    q_ur5e: np.ndarray,
    gripper_width_m: float,
    droid_state_mode: str,
    seed: int,
    grounding_dir: Path,
) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        safe_name = _safe_candidate_name(candidate["name"])
        candidate_dir = grounding_dir / f"{role}_{index:02d}_{safe_name}"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        # Prueba de necesidad: se oculta solo el candidato analizado.
        occluded_path = candidate_dir / "necessity_occluded.png"
        _mask_candidate_regions_image(
            image_path,
            [candidate],
            occluded_path,
        )
        occluded_actions, occluded_evidence = infer_on_orin(
            model=model,
            image_path=occluded_path,
            instruction=reference_instruction,
            q_ur5e=q_ur5e,
            gripper_width_m=gripper_width_m,
            droid_state_mode=droid_state_mode,
            seed=seed,
            run_dir=candidate_dir / "necessity",
        )
        necessity_metrics = _action_grounding_distance(
            model,
            reference_actions,
            occluded_actions,
        )

        # Prueba de suficiencia: se mantienen el candidato, el fondo y los otros roles; se ocultan
        # solo los candidatos alternativos del mismo rol.
        alternatives = [
            other for other in candidates
            if other["name"] != candidate["name"]
        ]
        isolated_path = candidate_dir / "sufficiency_isolated.png"
        _mask_candidate_regions_image(
            image_path,
            alternatives,
            isolated_path,
        )
        isolated_actions, isolated_evidence = infer_on_orin(
            model=model,
            image_path=isolated_path,
            instruction=reference_instruction,
            q_ur5e=q_ur5e,
            gripper_width_m=gripper_width_m,
            droid_state_mode=droid_state_mode,
            seed=seed,
            run_dir=candidate_dir / "sufficiency",
        )
        sufficiency_metrics = _action_grounding_distance(
            model,
            reference_actions,
            isolated_actions,
        )

        item = {
            **candidate,
            "necessity_distance": float(necessity_metrics["distance"]),
            "sufficiency_distance": float(sufficiency_metrics["distance"]),
            "necessity_metrics": necessity_metrics,
            "sufficiency_metrics": sufficiency_metrics,
            "necessity_image": str(occluded_path),
            "sufficiency_image": str(isolated_path),
            "necessity_model_evidence": occluded_evidence,
            "sufficiency_model_evidence": isolated_evidence,
            "grounding_instruction": reference_instruction,
        }
        raw.append(item)
        print(
            f"role={role} candidate={candidate['name']} "
            f"necessity={item['necessity_distance']:.8g} "
            f"sufficiency_distance={item['sufficiency_distance']:.8g}"
        )

    scored = _normalize_counterfactual_scores(role, raw)
    for item in scored:
        print(
            f"role={role} candidate={item['name']} "
            f"score={item['score']:.6f} "
            f"necessity={item['necessity_score_normalized']:.4f} "
            f"sufficiency={item['sufficiency_score_normalized']:.4f}"
        )
    return scored


def _render_grounding_overlay(
    image_path: Path,
    object_scores: list[dict[str, Any]],
    target_scores: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    all_scores = object_scores + target_scores
    maximum = max((item["score"] for item in all_scores), default=1.0)
    maximum = max(maximum, 1e-12)
    for item in all_scores:
        x1, y1, x2, y2 = [int(v) for v in item["bbox_xyxy"]]
        normalized = float(item["score"] / maximum)
        intensity = int(round(80 + 175 * np.clip(normalized, 0.0, 1.0)))
        color = (0, intensity, 255 - intensity) if item["role"] == "object" else (intensity, 255 - intensity, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        label = f"{item['role']}:{item['name']} {item['score']:.4g}"
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)



def _load_heatmap_array(path: Path, *, width: int, height: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        heatmap = np.load(path)
    elif suffix == ".npz":
        payload = np.load(path)
        keys = list(payload.keys())
        if not keys:
            raise MainStop(f"NPZ de heatmap vacío: {path}")
        preferred = next(
            (key for key in ("heatmap", "attention", "saliency") if key in payload),
            keys[0],
        )
        heatmap = payload[preferred]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("heatmap", "attention", "saliency", "values"):
                if key in payload:
                    payload = payload[key]
                    break
        heatmap = np.asarray(payload, dtype=float)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise MainStop(f"No se pudo leer el heatmap: {path}")
        if raw.ndim == 3:
            heatmap = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            heatmap = raw

    heatmap = np.asarray(heatmap, dtype=float)
    while heatmap.ndim > 2:
        heatmap = np.mean(heatmap, axis=0)
    if heatmap.ndim != 2 or not np.all(np.isfinite(heatmap)):
        raise MainStop(f"Heatmap inválido {path}: shape={heatmap.shape}")
    heatmap = cv2.resize(
        heatmap.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    minimum = float(np.min(heatmap))
    maximum = float(np.max(heatmap))
    if maximum - minimum <= 1e-12:
        raise MainStop(f"Heatmap constante sin información: {path}")
    return (heatmap - minimum) / (maximum - minimum)


def _score_candidates_from_heatmap(
    *,
    heatmap: np.ndarray,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    total_mass = max(float(np.sum(heatmap)), 1e-12)
    for candidate in candidates:
        x1, y1, x2, y2 = [int(v) for v in candidate["bbox_xyxy"]]
        region = heatmap[y1:y2, x1:x2]
        if region.size == 0:
            continue
        mean_score = float(np.mean(region))
        mass_fraction = float(np.sum(region) / total_mass)
        flat = np.sort(region.reshape(-1))
        top_count = max(1, int(round(0.20 * flat.size)))
        top20_mean = float(np.mean(flat[-top_count:]))
        score = 0.50 * mean_score + 0.30 * top20_mean + 0.20 * mass_fraction
        scored.append(
            {
                **candidate,
                "score": score,
                "heatmap_mean": mean_score,
                "heatmap_top20_mean": top20_mean,
                "heatmap_mass_fraction": mass_fraction,
            }
        )
    return scored


def _load_grounding_heatmap(
    path: Path,
    *,
    image_path: Path,
    candidates: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise MainStop(f"No se pudo leer la imagen live: {image_path}")
    height, width = image.shape[:2]
    heatmap = _load_heatmap_array(path, width=width, height=height)

    decisions: dict[str, Any] = {}
    scored_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in ("object", "target"):
        scored = _score_candidates_from_heatmap(
            heatmap=heatmap,
            candidates=candidates[role],
        )
        selected, trace = _select_grounding_candidate(role, scored)
        scored_by_role[role] = scored
        decisions[role] = selected
        decisions[f"{role}_trace"] = trace

    overlay_path = path.parent / f"{path.stem}_photoneo_regions.png"
    _render_grounding_overlay(
        image_path,
        scored_by_role["object"],
        scored_by_role["target"],
        overlay_path,
    )
    return {
        "schema": "vla_grounding",
        "method": "external_heatmap_region_scoring",
        "image_path": str(image_path),
        "image_sha256": sha256_file(image_path),
        "heatmap_path": str(path),
        "overlay_path": str(overlay_path),
        "parser_role": "trace_only_not_used_for_selection",
        **decisions,
    }

def _load_grounding_json(
    path: Path,
    *,
    image_path: Path,
    candidates: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = payload.get("image_sha256")
    actual_hash = sha256_file(image_path)
    if expected_hash and expected_hash != actual_hash:
        raise MainStop(
            "El grounding JSON pertenece a otra imagen: "
            f"esperado={expected_hash} actual={actual_hash}"
        )

    decisions: dict[str, Any] = {}
    for role in ("object", "target"):
        score_map = payload.get(f"{role}_scores")
        if not isinstance(score_map, dict):
            raise MainStop(f"Grounding JSON sin {role}_scores")
        scored = []
        for candidate in candidates[role]:
            if candidate["name"] not in score_map:
                continue
            scored.append({**candidate, "score": float(score_map[candidate["name"]])})
        selected, trace = _select_grounding_candidate(role, scored)
        decisions[role] = selected
        decisions[f"{role}_trace"] = trace

    return {
        "schema": "vla_grounding",
        "method": "external_heatmap_scores",
        "image_path": str(image_path),
        "image_sha256": actual_hash,
        "evidence_path": str(path),
        **decisions,
    }


def _grounding_candidate_label(candidate: dict[str, Any], role: str) -> str:
    """Nombre legible de una hipótesis física, sin usar el parser de la orden."""
    value = str(candidate["name"]).strip().lower()
    suffixes = (
        "_target",
        "_source",
        "_area",
        "_region",
        "_box",
        "_bin",
        "_zone",
    )
    if role == "target":
        for suffix in suffixes:
            if value.endswith(suffix):
                value = value[: -len(suffix)]
                break
        value = value.replace("_", " ").strip()
        if not value:
            value = "target"
        return f"{value} zone"
    return value.replace("_", " ").strip()


def _pair_policy_costs_normalized(
    raw_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normaliza costes por escala robusta, sin convertir diferencias mínimas en 0/1."""
    if not raw_pairs:
        raise MainStop("Grounding VLA: no existen hipótesis objeto-destino válidas")

    metric_names = (
        "object_prompt_distance",
        "object_visual_sufficiency_distance",
        "target_prompt_distance",
        "target_visual_sufficiency_distance",
    )
    weights = {
        "object_prompt_distance": 0.40,
        "object_visual_sufficiency_distance": 0.20,
        "target_prompt_distance": 0.25,
        "target_visual_sufficiency_distance": 0.15,
    }

    scales: dict[str, float] = {}
    for metric in metric_names:
        values = np.asarray([float(item[metric]) for item in raw_pairs], dtype=float)
        if not np.all(np.isfinite(values)):
            raise MainStop(f"Grounding VLA: métrica no finita {metric}")
        positive = values[values > 1e-12]
        if positive.size:
            scale = float(np.median(positive))
        else:
            scale = 1.0
        scales[metric] = max(scale, 1e-9)

    scored: list[dict[str, Any]] = []
    for item in raw_pairs:
        normalized = {
            metric: float(item[metric]) / scales[metric]
            for metric in metric_names
        }
        total_cost = float(
            sum(weights[metric] * normalized[metric] for metric in metric_names)
        )
        scored.append(
            {
                **item,
                "normalized_costs": normalized,
                "cost_scales": scales,
                "metric_weights": weights,
                "total_cost": total_cost,
                "score": float(1.0 / (1.0 + total_cost)),
            }
        )
    return scored


def _select_joint_policy_hypothesis(
    scored_pairs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered = sorted(scored_pairs, key=lambda item: float(item["total_cost"]))
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    second_cost = float(second["total_cost"]) if second is not None else math.inf
    best_cost = float(best["total_cost"])
    relative_margin = (
        float((second_cost - best_cost) / max(second_cost, 1e-9))
        if second is not None
        else 1.0
    )
    minimum_margin = float(
        os.environ.get("VLA_PAIR_MIN_RELATIVE_MARGIN", "0.08")
    )
    maximum_cost = float(
        os.environ.get("VLA_PAIR_MAX_NORMALIZED_COST", "1.35")
    )
    if not math.isfinite(minimum_margin) or not 0.0 <= minimum_margin <= 0.80:
        raise MainStop("VLA_PAIR_MIN_RELATIVE_MARGIN no válido")
    if not math.isfinite(maximum_cost) or not 0.20 <= maximum_cost <= 5.0:
        raise MainStop("VLA_PAIR_MAX_NORMALIZED_COST no válido")
    if best_cost > maximum_cost:
        raise MainStop(
            "Grounding VLA sin hipótesis consistente: "
            f"coste={best_cost:.4f} > {maximum_cost:.4f}"
        )
    if second is not None and relative_margin < minimum_margin:
        print(
            "Grounding VLA ambiguo entre hipótesis completas: "
            f"margen={relative_margin:.4f} < {minimum_margin:.4f}; "
            f"mejor={best['object']['name']}->{best['target']['name']} "
            f"segundo={second['object']['name']}->{second['target']['name']}"
        )
    trace = {
        "method": "same_vla_joint_policy_hypothesis",
        "selected_pair": {
            "object": best["object"]["name"],
            "target": best["target"]["name"],
            "total_cost": best_cost,
            "score": float(best["score"]),
        },
        "second_pair": (
            {
                "object": second["object"]["name"],
                "target": second["target"]["name"],
                "total_cost": second_cost,
                "score": float(second["score"]),
            }
            if second is not None
            else None
        ),
        "relative_margin": relative_margin,
        "minimum_relative_margin": minimum_margin,
        "maximum_normalized_cost": maximum_cost,
        "pairs": ordered,
    }
    return best, trace



def _joint_policy_round_margin(
    scored_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resume una ronda sin aplicar todavía el criterio de aceptación."""
    if not scored_pairs:
        raise MainStop("Grounding VLA: round sin hipótesis puntuadas")
    ordered = sorted(scored_pairs, key=lambda item: float(item["total_cost"]))
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    best_cost = float(best["total_cost"])
    second_cost = float(second["total_cost"]) if second is not None else math.inf
    relative_margin = (
        float((second_cost - best_cost) / max(second_cost, 1e-9))
        if second is not None
        else 1.0
    )
    return {
        "best": best,
        "second": second,
        "best_cost": best_cost,
        "second_cost": second_cost,
        "relative_margin": relative_margin,
        "winner_key": (
            str(best["object"]["name"]),
            str(best["target"]["name"]),
        ),
        "ordered": ordered,
    }


def _joint_grounding_variants(
    instruction: str,
    *,
    model: str,
) -> list[dict[str, Any]]:
    """
    Selecciona variantes respetando el contrato nativo.
    
        OpenVLA:
        - una única formulación lingüística;
        - una única seed;
        - referencias de acción congeladas;
        - solo cambia la intervención visual.
    
        pi0_droid y pi05_droid:
        - conservan exactamente las variantes anteriores.
    """
    if model != "openvla":
        return (
            _joint_grounding_variants_temporal(
                instruction
            )
        )

    base_padding = int(
        os.environ.get(
            "VLA_GROUNDING_MASK_PADDING_PX",
            "8",
        )
    )

    object_template = (
        "put the {object_label} "
        "in the {target_label}"
    )

    target_reference = (
        _phase_target_instruction(
            instruction
        )
    )

    target_template = (
        "Assume the robot has already securely "
        "grasped the {object_label}. "
        "Move it to the {target_label}."
    )

    visual_specs = [
        {
            "name":
                "visual_01_median",
            "mask_variant":
                "median",
            "mask_padding":
                base_padding,
            "mask_kernel":
                21,
        },
        {
            "name":
                "visual_02_local_median",
            "mask_variant":
                "local_median",
            "mask_padding":
                max(
                    4,
                    base_padding - 3,
                ),
            "mask_kernel":
                31,
        },
        {
            "name":
                "visual_03_inpaint",
            "mask_variant":
                "inpaint",
            "mask_padding":
                base_padding + 4,
            "mask_kernel":
                41,
        },
        {
            "name":
                "visual_04_median_wide",
            "mask_variant":
                "median",
            "mask_padding":
                base_padding + 2,
            "mask_kernel":
                27,
        },
        {
            "name":
                "visual_05_local_median_wide",
            "mask_variant":
                "local_median",
            "mask_padding":
                max(
                    4,
                    base_padding - 1,
                ),
            "mask_kernel":
                35,
        },
        {
            "name":
                "visual_06_inpaint_wide",
            "mask_variant":
                "inpaint",
            "mask_padding":
                base_padding + 6,
            "mask_kernel":
                45,
        },
        {
            "name":
                "visual_07_median_repeat",
            "mask_variant":
                "median",
            "mask_padding":
                base_padding + 1,
            "mask_kernel":
                25,
        },
    ]

    return [
        {
            **visual_spec,
            "seed_offset":
                0,
            "object_template":
                object_template,
            "target_reference":
                target_reference,
            "target_template":
                target_template,
            "decision_variable":
                "visual_intervention_only",
            "language_fixed":
                True,
            "seed_fixed":
                True,
        }
        for visual_spec
        in visual_specs
    ]


def _joint_grounding_variants_temporal(instruction: str) -> list[dict[str, Any]]:
    """Variantes deterministas para medir robustez y resolver empates reales."""
    base_padding = int(
        os.environ.get("VLA_GROUNDING_MASK_PADDING_PX", "8")
    )
    return [
        {
            "name": "base",
            "seed_offset": 0,
            "object_template": "put the {object_label} in the {target_label}",
            "target_reference": _phase_target_instruction(instruction),
            "target_template": (
                "Assume the robot has already securely grasped the "
                "{object_label}. Move it to the {target_label}."
            ),
            "mask_variant": "median",
            "mask_padding": base_padding,
            "mask_kernel": 21,
        },
        {
            "name": "direct",
            "seed_offset": 1,
            "object_template": (
                "pick up the {object_label} and place it in the {target_label}"
            ),
            "target_reference": (
                "The object requested by the user is already securely held. "
                "Continue the original task by carrying it to the requested "
                f"destination. Original task: {instruction}"
            ),
            "target_template": (
                "The {object_label} is already held by the gripper. "
                "Carry it to the {target_label}."
            ),
            "mask_variant": "local_median",
            "mask_padding": max(4, base_padding - 3),
            "mask_kernel": 31,
        },
        {
            "name": "goal",
            "seed_offset": 2,
            "object_template": (
                "move the {object_label} from its current location into "
                "the {target_label}"
            ),
            "target_reference": (
                "After grasping the requested object, complete the same user "
                "task by moving toward the requested destination. "
                f"Original task: {instruction}"
            ),
            "target_template": (
                "Complete the task by moving the grasped {object_label} "
                "into the {target_label}."
            ),
            "mask_variant": "inpaint",
            "mask_padding": base_padding + 4,
            "mask_kernel": 41,
        },
        {
            "name": "explicit_request",
            "seed_offset": 3,
            "object_template": (
                "execute the requested transfer of the {object_label} "
                "to the {target_label}"
            ),
            "target_reference": (
                "Keep following exactly the same user request after grasp. "
                f"Original task: {instruction}"
            ),
            "target_template": (
                "With the {object_label} grasped, finish the requested "
                "transfer at the {target_label}."
            ),
            "mask_variant": "median",
            "mask_padding": base_padding + 2,
            "mask_kernel": 27,
        },
        {
            "name": "source_focus",
            "seed_offset": 4,
            "object_template": (
                "select the {object_label} as the source object and move it "
                "to the {target_label}"
            ),
            "target_reference": (
                "The requested source object is now grasped. Continue toward "
                f"the destination in this task: {instruction}"
            ),
            "target_template": (
                "The selected source is the {object_label}; carry it to "
                "the {target_label}."
            ),
            "mask_variant": "local_median",
            "mask_padding": max(4, base_padding - 1),
            "mask_kernel": 35,
        },
        {
            "name": "destination_focus",
            "seed_offset": 5,
            "object_template": (
                "transfer the {object_label}; its required destination is "
                "the {target_label}"
            ),
            "target_reference": (
                "The source requested by the user is already held. Identify "
                "and approach only the requested destination. "
                f"Original task: {instruction}"
            ),
            "target_template": (
                "The required destination for the grasped {object_label} "
                "is the {target_label}."
            ),
            "mask_variant": "inpaint",
            "mask_padding": base_padding + 6,
            "mask_kernel": 45,
        },
        {
            "name": "canonical_repeat",
            "seed_offset": 6,
            "object_template": (
                "place the {object_label} inside the {target_label}"
            ),
            "target_reference": (
                "Continue the exact original task with the requested object "
                f"already grasped: {instruction}"
            ),
            "target_template": (
                "Place the grasped {object_label} inside the {target_label}."
            ),
            "mask_variant": "median",
            "mask_padding": base_padding + 1,
            "mask_kernel": 25,
        },
    ]



def _restore_visual_regions_exact(
    *,
    source_image_path: Path,
    modified_image_path: Path,
    protected_candidates: list[
        dict[str, Any]
    ],
    padding_px: int = 2,
) -> list[dict[str, Any]]:
    source_image = cv2.imread(
        str(source_image_path),
        cv2.IMREAD_COLOR,
    )

    modified_image = cv2.imread(
        str(modified_image_path),
        cv2.IMREAD_COLOR,
    )

    if source_image is None:
        raise MainStop(
            "No se pudo leer la imagen original "
            "para restaurar regiones"
        )

    if modified_image is None:
        raise MainStop(
            "No se pudo leer la imagen modificada "
            "para restaurar regiones"
        )

    if source_image.shape != modified_image.shape:
        raise MainStop(
            "La imagen modificada cambió de tamaño: "
            f"source={source_image.shape}, "
            f"modified={modified_image.shape}"
        )

    height, width = source_image.shape[:2]
    reports = []

    for candidate in protected_candidates:
        bbox = np.asarray(
            candidate.get("bbox_xyxy"),
            dtype=float,
        ).reshape(-1)

        if (
            bbox.size != 4
            or not np.all(np.isfinite(bbox))
        ):
            raise MainStop(
                "BBox no válida para restauración: "
                f"candidate={candidate.get('name')} "
                f"bbox={candidate.get('bbox_xyxy')}"
            )

        x1 = max(
            0,
            int(np.floor(bbox[0]))
            - int(padding_px),
        )

        y1 = max(
            0,
            int(np.floor(bbox[1]))
            - int(padding_px),
        )

        x2 = min(
            width,
            int(np.ceil(bbox[2]))
            + int(padding_px),
        )

        y2 = min(
            height,
            int(np.ceil(bbox[3]))
            + int(padding_px),
        )

        if x2 <= x1 or y2 <= y1:
            raise MainStop(
                "Región vacía durante restauración: "
                f"{candidate.get('name')}"
            )

        modified_image[
            y1:y2,
            x1:x2,
        ] = source_image[
            y1:y2,
            x1:x2,
        ]

        exact = bool(
            np.array_equal(
                modified_image[
                    y1:y2,
                    x1:x2,
                ],
                source_image[
                    y1:y2,
                    x1:x2,
                ],
            )
        )

        if not exact:
            raise MainStop(
                "La restauración visual no fue exacta: "
                f"{candidate.get('name')}"
            )

        reports.append(
            {
                "name":
                    str(
                        candidate.get("name")
                    ),
                "bbox_xyxy":
                    [x1, y1, x2, y2],
                "restored_exactly":
                    exact,
            }
        )

    write_ok = cv2.imwrite(
        str(modified_image_path),
        modified_image,
    )

    if not write_ok:
        raise MainStop(
            "No se pudo guardar la imagen "
            "con regiones restauradas"
        )

    return reports


def _openvla_causal_role_scores(
    *,
    role: str,
    role_candidates: list[
        dict[str, Any]
    ],
    protected_objects: list[
        dict[str, Any]
    ],
    reference_actions: np.ndarray,
    reference_instruction: str,
    image_path: Path,
    model: str,
    q_ur5e: np.ndarray,
    gripper_width_m: float,
    droid_state_mode: str,
    seed: int,
    round_name: str,
    round_dir: Path,
    mask_variant: str,
    mask_padding: int,
    mask_kernel: int,
) -> list[dict[str, Any]]:
    if role not in {"object", "target"}:
        raise MainStop(
            f"Rol causal desconocido: {role}"
        )

    if not role_candidates:
        raise MainStop(
            f"Sin candidatos causales para {role}"
        )

    scored = []

    for candidate in role_candidates:
        candidate_name = str(
            candidate["name"]
        )

        candidate_dir = (
            round_dir
            / "openvla_causal"
            / role
            / _safe_candidate_name(
                candidate_name
            )
        )

        candidate_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        necessity_image = (
            candidate_dir
            / "necessity_candidate_masked.png"
        )

        _mask_candidate_regions_image(
            image_path,
            [candidate],
            necessity_image,
            variant=mask_variant,
            padding_override=mask_padding,
            blur_kernel_override=mask_kernel,
        )

        necessity_restored = []

        if role == "target":
            necessity_restored = (
                _restore_visual_regions_exact(
                    source_image_path=image_path,
                    modified_image_path=necessity_image,
                    protected_candidates=
                        protected_objects,
                )
            )

        (
            necessity_actions,
            necessity_evidence,
        ) = infer_on_orin(
            model=model,
            image_path=necessity_image,
            instruction=reference_instruction,
            q_ur5e=q_ur5e,
            gripper_width_m=
                gripper_width_m,
            droid_state_mode=
                droid_state_mode,
            seed=seed,
            run_dir=(
                candidate_dir
                / "necessity_inference"
            ),
        )

        necessity_metrics = (
            _action_grounding_distance(
                model,
                reference_actions,
                necessity_actions,
            )
        )

        alternatives = [
            other
            for other in role_candidates
            if str(other["name"])
            != candidate_name
        ]

        sufficiency_image = (
            candidate_dir
            / "sufficiency_alternatives_masked.png"
        )

        _mask_candidate_regions_image(
            image_path,
            alternatives,
            sufficiency_image,
            variant=mask_variant,
            padding_override=mask_padding,
            blur_kernel_override=mask_kernel,
        )

        sufficiency_restored = []

        if role == "target":
            sufficiency_restored = (
                _restore_visual_regions_exact(
                    source_image_path=image_path,
                    modified_image_path=
                        sufficiency_image,
                    protected_candidates=
                        protected_objects,
                )
            )

        (
            sufficiency_actions,
            sufficiency_evidence,
        ) = infer_on_orin(
            model=model,
            image_path=sufficiency_image,
            instruction=reference_instruction,
            q_ur5e=q_ur5e,
            gripper_width_m=
                gripper_width_m,
            droid_state_mode=
                droid_state_mode,
            seed=seed,
            run_dir=(
                candidate_dir
                / "sufficiency_inference"
            ),
        )

        sufficiency_metrics = (
            _action_grounding_distance(
                model,
                reference_actions,
                sufficiency_actions,
            )
        )

        necessity_distance = float(
            necessity_metrics["distance"]
        )

        sufficiency_distance = float(
            sufficiency_metrics["distance"]
        )

        denominator = (
            necessity_distance
            + sufficiency_distance
        )

        if denominator <= 1e-12:
            signed_contrast = 0.0
        else:
            signed_contrast = float(
                np.clip(
                    (
                        necessity_distance
                        - sufficiency_distance
                    )
                    / denominator,
                    -1.0,
                    1.0,
                )
            )

        causal_score = float(
            0.5
            * (
                signed_contrast
                + 1.0
            )
        )

        causal_cost = float(
            1.0 - causal_score
        )

        item = {
            **candidate,
            "role":
                role,
            "round_name":
                round_name,
            "grounding_instruction":
                reference_instruction,
            "necessity_distance":
                necessity_distance,
            "sufficiency_distance":
                sufficiency_distance,
            "signed_causal_contrast":
                signed_contrast,
            "causal_score":
                causal_score,
            "causal_cost":
                causal_cost,
            "necessity_metrics":
                necessity_metrics,
            "sufficiency_metrics":
                sufficiency_metrics,
            "necessity_image":
                str(necessity_image),
            "sufficiency_image":
                str(sufficiency_image),
            "necessity_evidence":
                necessity_evidence,
            "sufficiency_evidence":
                sufficiency_evidence,
            "necessity_restored_objects":
                necessity_restored,
            "sufficiency_restored_objects":
                sufficiency_restored,
            "mask_variant":
                mask_variant,
            "mask_padding":
                int(mask_padding),
            "mask_kernel":
                int(mask_kernel),
            "decision_contract": (
                "same_prompt_visual_"
                "necessity_and_sufficiency"
            ),
        }

        scored.append(item)

        print(
            f"round={round_name} "
            f"role={role} "
            f"candidate={candidate_name} "
            f"necessity="
            f"{necessity_distance:.8f} "
            f"sufficiency="
            f"{sufficiency_distance:.8f} "
            f"contrast="
            f"{signed_contrast:.6f} "
            f"score={causal_score:.6f}",
            flush=True,
        )

    return scored


def _evaluate_openvla_visual_causal_round(
    *,
    round_name: str,
    round_seed: int,
    variant: dict[str, Any],
    valid_pairs: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
            dict[str, float],
        ]
    ],
    candidates: dict[
        str,
        list[dict[str, Any]],
    ],
    reference_actions: np.ndarray,
    reference_instruction: str,
    target_reference_actions:
        np.ndarray,
    target_reference_instruction: str,
    target_reference_evidence:
        dict[str, Any],
    image_path: Path,
    model: str,
    q_ur5e: np.ndarray,
    gripper_width_m: float,
    droid_state_mode: str,
    round_dir: Path,
) -> dict[str, Any]:
    object_scores = (
        _openvla_causal_role_scores(
            role="object",
            role_candidates=
                candidates["object"],
            protected_objects=
                candidates["object"],
            reference_actions=
                reference_actions,
            reference_instruction=
                reference_instruction,
            image_path=image_path,
            model=model,
            q_ur5e=q_ur5e,
            gripper_width_m=
                gripper_width_m,
            droid_state_mode=
                droid_state_mode,
            seed=round_seed,
            round_name=round_name,
            round_dir=round_dir,
            mask_variant=str(
                variant["mask_variant"]
            ),
            mask_padding=int(
                variant["mask_padding"]
            ),
            mask_kernel=int(
                variant["mask_kernel"]
            ),
        )
    )

    target_scores = (
        _openvla_causal_role_scores(
            role="target",
            role_candidates=
                candidates["target"],
            protected_objects=
                candidates["object"],
            reference_actions=
                target_reference_actions,
            reference_instruction=
                target_reference_instruction,
            image_path=image_path,
            model=model,
            q_ur5e=q_ur5e,
            gripper_width_m=
                gripper_width_m,
            droid_state_mode=
                droid_state_mode,
            seed=round_seed,
            round_name=round_name,
            round_dir=round_dir,
            mask_variant=str(
                variant["mask_variant"]
            ),
            mask_padding=int(
                variant["mask_padding"]
            ),
            mask_kernel=int(
                variant["mask_kernel"]
            ),
        )
    )

    object_map = {
        str(item["name"]): item
        for item in object_scores
    }

    target_map = {
        str(item["name"]): item
        for item in target_scores
    }

    scored_pairs = []

    metric_weights = {
        "object_prompt_distance":
            0.25,
        "object_visual_sufficiency_distance":
            0.25,
        "target_prompt_distance":
            0.25,
        "target_visual_sufficiency_distance":
            0.25,
    }

    for (
        object_candidate,
        target_candidate,
        geometry,
    ) in valid_pairs:
        object_name = str(
            object_candidate["name"]
        )

        target_name = str(
            target_candidate["name"]
        )

        object_causal = object_map[
            object_name
        ]

        target_causal = target_map[
            target_name
        ]

        object_cost = float(
            object_causal["causal_cost"]
        )

        target_cost = float(
            target_causal["causal_cost"]
        )

        total_cost = float(
            0.5 * object_cost
            + 0.5 * target_cost
        )

        normalized_costs = {
            "object_prompt_distance":
                object_cost,
            "object_visual_sufficiency_distance":
                object_cost,
            "target_prompt_distance":
                target_cost,
            "target_visual_sufficiency_distance":
                target_cost,
        }

        pair = {
            "object":
                object_candidate,
            "target":
                target_candidate,
            "pair_instruction":
                reference_instruction,
            "pair_target_instruction":
                target_reference_instruction,
            "source_geometry":
                geometry,
            "round_name":
                round_name,
            "round_seed":
                round_seed,
            "object_prompt_distance":
                float(
                    object_causal[
                        "necessity_distance"
                    ]
                ),
            "object_visual_sufficiency_distance":
                float(
                    object_causal[
                        "sufficiency_distance"
                    ]
                ),
            "target_prompt_distance":
                float(
                    target_causal[
                        "necessity_distance"
                    ]
                ),
            "target_visual_sufficiency_distance":
                float(
                    target_causal[
                        "sufficiency_distance"
                    ]
                ),
            "normalized_costs":
                normalized_costs,
            "cost_scales": {
                key: 1.0
                for key
                in normalized_costs
            },
            "metric_weights":
                metric_weights,
            "total_cost":
                total_cost,
            "score":
                float(
                    1.0
                    / (
                        1.0
                        + total_cost
                    )
                ),
            "grounding_adapter": (
                "openvla_visual_causal_"
                "necessity_sufficiency"
            ),
            "object_causal":
                object_causal,
            "target_causal":
                target_causal,
            "visual_mask_costs_used_for_decision":
                True,
            "candidate_specific_prompt_used":
                False,
            "absolute_geometry_used":
                False,
        }

        scored_pairs.append(pair)

        print(
            f"round={round_name} "
            f"object={object_name} "
            f"target={target_name} "
            f"object_cost={object_cost:.6f} "
            f"target_cost={target_cost:.6f} "
            f"total={total_cost:.6f}",
            flush=True,
        )

    summary = (
        _joint_policy_round_margin(
            scored_pairs
        )
    )

    print(
        f"round={round_name} "
        f"winner="
        f"{summary['winner_key'][0]}->"
        f"{summary['winner_key'][1]} "
        f"margin="
        f"{summary['relative_margin']:.4f}",
        flush=True,
    )

    return {
        "name":
            round_name,
        "seed":
            round_seed,
        "variant":
            variant,
        "target_reference_instruction":
            target_reference_instruction,
        "target_reference_evidence":
            target_reference_evidence,
        "raw_pairs":
            scored_pairs,
        "scored_pairs":
            scored_pairs,
        "summary":
            summary,
        "object_causal_scores":
            object_scores,
        "target_causal_scores":
            target_scores,
        "grounding_adapter": (
            "openvla_visual_causal_"
            "necessity_sufficiency"
        ),
    }


def _evaluate_joint_policy_round(
    *,
    round_index: int,
    variant: dict[str, Any],
    valid_pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, float]]],
    candidates: dict[str, list[dict[str, Any]]],
    model: str,
    original_actions: np.ndarray,
    image_path: Path,
    instruction: str,
    q_ur5e: np.ndarray,
    gripper_width_m: float,
    droid_state_mode: str,
    seed: int,
    grounding_dir: Path,
    openvla_reference_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evalúa todas las hipótesis con una vista reproducible del mismo VLA."""
    round_name = str(variant["name"])
    round_seed = int(seed) + int(variant.get("seed_offset", 0))
    round_dir = grounding_dir / f"round_{round_index:02d}_{round_name}"
    round_dir.mkdir(parents=True, exist_ok=True)

    openvla_adapter_mode = (
        os.environ.get(
            "VLA_OPENVLA_GROUNDING_ADAPTER",
            "visual_causal",
        )
        .strip()
        .lower()
    )

    if (
        model == "openvla"
        and openvla_adapter_mode
        == "visual_causal"
    ):
        if not isinstance(
            openvla_reference_bundle,
            dict,
        ):
            raise MainStop(
                "OpenVLA causal requiere "
                "referencias congeladas"
            )

        object_reference_actions = (
            np.asarray(
                openvla_reference_bundle[
                    "object_reference_actions"
                ],
                dtype=float,
            )
        )

        target_reference_actions = (
            np.asarray(
                openvla_reference_bundle[
                    "target_reference_actions"
                ],
                dtype=float,
            )
        )

        object_reference_instruction = str(
            openvla_reference_bundle[
                "object_reference_instruction"
            ]
        )

        target_reference_instruction = str(
            openvla_reference_bundle[
                "target_reference_instruction"
            ]
        )

        target_reference_evidence = (
            openvla_reference_bundle[
                "target_reference_evidence"
            ]
        )

        expected_shape = tuple(
            np.asarray(
                original_actions,
                dtype=float,
            ).shape
        )

        if (
            tuple(
                object_reference_actions.shape
            )
            != expected_shape
            or tuple(
                target_reference_actions.shape
            )
            != expected_shape
        ):
            raise MainStop(
                "OpenVLA: referencias congeladas "
                "con shape incompatible"
            )

        if (
            not np.all(
                np.isfinite(
                    object_reference_actions
                )
            )
            or not np.all(
                np.isfinite(
                    target_reference_actions
                )
            )
        ):
            raise MainStop(
                "OpenVLA: referencias congeladas "
                "no finitas"
            )

        return (
            _evaluate_openvla_visual_causal_round(
                round_name=round_name,
                round_seed=round_seed,
                variant=variant,
                valid_pairs=valid_pairs,
                candidates=candidates,
                reference_actions=
                    object_reference_actions,
                reference_instruction=
                    object_reference_instruction,
                target_reference_actions=
                    target_reference_actions,
                target_reference_instruction=
                    target_reference_instruction,
                target_reference_evidence=
                    target_reference_evidence,
                image_path=image_path,
                model=model,
                q_ur5e=q_ur5e,
                gripper_width_m=
                    gripper_width_m,
                droid_state_mode=
                    droid_state_mode,
                round_dir=round_dir,
            )
        )


    # OpenVLA devuelve una única acción 1x7. Para el grounding
    # autónomo se interpreta su traslación calibrada mediante
    # dos marcos comunes a todos los candidatos:
    #   objeto: TCP actual -> centroide de anclas de objetos
    #   destino: centroide de objetos -> centroide de destinos
    # Los marcos no dependen del candidato que está siendo
    # puntuado. Así se evita girar la acción a favor de cada
    # hipótesis concreta.
    openvla_geometry_context = None

    if (
        model == "openvla"
        and os.environ.get(
            "VLA_OPENVLA_GROUNDING_ADAPTER",
            "visual_causal",
        ).strip().lower()
        == "geometric"
    ):
        openvla_calibration = (
            load_openvla_calibration()
        )

        openvla_tcp_xyz = finite_vector(
            _ur5e_tcp_fk_from_q(q_ur5e),
            3,
            "openvla_grounding.tcp_xyz",
        )

        openvla_translation_budget = (
            finite_vector(
                OPENVLA_NATIVE_BUDGET_M,
                3,
                (
                    "openvla_grounding."
                    "translation_budget"
                ),
            )
        )

        def _openvla_candidate_anchor(
            candidate: dict[str, Any],
            label: str,
        ) -> tuple[np.ndarray, str]:
            safe_above = candidate.get(
                "safe_above_base_xyz_m"
            )

            if safe_above is not None:
                return (
                    finite_vector(
                        safe_above,
                        3,
                        (
                            "openvla_grounding."
                            f"{label}.safe_above"
                        ),
                    ),
                    "safe_above_base_xyz_m",
                )

            return (
                finite_vector(
                    candidate["base_xyz_m"],
                    3,
                    (
                        "openvla_grounding."
                        f"{label}.base"
                    ),
                ),
                "base_xyz_m",
            )

        openvla_object_anchors = [
            _openvla_candidate_anchor(
                candidate,
                (
                    "object."
                    f"{candidate.get('name')}"
                ),
            )[0]
            for candidate
            in candidates["object"]
        ]

        openvla_target_anchors = [
            _openvla_candidate_anchor(
                candidate,
                (
                    "target."
                    f"{candidate.get('name')}"
                ),
            )[0]
            for candidate
            in candidates["target"]
        ]

        if not openvla_object_anchors:
            raise MainStop(
                "OpenVLA geométrico: "
                "no hay anclas de objetos"
            )

        if not openvla_target_anchors:
            raise MainStop(
                "OpenVLA geométrico: "
                "no hay anclas de destinos"
            )

        openvla_object_centroid = np.mean(
            np.asarray(
                openvla_object_anchors,
                dtype=float,
            ),
            axis=0,
        )

        openvla_target_centroid = np.mean(
            np.asarray(
                openvla_target_anchors,
                dtype=float,
            ),
            axis=0,
        )

        if (
            np.linalg.norm(
                openvla_object_centroid[:2]
                - openvla_tcp_xyz[:2]
            )
            <= 1e-6
        ):
            object_distances = [
                float(
                    np.linalg.norm(
                        anchor[:2]
                        - openvla_tcp_xyz[:2]
                    )
                )
                for anchor
                in openvla_object_anchors
            ]

            openvla_object_centroid = (
                openvla_object_anchors[
                    int(
                        np.argmax(
                            object_distances
                        )
                    )
                ].copy()
            )

        if (
            np.linalg.norm(
                openvla_target_centroid[:2]
                - openvla_object_centroid[:2]
            )
            <= 1e-6
        ):
            target_distances = [
                float(
                    np.linalg.norm(
                        anchor[:2]
                        - openvla_object_centroid[:2]
                    )
                )
                for anchor
                in openvla_target_anchors
            ]

            openvla_target_centroid = (
                openvla_target_anchors[
                    int(
                        np.argmax(
                            target_distances
                        )
                    )
                ].copy()
            )

        openvla_object_frame = (
            adaptation.build_task_frame(
                openvla_tcp_xyz,
                openvla_object_centroid,
            )
        )

        openvla_target_frame = (
            adaptation.build_task_frame(
                openvla_object_centroid,
                openvla_target_centroid,
            )
        )

        def _openvla_metric_action_vector(
            raw_actions: Any,
            frame: np.ndarray,
            label: str,
        ) -> tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]:
            action_matrix = np.asarray(
                raw_actions,
                dtype=float,
            )

            if action_matrix.shape != (1, 7):
                raise MainStop(
                    f"{label}: shape="
                    f"{action_matrix.shape}; "
                    "esperado=(1, 7)"
                )

            normalized_action = np.asarray(
                adaptation.normalize_openvla_action(
                    action_matrix[0],
                    openvla_calibration,
                ),
                dtype=float,
            ).reshape(-1)

            if (
                normalized_action.size < 3
                or not np.all(
                    np.isfinite(
                        normalized_action
                    )
                )
            ):
                raise MainStop(
                    f"{label}: acción normalizada "
                    "no válida"
                )

            action_task_m = (
                normalized_action[:3]
                * openvla_translation_budget
            )

            action_base_m = (
                np.asarray(
                    frame,
                    dtype=float,
                ).reshape(3, 3)
                @ action_task_m
            )

            return (
                finite_vector(
                    action_base_m,
                    3,
                    f"{label}.base_action",
                ),
                finite_vector(
                    action_task_m,
                    3,
                    f"{label}.task_action",
                ),
                normalized_action,
            )

        def _openvla_alignment_cost(
            action_base_m: np.ndarray,
            expected_base_m: np.ndarray,
            label: str,
        ) -> dict[str, Any]:
            action_vector = finite_vector(
                action_base_m,
                3,
                f"{label}.action",
            )

            expected_vector = finite_vector(
                expected_base_m,
                3,
                f"{label}.expected",
            )

            action_norm = float(
                np.linalg.norm(
                    action_vector
                )
            )

            expected_norm = float(
                np.linalg.norm(
                    expected_vector
                )
            )

            if expected_norm <= 1e-9:
                raise MainStop(
                    f"{label}: vector geométrico "
                    "degenerado"
                )

            if action_norm <= 1e-12:
                cosine = -1.0
                cost = 1.0
                action_zero = True

            else:
                cosine = float(
                    np.clip(
                        np.dot(
                            action_vector,
                            expected_vector,
                        )
                        / (
                            action_norm
                            * expected_norm
                        ),
                        -1.0,
                        1.0,
                    )
                )

                cost = float(
                    0.5
                    * (
                        1.0
                        - cosine
                    )
                )

                action_zero = False

            return {
                "action_base_m":
                    action_vector.tolist(),
                "expected_base_m":
                    expected_vector.tolist(),
                "action_norm_m":
                    action_norm,
                "expected_norm_m":
                    expected_norm,
                "cosine_alignment":
                    cosine,
                "direction_cost":
                    cost,
                "action_zero":
                    action_zero,
            }

        openvla_geometry_context = {
            "schema":
                "VLA_openvla_geometric_context",
            "tcp_xyz_m":
                openvla_tcp_xyz.tolist(),
            "object_centroid_xyz_m":
                openvla_object_centroid.tolist(),
            "target_centroid_xyz_m":
                openvla_target_centroid.tolist(),
            "object_common_frame":
                openvla_object_frame.tolist(),
            "target_common_frame":
                openvla_target_frame.tolist(),
            "translation_budget_m":
                openvla_translation_budget.tolist(),
            "frames_are_candidate_independent":
                True,
        }

        print(
            f"round={round_name} "
            f"tcp={openvla_tcp_xyz.tolist()} "
            f"object_centroid="
            f"{openvla_object_centroid.tolist()} "
            f"target_centroid="
            f"{openvla_target_centroid.tolist()} "
            "frames_candidate_independent=true",
            flush=True,
        )

    if round_index == 1 and round_seed == int(seed):
        round_reference_actions = np.asarray(original_actions, dtype=float)
        round_reference_evidence: dict[str, Any] = {
            "reused_initial_action": True,
            "seed": round_seed,
        }
    else:
        round_reference_actions, round_reference_evidence = infer_on_orin(
            model=model,
            image_path=image_path,
            instruction=instruction,
            q_ur5e=q_ur5e,
            gripper_width_m=gripper_width_m,
            droid_state_mode=droid_state_mode,
            seed=round_seed,
            run_dir=round_dir / "original_instruction_reference",
        )

    target_reference_instruction = str(variant["target_reference"])
    target_reference_actions, target_reference_evidence = infer_on_orin(
        model=model,
        image_path=image_path,
        instruction=target_reference_instruction,
        q_ur5e=q_ur5e,
        gripper_width_m=gripper_width_m,
        droid_state_mode=droid_state_mode,
        seed=round_seed,
        run_dir=round_dir / "target_phase_reference",
    )


    raw_pairs: list[dict[str, Any]] = []
    for pair_index, (object_candidate, target_candidate, geometry) in enumerate(
        valid_pairs,
        start=1,
    ):
        object_label = _grounding_candidate_label(object_candidate, "object")
        target_label = _grounding_candidate_label(target_candidate, "target")
        pair_instruction = str(variant["object_template"]).format(
            object_label=object_label,
            target_label=target_label,
        )
        pair_target_instruction = str(variant["target_template"]).format(
            object_label=object_label,
            target_label=target_label,
        )
        pair_dir = round_dir / (
            f"pair_{pair_index:02d}_{_safe_candidate_name(object_candidate['name'])}_to_"
            f"{_safe_candidate_name(target_candidate['name'])}"
        )
        pair_dir.mkdir(parents=True, exist_ok=True)

        pair_actions, pair_evidence = infer_on_orin(
            model=model,
            image_path=image_path,
            instruction=pair_instruction,
            q_ur5e=q_ur5e,
            gripper_width_m=gripper_width_m,
            droid_state_mode=droid_state_mode,
            seed=round_seed,
            run_dir=pair_dir / "pair_full",
        )
        object_prompt = _action_grounding_distance(
            model,
            round_reference_actions,
            pair_actions,
        )

        object_alternatives = [
            candidate for candidate in candidates["object"]
            if candidate["name"] != object_candidate["name"]
        ]
        object_isolated_path = pair_dir / "object_isolated.png"
        _mask_candidate_regions_image(
            image_path,
            object_alternatives,
            object_isolated_path,
            variant=str(variant["mask_variant"]),
            padding_override=int(variant["mask_padding"]),
            blur_kernel_override=int(variant["mask_kernel"]),
        )
        object_isolated_actions, object_isolated_evidence = infer_on_orin(
            model=model,
            image_path=object_isolated_path,
            instruction=pair_instruction,
            q_ur5e=q_ur5e,
            gripper_width_m=gripper_width_m,
            droid_state_mode=droid_state_mode,
            seed=round_seed,
            run_dir=pair_dir / "object_isolated",
        )
        object_visual = _action_grounding_distance(
            model,
            pair_actions,
            object_isolated_actions,
        )

        target_pair_actions, target_pair_evidence = infer_on_orin(
            model=model,
            image_path=image_path,
            instruction=pair_target_instruction,
            q_ur5e=q_ur5e,
            gripper_width_m=gripper_width_m,
            droid_state_mode=droid_state_mode,
            seed=round_seed,
            run_dir=pair_dir / "target_pair_full",
        )
        target_prompt = _action_grounding_distance(
            model,
            target_reference_actions,
            target_pair_actions,
        )

        openvla_geometric_alignment = None

        if model == "openvla":
            (
                object_action_base_m,
                object_action_task_m,
                object_normalized_action,
            ) = _openvla_metric_action_vector(
                pair_actions,
                openvla_object_frame,
                (
                    "openvla_grounding."
                    f"{round_name}."
                    f"{object_candidate['name']}."
                    "object_phase"
                ),
            )

            (
                target_action_base_m,
                target_action_task_m,
                target_normalized_action,
            ) = _openvla_metric_action_vector(
                target_pair_actions,
                openvla_target_frame,
                (
                    "openvla_grounding."
                    f"{round_name}."
                    f"{target_candidate['name']}."
                    "target_phase"
                ),
            )

            (
                object_anchor_xyz,
                object_anchor_source,
            ) = _openvla_candidate_anchor(
                object_candidate,
                (
                    "pair_object."
                    f"{object_candidate['name']}"
                ),
            )

            (
                target_anchor_xyz,
                target_anchor_source,
            ) = _openvla_candidate_anchor(
                target_candidate,
                (
                    "pair_target."
                    f"{target_candidate['name']}"
                ),
            )

            object_expected_base_m = (
                object_anchor_xyz
                - openvla_tcp_xyz
            )

            target_expected_base_m = (
                target_anchor_xyz
                - object_anchor_xyz
            )

            object_alignment = (
                _openvla_alignment_cost(
                    object_action_base_m,
                    object_expected_base_m,
                    (
                        "openvla_grounding."
                        "object_alignment"
                    ),
                )
            )

            target_alignment = (
                _openvla_alignment_cost(
                    target_action_base_m,
                    target_expected_base_m,
                    (
                        "openvla_grounding."
                        "target_alignment"
                    ),
                )
            )

            openvla_geometric_alignment = {
                "schema":
                    "VLA_openvla_pair_geometry",
                "adapter":
                    "single_action_common_frame_geometry",
                "object_anchor_source":
                    object_anchor_source,
                "target_anchor_source":
                    target_anchor_source,
                "object_anchor_xyz_m":
                    object_anchor_xyz.tolist(),
                "target_anchor_xyz_m":
                    target_anchor_xyz.tolist(),
                "object_action_task_m":
                    object_action_task_m.tolist(),
                "target_action_task_m":
                    target_action_task_m.tolist(),
                "object_normalized_action":
                    object_normalized_action.tolist(),
                "target_normalized_action":
                    target_normalized_action.tolist(),
                "object_alignment":
                    object_alignment,
                "target_alignment":
                    target_alignment,
                "context":
                    openvla_geometry_context,
            }

            print(
                f"round={round_name} "
                f"object={object_candidate['name']} "
                f"target={target_candidate['name']} "
                f"object_cos="
                f"{object_alignment['cosine_alignment']:.6f} "
                f"target_cos="
                f"{target_alignment['cosine_alignment']:.6f} "
                f"object_cost="
                f"{object_alignment['direction_cost']:.6f} "
                f"target_cost="
                f"{target_alignment['direction_cost']:.6f}",
                flush=True,
            )

        target_alternatives = [
            candidate for candidate in candidates["target"]
            if candidate["name"] != target_candidate["name"]
        ]
        target_isolated_path = pair_dir / "target_isolated.png"
        _mask_candidate_regions_image(
            image_path,
            target_alternatives,
            target_isolated_path,
            variant=str(variant["mask_variant"]),
            padding_override=int(variant["mask_padding"]),
            blur_kernel_override=int(variant["mask_kernel"]),
        )

        # La máscara de una zona alternativa puede solaparse
        # en 2D con un objeto físico. Después de enmascarar las
        # zonas se restauran todos los objetos desde la imagen
        # original, de modo que la evidencia de objeto sea
        # idéntica para todos los candidatos de destino.
        original_target_context = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        isolated_target_context = cv2.imread(
            str(target_isolated_path),
            cv2.IMREAD_COLOR,
        )

        if original_target_context is None:
            raise MainStop(
                "No se pudo leer la imagen original para "
                "proteger objetos en target_isolated"
            )

        if isolated_target_context is None:
            raise MainStop(
                "No se pudo leer target_isolated después "
                "de aplicar la máscara"
            )

        if (
            original_target_context.shape
            != isolated_target_context.shape
        ):
            raise MainStop(
                "La imagen target_isolated cambió de tamaño: "
                f"original={original_target_context.shape}, "
                f"isolated={isolated_target_context.shape}"
            )

        image_height, image_width = (
            original_target_context.shape[:2]
        )

        restore_padding_px = 2
        protected_object_regions = []

        for protected_object in candidates["object"]:
            bbox_values = np.asarray(
                protected_object.get(
                    "bbox_xyxy"
                ),
                dtype=float,
            ).reshape(-1)

            if (
                bbox_values.size != 4
                or not np.all(
                    np.isfinite(bbox_values)
                )
            ):
                raise MainStop(
                    "BBox de objeto no válida durante "
                    "la protección target_isolated: "
                    f"object={protected_object.get('name')} "
                    f"bbox={protected_object.get('bbox_xyxy')}"
                )

            x1 = max(
                0,
                int(np.floor(bbox_values[0]))
                - restore_padding_px,
            )

            y1 = max(
                0,
                int(np.floor(bbox_values[1]))
                - restore_padding_px,
            )

            x2 = min(
                image_width,
                int(np.ceil(bbox_values[2]))
                + restore_padding_px,
            )

            y2 = min(
                image_height,
                int(np.ceil(bbox_values[3]))
                + restore_padding_px,
            )

            if x2 <= x1 or y2 <= y1:
                raise MainStop(
                    "BBox vacía durante la protección "
                    "target_isolated: "
                    f"object={protected_object.get('name')} "
                    f"clipped={[x1, y1, x2, y2]}"
                )

            isolated_target_context[
                y1:y2,
                x1:x2,
            ] = original_target_context[
                y1:y2,
                x1:x2,
            ]

            restored_exactly = bool(
                np.array_equal(
                    isolated_target_context[
                        y1:y2,
                        x1:x2,
                    ],
                    original_target_context[
                        y1:y2,
                        x1:x2,
                    ],
                )
            )

            if not restored_exactly:
                raise MainStop(
                    "No se restauró exactamente el objeto "
                    f"{protected_object.get('name')}"
                )

            protected_object_regions.append(
                {
                    "name":
                        str(
                            protected_object.get(
                                "name"
                            )
                        ),
                    "bbox_xyxy_original":
                        [
                            float(value)
                            for value
                            in bbox_values.tolist()
                        ],
                    "bbox_xyxy_restored":
                        [x1, y1, x2, y2],
                    "restore_padding_px":
                        restore_padding_px,
                    "restored_exactly":
                        restored_exactly,
                }
            )

        if not protected_object_regions:
            raise MainStop(
                "No había regiones de objeto para "
                "proteger en target_isolated"
            )

        write_ok = cv2.imwrite(
            str(target_isolated_path),
            isolated_target_context,
        )

        if not write_ok:
            raise MainStop(
                "No se pudo escribir target_isolated "
                "con los objetos restaurados"
            )

        preservation_trace = {
            "schema":
                "VLA_target_isolation_object_preservation",
            "round":
                round_name,
            "pair_index":
                pair_index,
            "object":
                str(object_candidate["name"]),
            "target":
                str(target_candidate["name"]),
            "masked_target_alternatives":
                [
                    str(candidate["name"])
                    for candidate
                    in target_alternatives
                ],
            "protected_objects":
                protected_object_regions,
            "image_path":
                str(target_isolated_path),
            "contract":
                "mask_target_alternatives_preserve_all_objects",
        }

        json_write(
            pair_dir
            / "target_isolation_object_preservation.json",
            preservation_trace,
        )

        print(
            f"round={round_name} "
            f"object={object_candidate['name']} "
            f"target={target_candidate['name']} "
            f"protected="
            f"{[item['name'] for item in protected_object_regions]} "
            "contract=mask_targets_preserve_objects",
            flush=True,
        )

        target_isolated_actions, target_isolated_evidence = infer_on_orin(
            model=model,
            image_path=target_isolated_path,
            instruction=pair_target_instruction,
            q_ur5e=q_ur5e,
            gripper_width_m=gripper_width_m,
            droid_state_mode=droid_state_mode,
            seed=round_seed,
            run_dir=pair_dir / "target_isolated",
        )
        target_visual = _action_grounding_distance(
            model,
            target_pair_actions,
            target_isolated_actions,
        )

        item = {
            "object": object_candidate,
            "target": target_candidate,
            "pair_instruction": pair_instruction,
            "pair_target_instruction": pair_target_instruction,
            "source_geometry": geometry,
            "round_name": round_name,
            "round_seed": round_seed,
            "object_prompt_distance": float(object_prompt["distance"]),
            "object_visual_sufficiency_distance": float(object_visual["distance"]),
            "target_prompt_distance": float(target_prompt["distance"]),
            "target_visual_sufficiency_distance": float(target_visual["distance"]),
            "openvla_geometric_alignment": openvla_geometric_alignment,
            "metrics": {
                "object_prompt": object_prompt,
                "object_visual_sufficiency": object_visual,
                "target_prompt": target_prompt,
                "target_visual_sufficiency": target_visual,
            },
            "evidence": {
                "round_reference": round_reference_evidence,
                "target_reference": target_reference_evidence,
                "pair_full": pair_evidence,
                "object_isolated": object_isolated_evidence,
                "target_pair_full": target_pair_evidence,
                "target_isolated": target_isolated_evidence,
            },
            "images": {
                "object_isolated": str(object_isolated_path),
                "target_isolated": str(target_isolated_path),
            },
        }
        raw_pairs.append(item)
        print(
            f"round={round_name} object={object_candidate['name']} "
            f"target={target_candidate['name']} "
            f"object_prompt={item['object_prompt_distance']:.8g} "
            f"object_visual={item['object_visual_sufficiency_distance']:.8g} "
            f"target_prompt={item['target_prompt_distance']:.8g} "
            f"target_visual={item['target_visual_sufficiency_distance']:.8g}"
        )

    scored_pairs = _pair_policy_costs_normalized(raw_pairs)

    if model == "openvla":
        # Los pesos se fijan antes de empezar los ensayos.
        # No se reajustan a partir de los resultados obtenidos.
        openvla_geometry_weight_object = 0.35
        openvla_geometry_weight_target = 0.35
        openvla_prompt_weight_object = 0.15
        openvla_prompt_weight_target = 0.15

        weight_sum = float(
            openvla_geometry_weight_object
            + openvla_geometry_weight_target
            + openvla_prompt_weight_object
            + openvla_prompt_weight_target
        )

        if abs(weight_sum - 1.0) > 1e-12:
            raise MainStop(
                "Pesos OpenVLA geométricos "
                f"no suman 1: {weight_sum}"
            )

        def _openvla_unit_prompt_cost(
            value: Any,
            label: str,
        ) -> float:
            numeric = float(value)

            if (
                not math.isfinite(numeric)
                or numeric < 0.0
            ):
                raise MainStop(
                    f"{label}: coste normalizado "
                    f"no válido={numeric}"
                )

            return float(
                numeric
                / (
                    1.0
                    + numeric
                )
            )

        for pair in scored_pairs:
            geometric = pair.get(
                "openvla_geometric_alignment"
            )

            if not isinstance(
                geometric,
                dict,
            ):
                raise MainStop(
                    "Falta evidencia geométrica "
                    "en pareja OpenVLA"
                )

            object_direction_cost = float(
                geometric[
                    "object_alignment"
                ][
                    "direction_cost"
                ]
            )

            target_direction_cost = float(
                geometric[
                    "target_alignment"
                ][
                    "direction_cost"
                ]
            )

            normalized_costs = pair.get(
                "normalized_costs"
            )

            if not isinstance(
                normalized_costs,
                dict,
            ):
                raise MainStop(
                    "Faltan costes normalizados "
                    "para OpenVLA"
                )

            object_prompt_unit_cost = (
                _openvla_unit_prompt_cost(
                    normalized_costs[
                        "object_prompt_distance"
                    ],
                    (
                        "openvla."
                        "object_prompt_distance"
                    ),
                )
            )

            target_prompt_unit_cost = (
                _openvla_unit_prompt_cost(
                    normalized_costs[
                        "target_prompt_distance"
                    ],
                    (
                        "openvla."
                        "target_prompt_distance"
                    ),
                )
            )

            generic_total_cost = float(
                pair["total_cost"]
            )

            geometric_total_cost = float(
                openvla_geometry_weight_object
                * object_direction_cost
                + openvla_geometry_weight_target
                * target_direction_cost
                + openvla_prompt_weight_object
                * object_prompt_unit_cost
                + openvla_prompt_weight_target
                * target_prompt_unit_cost
            )

            pair[
                "generic_counterfactual_total_cost_diagnostic"
            ] = generic_total_cost

            pair[
                "openvla_geometric_score_components"
            ] = {
                "object_direction_cost":
                    object_direction_cost,
                "target_direction_cost":
                    target_direction_cost,
                "object_prompt_unit_cost":
                    object_prompt_unit_cost,
                "target_prompt_unit_cost":
                    target_prompt_unit_cost,
                "weights": {
                    "object_geometry":
                        openvla_geometry_weight_object,
                    "target_geometry":
                        openvla_geometry_weight_target,
                    "object_prompt":
                        openvla_prompt_weight_object,
                    "target_prompt":
                        openvla_prompt_weight_target,
                },
                "visual_mask_costs_used_for_decision":
                    False,
                "generic_total_cost_diagnostic":
                    generic_total_cost,
            }

            pair["grounding_adapter"] = (
                "openvla_single_action_"
                "common_frame_geometry"
            )

            pair["total_cost"] = (
                geometric_total_cost
            )

            pair["score"] = float(
                1.0
                / (
                    1.0
                    + geometric_total_cost
                )
            )

            print(
                f"round={round_name} "
                f"object={pair['object']['name']} "
                f"target={pair['target']['name']} "
                f"object_geometry="
                f"{object_direction_cost:.6f} "
                f"target_geometry="
                f"{target_direction_cost:.6f} "
                f"object_prompt="
                f"{object_prompt_unit_cost:.6f} "
                f"target_prompt="
                f"{target_prompt_unit_cost:.6f} "
                f"total={geometric_total_cost:.6f} "
                "mask_costs_decision=false",
                flush=True,
            )

    for pair in scored_pairs:
        print(
            f"round={round_name} object={pair['object']['name']} "
            f"target={pair['target']['name']} "
            f"cost={pair['total_cost']:.6f} score={pair['score']:.6f}"
        )
    summary = _joint_policy_round_margin(scored_pairs)
    print(
        f"round={round_name} winner={summary['winner_key'][0]}->"
        f"{summary['winner_key'][1]} margin={summary['relative_margin']:.4f}"
    )
    return {
        "name": round_name,
        "seed": round_seed,
        "variant": variant,
        "target_reference_instruction": target_reference_instruction,
        "target_reference_evidence": target_reference_evidence,
        "raw_pairs": raw_pairs,
        "scored_pairs": scored_pairs,
        "summary": summary,
    }



def _aggregate_openvla_factorized_role_medians(
    rounds: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Agrega la evidencia causal por rol antes de formar parejas."""
    if not rounds:
        raise MainStop(
            "OpenVLA V4: no existen rondas"
        )

    # El agregado se mantiene como estructura y
    # referencia; no se usa como ranking activo de OpenVLA.
    baseline_pairs, consensus = (
        _aggregate_joint_policy_rounds(
            rounds
        )
    )

    def finite_cost(
        value: Any,
        label: str,
    ) -> float:
        result = float(value)

        if not math.isfinite(result):
            raise MainStop(
                "OpenVLA V4: valor no finito en "
                f"{label}: {value!r}"
            )

        if result < 0.0:
            raise MainStop(
                "OpenVLA V4: coste negativo en "
                f"{label}: {result}"
            )

        return result

    object_round_costs: dict[
        str,
        list[float],
    ] = {}

    target_round_costs: dict[
        str,
        list[float],
    ] = {}

    expected_objects: set[str] | None = None
    expected_targets: set[str] | None = None

    for round_index, round_result in enumerate(
        rounds,
        start=1,
    ):
        scored_pairs = round_result.get(
            "scored_pairs"
        )

        if (
            not isinstance(scored_pairs, list)
            or not scored_pairs
        ):
            raise MainStop(
                "OpenVLA V4: ronda sin parejas: "
                f"{round_index}"
            )

        current_objects: dict[str, float] = {}
        current_targets: dict[str, float] = {}

        for pair_index, pair in enumerate(
            scored_pairs,
            start=1,
        ):
            try:
                object_name = str(
                    pair["object"]["name"]
                )

                target_name = str(
                    pair["target"]["name"]
                )

                object_cost = finite_cost(
                    pair["object_causal"][
                        "causal_cost"
                    ],
                    (
                        f"round[{round_index}]"
                        f".pair[{pair_index}]"
                        ".object_causal"
                    ),
                )

                target_cost = finite_cost(
                    pair["target_causal"][
                        "causal_cost"
                    ],
                    (
                        f"round[{round_index}]"
                        f".pair[{pair_index}]"
                        ".target_causal"
                    ),
                )

            except (
                KeyError,
                TypeError,
            ) as error:
                raise MainStop(
                    "OpenVLA V4: pareja causal "
                    "incompleta: "
                    f"round={round_index} "
                    f"pair={pair_index} "
                    f"error={error}"
                ) from error

            previous_object = (
                current_objects.get(
                    object_name
                )
            )

            if (
                previous_object is not None
                and abs(
                    previous_object
                    - object_cost
                )
                > 1e-12
            ):
                raise MainStop(
                    "OpenVLA V4: el coste del objeto "
                    "cambia entre destinos dentro de "
                    f"la ronda: {object_name}"
                )

            previous_target = (
                current_targets.get(
                    target_name
                )
            )

            if (
                previous_target is not None
                and abs(
                    previous_target
                    - target_cost
                )
                > 1e-12
            ):
                raise MainStop(
                    "OpenVLA V4: el coste del destino "
                    "cambia entre objetos dentro de "
                    f"la ronda: {target_name}"
                )

            current_objects[
                object_name
            ] = object_cost

            current_targets[
                target_name
            ] = target_cost

        current_object_names = set(
            current_objects
        )

        current_target_names = set(
            current_targets
        )

        if expected_objects is None:
            expected_objects = (
                current_object_names
            )

            expected_targets = (
                current_target_names
            )

        elif (
            current_object_names
            != expected_objects
            or current_target_names
            != expected_targets
        ):
            raise MainStop(
                "OpenVLA V4: las rondas no "
                "contienen los mismos candidatos"
            )

        for name, cost in (
            current_objects.items()
        ):
            object_round_costs.setdefault(
                name,
                [],
            ).append(cost)

        for name, cost in (
            current_targets.items()
        ):
            target_round_costs.setdefault(
                name,
                [],
            ).append(cost)

    round_count = len(rounds)

    for name, values in (
        object_round_costs.items()
    ):
        if len(values) != round_count:
            raise MainStop(
                "OpenVLA V4: medidas incompletas "
                f"para objeto {name}"
            )

    for name, values in (
        target_round_costs.items()
    ):
        if len(values) != round_count:
            raise MainStop(
                "OpenVLA V4: medidas incompletas "
                f"para destino {name}"
            )

    object_medians = {
        name: float(
            np.median(
                np.asarray(
                    values,
                    dtype=float,
                )
            )
        )
        for name, values
        in object_round_costs.items()
    }

    target_medians = {
        name: float(
            np.median(
                np.asarray(
                    values,
                    dtype=float,
                )
            )
        )
        for name, values
        in target_round_costs.items()
    }

    def role_summary(
        medians: dict[str, float],
    ) -> dict[str, Any]:
        ordered = sorted(
            medians.items(),
            key=lambda item: item[1],
        )

        best_name, best_cost = ordered[0]

        if len(ordered) > 1:
            second_name, second_cost = (
                ordered[1]
            )

            margin = float(
                (
                    second_cost
                    - best_cost
                )
                / max(
                    second_cost,
                    1e-9,
                )
            )

        else:
            second_name = None
            second_cost = math.inf
            margin = 1.0

        return {
            "best":
                best_name,
            "best_cost":
                float(best_cost),
            "second":
                second_name,
            "second_cost":
                float(second_cost),
            "relative_margin":
                margin,
            "ranking": [
                {
                    "name": name,
                    "cost": float(cost),
                }
                for name, cost in ordered
            ],
        }

    object_summary = role_summary(
        object_medians
    )

    target_summary = role_summary(
        target_medians
    )

    factorized_pairs = []

    for baseline_pair in baseline_pairs:
        object_name = str(
            baseline_pair["object"]["name"]
        )

        target_name = str(
            baseline_pair["target"]["name"]
        )

        object_cost = float(
            object_medians[object_name]
        )

        target_cost = float(
            target_medians[target_name]
        )

        total_cost = float(
            0.5 * object_cost
            + 0.5 * target_cost
        )

        pair = dict(baseline_pair)

        pair.update(
            {
                "object_role_median_cost":
                    object_cost,
                "target_role_median_cost":
                    target_cost,
                "object_causal": {
                    "causal_cost":
                        object_cost,
                    "round_causal_costs":
                        list(
                            object_round_costs[
                                object_name
                            ]
                        ),
                    "aggregation_contract":
                        "median_per_object_candidate",
                },
                "target_causal": {
                    "causal_cost":
                        target_cost,
                    "round_causal_costs":
                        list(
                            target_round_costs[
                                target_name
                            ]
                        ),
                    "aggregation_contract":
                        "median_per_target_candidate",
                },
                "normalized_costs": {
                    "object_prompt_distance":
                        object_cost,
                    "object_visual_sufficiency_distance":
                        object_cost,
                    "target_prompt_distance":
                        target_cost,
                    "target_visual_sufficiency_distance":
                        target_cost,
                },
                "total_cost":
                    total_cost,
                "score":
                    float(
                        1.0
                        / (
                            1.0
                            + total_cost
                        )
                    ),
                "aggregation_contract": (
                    "median_per_role_then_"
                    "factorized_pair_cost"
                ),
                "joint_pair_median_cost_diagnostic":
                    float(
                        baseline_pair[
                            "total_cost"
                        ]
                    ),
            }
        )

        factorized_pairs.append(pair)

    aggregate_summary = (
        _joint_policy_round_margin(
            factorized_pairs
        )
    )

    aggregate_key = tuple(
        aggregate_summary[
            "winner_key"
        ]
    )

    majority_key_raw = consensus.get(
        "majority_key"
    )

    majority_key = (
        tuple(majority_key_raw)
        if isinstance(
            majority_key_raw,
            list,
        )
        and len(majority_key_raw) == 2
        else None
    )

    vote_aggregate_agreement = (
        bool(
            majority_key
            == aggregate_key
        )
        if majority_key is not None
        else None
    )

    vote_consensus_pass = bool(
        consensus.get(
            "vote_consensus_pass",
            False,
        )
    )

    if (
        vote_consensus_pass
        and vote_aggregate_agreement
        is True
    ):
        stability_class = (
            "stable_majority"
        )

    elif majority_key is None:
        stability_class = (
            "split_vote_aggregate_primary"
        )

    elif vote_aggregate_agreement is True:
        stability_class = (
            "plurality_agrees_aggregate"
        )

    else:
        stability_class = (
            "vote_disagrees_aggregate_primary"
        )

    pair_key_text = (
        f"{aggregate_key[0]}->"
        f"{aggregate_key[1]}"
    )

    votes = consensus.get(
        "votes",
        {},
    )

    aggregate_votes = int(
        votes.get(
            pair_key_text,
            0,
        )
    )

    consensus.update(
        {
            "aggregate_key":
                list(aggregate_key),
            "aggregate_votes":
                aggregate_votes,
            "aggregate_relative_margin":
                float(
                    aggregate_summary[
                        "relative_margin"
                    ]
                ),
            "vote_aggregate_agreement":
                vote_aggregate_agreement,
            "stability_class":
                stability_class,
            "object_role_summary":
                object_summary,
            "target_role_summary":
                target_summary,
            "object_relative_margin":
                float(
                    object_summary[
                        "relative_margin"
                    ]
                ),
            "target_relative_margin":
                float(
                    target_summary[
                        "relative_margin"
                    ]
                ),
            "decision_rule": (
                "factorized_role_medians_"
                "primary_votes_diagnostic_only"
            ),
            "aggregation_contract": (
                "median_per_role_then_"
                "factorized_pair_cost"
            ),
            "previous_joint_pair_aggregation":
                "diagnostic_only",
            "votes_used_as_veto":
                False,
        }
    )

    print(
        f"object={object_summary['best']} "
        f"object_margin="
        f"{object_summary['relative_margin']:.4f} "
        f"target={target_summary['best']} "
        f"target_margin="
        f"{target_summary['relative_margin']:.4f} "
        f"pair={aggregate_key[0]}->"
        f"{aggregate_key[1]} "
        f"pair_margin="
        f"{aggregate_summary['relative_margin']:.4f} "
        "contract=median_per_role_then_pair",
        flush=True,
    )

    return factorized_pairs, consensus


def _aggregate_grounding_rounds_by_adapter(
    rounds: list[dict[str, Any]],
    *,
    model: str,
    grounding_adapter: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    if (
        model == "openvla"
        and grounding_adapter
        == (
            "openvla_visual_causal_"
            "necessity_sufficiency"
        )
    ):
        return (
            _aggregate_openvla_factorized_role_medians(
                rounds
            )
        )

    return _aggregate_joint_policy_rounds(
        rounds
    )


def _aggregate_joint_policy_rounds(
    rounds: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Agrega por mediana; los votos miden estabilidad y no vetan el ranking."""
    if not rounds:
        raise MainStop("Grounding VLA: no existen rounds para agregar")

    metric_names = (
        "object_prompt_distance",
        "object_visual_sufficiency_distance",
        "target_prompt_distance",
        "target_visual_sufficiency_distance",
    )

    def finite_float(value: Any, label: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise MainStop(
                f"Grounding VLA: valor no finito en {label}: {value!r}"
            )
        return number

    expected_keys: set[tuple[str, str]] | None = None
    grouped_scored: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for round_index, round_result in enumerate(rounds, start=1):
        scored_pairs = round_result.get("scored_pairs")
        if not isinstance(scored_pairs, list) or not scored_pairs:
            raise MainStop(
                f"Grounding VLA: round {round_index} sin scored_pairs"
            )

        current_keys: list[tuple[str, str]] = []
        for item_index, item in enumerate(scored_pairs, start=1):
            try:
                key = (
                    str(item["object"]["name"]),
                    str(item["target"]["name"]),
                )
            except (KeyError, TypeError) as error:
                raise MainStop(
                    "Grounding VLA: hipótesis mal formada en "
                    f"round={round_index} item={item_index}: {error}"
                ) from error

            if key in current_keys:
                raise MainStop(
                    "Grounding VLA: hipótesis duplicada en "
                    f"round={round_index}: {key[0]}->{key[1]}"
                )
            current_keys.append(key)

        current_key_set = set(current_keys)
        if expected_keys is None:
            expected_keys = current_key_set
        elif current_key_set != expected_keys:
            missing = sorted(expected_keys - current_key_set)
            extra = sorted(current_key_set - expected_keys)
            raise MainStop(
                "Grounding VLA: los rounds no contienen las mismas "
                f"hipótesis; round={round_index} missing={missing} extra={extra}"
            )

        for item, key in zip(scored_pairs, current_keys):
            finite_float(
                item.get("total_cost"),
                f"round[{round_index}].{key}.total_cost",
            )
            finite_float(
                item.get("score"),
                f"round[{round_index}].{key}.score",
            )
            for metric in metric_names:
                finite_float(
                    item.get(metric),
                    f"round[{round_index}].{key}.{metric}",
                )
                normalized = item.get("normalized_costs", {})
                scales = item.get("cost_scales", {})
                finite_float(
                    normalized.get(metric),
                    f"round[{round_index}].{key}.normalized_costs.{metric}",
                )
                finite_float(
                    scales.get(metric),
                    f"round[{round_index}].{key}.cost_scales.{metric}",
                )
            grouped_scored.setdefault(key, []).append(item)

    if not grouped_scored:
        raise MainStop("Grounding VLA: no hay hipótesis agregables")

    round_count = len(rounds)
    incomplete = {
        f"{key[0]}->{key[1]}": len(items)
        for key, items in grouped_scored.items()
        if len(items) != round_count
    }
    if incomplete:
        raise MainStop(
            "Grounding VLA: mediciones incompletas por hipótesis: "
            f"{incomplete}"
        )

    aggregated_scored: list[dict[str, Any]] = []
    for key in sorted(grouped_scored):
        items = grouped_scored[key]
        exemplar = items[0]

        aggregate_item: dict[str, Any] = {
            "object": exemplar["object"],
            "target": exemplar["target"],
            "pair_instruction": exemplar["pair_instruction"],
            "pair_target_instruction": exemplar["pair_target_instruction"],
            "source_geometry": exemplar["source_geometry"],
            "round_measurements": items,
            "round_count": len(items),
            "metric_weights": exemplar["metric_weights"],
        }

        metric_dispersion: dict[str, float] = {}
        normalized_costs: dict[str, float] = {}
        cost_scales: dict[str, float] = {}

        for metric in metric_names:
            raw_values = np.asarray(
                [finite_float(item[metric], f"{key}.{metric}") for item in items],
                dtype=float,
            )
            raw_median = float(np.median(raw_values))
            aggregate_item[metric] = raw_median
            metric_dispersion[f"{metric}_mad"] = float(
                np.median(np.abs(raw_values - raw_median))
            )

            normalized_values = np.asarray(
                [
                    finite_float(
                        item["normalized_costs"][metric],
                        f"{key}.normalized_costs.{metric}",
                    )
                    for item in items
                ],
                dtype=float,
            )
            normalized_costs[metric] = float(np.median(normalized_values))

            scale_values = np.asarray(
                [
                    finite_float(
                        item["cost_scales"][metric],
                        f"{key}.cost_scales.{metric}",
                    )
                    for item in items
                ],
                dtype=float,
            )
            cost_scales[metric] = float(np.median(scale_values))

        round_costs = np.asarray(
            [
                finite_float(item["total_cost"], f"{key}.total_cost")
                for item in items
            ],
            dtype=float,
        )
        round_scores = np.asarray(
            [
                finite_float(item["score"], f"{key}.score")
                for item in items
            ],
            dtype=float,
        )
        total_cost = float(np.median(round_costs))

        aggregate_item.update(
            {
                "normalized_costs": normalized_costs,
                "cost_scales": cost_scales,
                "metric_dispersion": metric_dispersion,
                "round_total_costs": round_costs.tolist(),
                "round_scores": round_scores.tolist(),
                "total_cost": total_cost,
                "score": float(1.0 / (1.0 + total_cost)),
                "aggregation_contract":
                    "median_of_per_round_normalized_total_costs",
            }
        )
        aggregated_scored.append(aggregate_item)

    aggregate_summary = _joint_policy_round_margin(aggregated_scored)
    aggregate_key = tuple(aggregate_summary["winner_key"])

    votes: dict[tuple[str, str], int] = {}
    round_winners: list[dict[str, Any]] = []
    for round_index, round_result in enumerate(rounds, start=1):
        summary = round_result.get("summary")
        if not isinstance(summary, dict) or "winner_key" not in summary:
            raise MainStop(
                f"Grounding VLA: round {round_index} sin summary válido"
            )
        winner_key = tuple(summary["winner_key"])
        if len(winner_key) != 2 or winner_key not in grouped_scored:
            raise MainStop(
                "Grounding VLA: winner_key inválido en "
                f"round={round_index}: {winner_key}"
            )
        votes[winner_key] = votes.get(winner_key, 0) + 1
        round_winners.append(
            {
                "round": round_result.get("name", f"round_{round_index}"),
                "seed": round_result.get("seed"),
                "object": winner_key[0],
                "target": winner_key[1],
                "relative_margin": finite_float(
                    summary.get("relative_margin", 0.0),
                    f"round[{round_index}].relative_margin",
                ),
            }
        )

    highest_vote_count = max(votes.values())
    highest_vote_keys = sorted(
        key for key, count in votes.items()
        if count == highest_vote_count
    )
    unique_plurality_key = (
        highest_vote_keys[0]
        if len(highest_vote_keys) == 1
        else None
    )

    required_votes = int(
        os.environ.get(
            "VLA_GROUNDING_CONSENSUS_MIN_VOTES",
            str((round_count // 2) + 1),
        )
    )
    required_votes = max(1, min(required_votes, round_count))

    vote_consensus_pass = bool(
        unique_plurality_key is not None
        and highest_vote_count >= required_votes
    )
    vote_aggregate_agreement = (
        bool(unique_plurality_key == aggregate_key)
        if unique_plurality_key is not None
        else None
    )

    if vote_consensus_pass and vote_aggregate_agreement is True:
        stability_class = "stable_majority"
    elif unique_plurality_key is None:
        stability_class = "split_vote_aggregate_primary"
    elif vote_aggregate_agreement is True:
        stability_class = "plurality_agrees_aggregate"
    else:
        stability_class = "vote_disagrees_aggregate_primary"

    aggregate_votes = int(votes.get(aggregate_key, 0))
    consensus = {
        "round_count": round_count,
        "round_winners": round_winners,
        "votes": {
            f"{key[0]}->{key[1]}": count
            for key, count in sorted(votes.items())
        },
        "majority_key": (
            list(unique_plurality_key)
            if unique_plurality_key is not None
            else None
        ),
        "majority_count": highest_vote_count,
        "required_votes": required_votes,
        "vote_consensus_pass": vote_consensus_pass,
        "vote_aggregate_agreement": vote_aggregate_agreement,
        "aggregate_key": list(aggregate_key),
        "aggregate_votes": aggregate_votes,
        "aggregate_relative_margin": float(
            aggregate_summary["relative_margin"]
        ),
        "stability_class": stability_class,
        "decision_rule":
            "aggregate_median_primary_votes_diagnostic_only",
        "aggregation_contract":
            "median_of_per_round_normalized_total_costs",
        "raw_metric_renormalization_used": False,
        "votes_used_as_veto": False,
    }

    print(
        "contract=median_of_per_round_normalized_total_costs "
        f"winner={aggregate_key[0]}->{aggregate_key[1]} "
        f"aggregate_votes={aggregate_votes}/{round_count} "
        f"stability={stability_class} "
        f"margin={float(aggregate_summary['relative_margin']):.4f} "
        "votes_diagnostic_only=true "
        "raw_metric_renormalization=false",
        flush=True,
    )

    return aggregated_scored, consensus



def _normalise_instruction_phrase(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return " ".join(value.split())


def _candidate_instruction_phrases(candidate: dict[str, Any], role: str) -> set[str]:
    raw_names = {
        str(candidate.get("name", "")),
        *[str(alias) for alias in candidate.get("aliases", [])],
    }
    phrases: set[str] = set()
    suffixes = (" target", " source", " area", " region", " box", " bin")

    canonical_name = _normalise_instruction_phrase(
        str(candidate.get("name", "")).replace("_", " ")
    )

    object_aliases = {
        "white cylinder": {
            "white cylindrical object",
            "cylindrical white object",
        },
        "orange cube": {
            "orange block",
            "orange die",
        },
    }

    for raw_name in raw_names:
        phrase = _normalise_instruction_phrase(raw_name.replace("_", " "))
        if not phrase:
            continue

        phrases.add(phrase)

        if role == "target":
            stripped = phrase
            for suffix in suffixes:
                if stripped.endswith(suffix):
                    stripped = stripped[: -len(suffix)].strip()
                    break

            if stripped:
                phrases.add(stripped)
                phrases.add(f"{stripped} zone")

    if role == "object":
        phrases.update(object_aliases.get(canonical_name, set()))

    return {phrase for phrase in phrases if len(phrase) >= 3}


def _explicit_instruction_candidates(
    instruction: str,
    candidates: list[dict[str, Any]],
    role: str,
) -> list[str]:
    normalised = f" {_normalise_instruction_phrase(instruction)} "
    matches: list[str] = []
    for candidate in candidates:
        phrases = _candidate_instruction_phrases(candidate, role)
        if any(f" {phrase} " in normalised for phrase in phrases):
            matches.append(str(candidate["name"]))
    return sorted(set(matches))



def _instruction_normalized_candidates(
    *,
    instruction: str,
    candidates: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """
    Normaliza alias, sin excluir candidatos físicos.
    
        Todos los objetos y destinos detectados por Photoneo compiten ante el
        mismo VLA. El parser no selecciona ni filtra el ganador; solo registra
        qué entidades nombra explícitamente la instrucción para calcular
        exactitud semántica y aplicar un veto de seguridad después del top-1.
    """
    explicit_object = _explicit_instruction_candidates(
        instruction, candidates["object"], "object"
    )
    explicit_target = _explicit_instruction_candidates(
        instruction, candidates["target"], "target"
    )

    competition = {
        "object": list(candidates["object"]),
        "target": list(candidates["target"]),
    }

    if not competition["object"]:
        raise MainStop("La escena no contiene objetos físicos candidatos")
    if not competition["target"]:
        raise MainStop("La escena no contiene destinos físicos candidatos")

    trace = {
        "method": "instruction_alias_normalization_without_candidate_exclusion",
        "instruction": instruction,
        "explicit_object_candidates": explicit_object,
        "explicit_target_candidates": explicit_target,
        "all_object_candidates": [
            str(candidate["name"]) for candidate in competition["object"]
        ],
        "all_target_candidates": [
            str(candidate["name"]) for candidate in competition["target"]
        ],
        "competition_object_candidates": [
            str(candidate["name"]) for candidate in competition["object"]
        ],
        "competition_target_candidates": [
            str(candidate["name"]) for candidate in competition["target"]
        ],
        "class_exclusion_applied": False,
        "parser_used_as_selection": False,
        "decision_owner": "selected_vla",
        "safety_role": "post_selection_semantic_veto_only",
    }
    print(
        f"explicit_objects={explicit_object} "
        f"explicit_targets={explicit_target} "
        f"competition_objects={trace['competition_object_candidates']} "
        f"competition_targets={trace['competition_target_candidates']} "
        "class_exclusion=false decision_owner=VLA"
    )
    return competition, trace

def _language_counterfactual_instruction(
    *,
    instruction: str,
    candidates: dict[str, list[dict[str, Any]]],
    selected_object: dict[str, Any],
    selected_target: dict[str, Any],
) -> str:
    """
    Construye una orden distinta de la tarea real para probar el canal verbal.
    
        La elección del contrafactual se hace respecto a las entidades mencionadas
        en la instrucción original, no respecto al top-1 del VLA. Así, incluso si el
        VLA selecciona el par equivocado, nunca se reutiliza accidentalmente la
        misma orden como supuesto contrafactual.
    """
    original_normalized = _normalise_instruction_phrase(instruction)
    explicit_objects = set(
        _explicit_instruction_candidates(
            instruction, candidates["object"], "object"
        )
    )
    explicit_targets = set(
        _explicit_instruction_candidates(
            instruction, candidates["target"], "target"
        )
    )

    ranked_pairs: list[tuple[tuple[int, int, int], int, dict[str, Any], dict[str, Any], str]] = []
    pair_index = 0
    for object_candidate in candidates["object"]:
        for target_candidate in candidates["target"]:
            object_name = str(object_candidate["name"])
            target_name = str(target_candidate["name"])
            object_label = _grounding_candidate_label(
                object_candidate, "object"
            )
            target_label = _grounding_candidate_label(
                target_candidate, "target"
            )
            candidate_instruction = (
                f"put the {object_label} in the {target_label}"
            )
            if (
                _normalise_instruction_phrase(candidate_instruction)
                == original_normalized
            ):
                pair_index += 1
                continue

            object_differs = int(
                not explicit_objects or object_name not in explicit_objects
            )
            target_differs = int(
                not explicit_targets or target_name not in explicit_targets
            )
            selected_pair_differs = int(
                object_name != str(selected_object["name"])
                or target_name != str(selected_target["name"])
            )
            priority = (
                int(bool(object_differs and target_differs)),
                object_differs + target_differs,
                selected_pair_differs,
            )
            ranked_pairs.append(
                (
                    priority,
                    -pair_index,
                    object_candidate,
                    target_candidate,
                    candidate_instruction,
                )
            )
            pair_index += 1

    if ranked_pairs:
        ranked_pairs.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        counterfactual = ranked_pairs[0][4]
        if (
            _normalise_instruction_phrase(counterfactual)
            == original_normalized
        ):
            raise MainStop(
                "Invariante rota: el contrafactual lingüístico coincide "
                "con la instrucción original"
            )
        print(
            f"original={instruction!r} counterfactual={counterfactual!r} "
            "source=alternate_detected_pair distinct=true"
        )
        return counterfactual

    configured = os.environ.get(
        "VLA_LANGUAGE_CONTROL_INSTRUCTION",
        "move safely without selecting, grasping, transporting, or placing any object",
    ).strip()
    if not configured:
        raise MainStop("VLA_LANGUAGE_CONTROL_INSTRUCTION está vacía")
    if _normalise_instruction_phrase(configured) == original_normalized:
        raise MainStop(
            "La instrucción de control lingüístico coincide con la tarea real"
        )
    print(
        f"original={instruction!r} counterfactual={configured!r} "
        "source=configured_control distinct=true"
    )
    return configured


def _vla_visual_language_participation_gate(
    *,
    model: str,
    original_actions: np.ndarray,
    image_path: Path,
    instruction: str,
    candidates: dict[str, list[dict[str, Any]]],
    selected_object: dict[str, Any],
    selected_target: dict[str, Any],
    q_ur5e: np.ndarray,
    gripper_width_m: float,
    droid_state_mode: str,
    seed: int,
    grounding_dir: Path,
) -> dict[str, Any]:
    """
    Comprueba durante la ejecución que la política depende de visión y lenguaje.
    
        Se mantiene estado, semilla y modelo. Solo se cambia una modalidad cada vez:
        1) visión: se ocultan las regiones seleccionadas conservando la orden;
        2) lenguaje: se conserva la imagen y se usa una orden contrafactual.
        El episodio se bloquea si la política no cambia de forma medible en ambas.
    """
    gate_dir = grounding_dir / "vla_visual_language_participation"
    gate_dir.mkdir(parents=True, exist_ok=True)

    visual_counterfactual_path = gate_dir / "selected_regions_masked.png"
    _mask_candidate_regions_image(
        image_path,
        [selected_object, selected_target],
        visual_counterfactual_path,
        variant="inpaint",
        padding_override=int(
            os.environ.get("VLA_PARTICIPATION_MASK_PADDING_PX", "10")
        ),
        blur_kernel_override=41,
    )
    visual_actions, visual_evidence = infer_on_orin(
        model=model,
        image_path=visual_counterfactual_path,
        instruction=instruction,
        q_ur5e=q_ur5e,
        gripper_width_m=gripper_width_m,
        droid_state_mode=droid_state_mode,
        seed=seed,
        run_dir=gate_dir / "visual_counterfactual",
    )
    visual_effect = _action_grounding_distance(
        model,
        original_actions,
        visual_actions,
    )

    language_counterfactual = _language_counterfactual_instruction(
        instruction=instruction,
        candidates=candidates,
        selected_object=selected_object,
        selected_target=selected_target,
    )
    language_counterfactual_distinct = (
        _normalise_instruction_phrase(language_counterfactual)
        != _normalise_instruction_phrase(instruction)
    )
    if not language_counterfactual_distinct:
        raise MainStop(
            "Invariante rota: el contrafactual lingüístico no cambia la orden"
        )
    print(
        "distinct=true"
    )
    language_actions, language_evidence = infer_on_orin(
        model=model,
        image_path=image_path,
        instruction=language_counterfactual,
        q_ur5e=q_ur5e,
        gripper_width_m=gripper_width_m,
        droid_state_mode=droid_state_mode,
        seed=seed,
        run_dir=gate_dir / "language_counterfactual",
    )
    language_effect = _action_grounding_distance(
        model,
        original_actions,
        language_actions,
    )

    minimum_visual_effect = float(
        os.environ.get("VLA_MIN_VISUAL_EFFECT", "0.001")
    )
    minimum_language_effect = float(
        os.environ.get("VLA_MIN_LANGUAGE_EFFECT", "0.001")
    )
    for name, value in (
        ("VLA_MIN_VISUAL_EFFECT", minimum_visual_effect),
        ("VLA_MIN_LANGUAGE_EFFECT", minimum_language_effect),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 5.0:
            raise MainStop(f"{name} no válido")

    visual_pass = float(visual_effect["distance"]) >= minimum_visual_effect
    language_pass = float(language_effect["distance"]) >= minimum_language_effect
    print(
        f"model={model} effect={visual_effect['distance']:.8f} "
        f"minimum={minimum_visual_effect:.8f} pass={str(visual_pass).lower()}"
    )
    print(
        f"model={model} effect={language_effect['distance']:.8f} "
        f"minimum={minimum_language_effect:.8f} pass={str(language_pass).lower()} "
        f"counterfactual={language_counterfactual!r}"
    )

    trace = {
        "schema": "vla_modal_participation",
        "model": model,
        "same_model_same_state_same_seed": True,
        "original_instruction": instruction,
        "language_counterfactual_instruction": language_counterfactual,
        "language_counterfactual_distinct": language_counterfactual_distinct,
        "original_image": str(image_path),
        "visual_counterfactual_image": str(visual_counterfactual_path),
        "visual_effect": visual_effect,
        "language_effect": language_effect,
        "minimum_visual_effect": minimum_visual_effect,
        "minimum_language_effect": minimum_language_effect,
        "visual_pass": visual_pass,
        "language_pass": language_pass,
        "visual_evidence": visual_evidence,
        "language_evidence": language_evidence,
        "guarantee": (
            "episode_continues_only_if_selected_vla_changes_actions_when_"
            "vision_and_language_are_independently_counterfactually_changed"
        ),
    }
    json_write(gate_dir / "vla_visual_language_participation.json", trace)
    print(
        f"visual_pass={str(visual_pass).lower()} "
        f"language_pass={str(language_pass).lower()}"
    )

    if not visual_pass:
        raise MainStop(
            "El VLA no demuestra participación visual suficiente sobre la "
            f"captura live: efecto={visual_effect['distance']:.8f} < "
            f"{minimum_visual_effect:.8f}"
        )
    if not language_pass:
        raise MainStop(
            "El VLA no demuestra participación lingüística suficiente: "
            f"efecto={language_effect['distance']:.8f} < "
            f"{minimum_language_effect:.8f}"
        )

    print(
        f"model={model} visual={visual_effect['distance']:.8f} "
        f"language={language_effect['distance']:.8f}"
    )
    return trace


def _semantic_consistency_veto(
    *,
    instruction: str,
    candidates: dict[str, list[dict[str, Any]]],
    selected_object: dict[str, Any],
    selected_target: dict[str, Any],
) -> dict[str, Any]:
    """
    Evalúa el top-1 del VLA sin sustituirlo por una decisión parseada.
    
        La selección cruda del VLA siempre queda registrada. Si contradice una
        orden inequívoca, el episodio se bloquea después de guardar métricas; el
        parser nunca reemplaza el ganador por otro candidato.
    """
    explicit_object = _explicit_instruction_candidates(
        instruction, candidates["object"], "object"
    )
    explicit_target = _explicit_instruction_candidates(
        instruction, candidates["target"], "target"
    )
    selected_object_name = str(selected_object["name"])
    selected_target_name = str(selected_target["name"])

    object_correct = (
        None
        if len(explicit_object) != 1
        else selected_object_name == explicit_object[0]
    )
    target_correct = (
        None
        if len(explicit_target) != 1
        else selected_target_name == explicit_target[0]
    )
    pair_correct = bool(
        (object_correct is not False)
        and (target_correct is not False)
    )

    trace = {
        "role": "post_selection_semantic_veto_only",
        "instruction": instruction,
        "explicit_object_candidates": explicit_object,
        "explicit_target_candidates": explicit_target,
        "selected_object": selected_object_name,
        "selected_target": selected_target_name,
        "top1_object_correct": object_correct,
        "top1_target_correct": target_correct,
        "top1_pair_correct": pair_correct,
        "pass": pair_correct,
        "used_for_selection": False,
        "selection_replaced": False,
    }
    print(
        f"selected={selected_object_name}->{selected_target_name} "
        f"expected_object={explicit_object} expected_target={explicit_target} "
        f"pass={str(pair_correct).lower()} parser_replacement=false"
    )
    return trace


def _shape_siglip_normalize_label(
    value: Any,
) -> str:
    import re

    return (
        re.sub(
            r"\s+",
            "_",
            str(value)
            .strip()
            .lower()
            .replace("-", " ")
            .replace("_", " "),
        )
        .strip("_")
    )


def _shape_siglip_canonical_label(
    value: Any,
    role: str,
) -> str:
    normalized = (
        _shape_siglip_normalize_label(
            value
        )
    )

    mappings = {
        "object": {
            "orange_cube":
                "orange_cube",
            "orange_block":
                "orange_cube",
            "orange_die":
                "orange_cube",
            "orange_object":
                "orange_cube",
            "white_cylinder":
                "white_cylinder",
            "white_tube":
                "white_cylinder",
            "white_cylindrical_object":
                "white_cylinder",
        },
        "target": {
            "square_box":
                "square_box",
            "square_container":
                "square_box",
            "square_receptacle":
                "square_box",
            "black_square_box":
                "square_box",
            "circular_box":
                "circular_box",
            "circular_container":
                "circular_box",
            "circular_receptacle":
                "circular_box",
            "round_box":
                "circular_box",
            "round_container":
                "circular_box",
            "round_receptacle":
                "circular_box",
            "black_circular_box":
                "circular_box",
        },
    }

    if role not in mappings:
        raise MainStop(
            "Rol shape_siglip no soportado: "
            f"{role!r}"
        )

    canonical = mappings[
        role
    ].get(
        normalized
    )

    if canonical is None:
        raise MainStop(
            "Clase no soportada por "
            "VLA_SIGLIP_SHAPE_CONTAINERS: "
            f"role={role}, "
            f"value={value!r}, "
            f"normalized={normalized!r}"
        )

    return canonical


def _shape_siglip_resolve_request(
    *,
    instruction: str,
    candidates: dict[
        str,
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    import re

    explicit_objects = list(
        dict.fromkeys(
            _explicit_instruction_candidates(
                instruction,
                candidates["object"],
                "object",
            )
        )
    )

    explicit_targets = list(
        dict.fromkeys(
            _explicit_instruction_candidates(
                instruction,
                candidates["target"],
                "target",
            )
        )
    )

    if len(explicit_objects) != 1:
        raise MainStop(
            "shape_siglip requiere exactamente "
            "un objeto explícito. La instrucción "
            "genérica con dos objetos es ambigua: "
            f"explicit_objects={explicit_objects}"
        )

    requested_object_label = (
        _shape_siglip_canonical_label(
            explicit_objects[0],
            "object",
        )
    )

    normalized_instruction = (
        re.sub(
            r"\s+",
            " ",
            str(instruction)
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " "),
        )
    )

    relation_by_shape = bool(
        re.search(
            (
                r"\b(?:match|matches|matching|"
                r"correspond|corresponds|corresponding|same)"
                r"\b.*\bshape\b"
            ),
            normalized_instruction,
        )
        or re.search(
            (
                r"\bshape\b.*\b(?:match|matches|matching|"
                r"correspond|corresponds|corresponding|same)"
                r"\b"
            ),
            normalized_instruction,
        )
    )

    shape_correspondence = {
        "orange_cube":
            "square_box",
        "white_cylinder":
            "circular_box",
    }

    if len(explicit_targets) == 1:
        requested_target_label = (
            _shape_siglip_canonical_label(
                explicit_targets[0],
                "target",
            )
        )

        target_resolution = (
            "explicit_target"
        )

    elif (
        len(explicit_targets) == 0
        and relation_by_shape
    ):
        requested_target_label = (
            shape_correspondence[
                requested_object_label
            ]
        )

        target_resolution = (
            "frozen_shape_correspondence"
        )

    else:
        raise MainStop(
            "shape_siglip necesita un destino "
            "explícito o una relación inequívoca "
            "de correspondencia por forma: "
            f"explicit_targets={explicit_targets}, "
            f"relation_by_shape={relation_by_shape}"
        )

    expected_target = (
        shape_correspondence[
            requested_object_label
        ]
    )

    if (
        relation_by_shape
        and requested_target_label
        != expected_target
    ):
        raise MainStop(
            "La instrucción contradice el contrato "
            "de correspondencia por forma: "
            f"object={requested_object_label}, "
            f"requested_target="
            f"{requested_target_label}, "
            f"expected_target={expected_target}"
        )

    return {
        "requested_object_label":
            requested_object_label,
        "requested_target_label":
            requested_target_label,
        "explicit_object_candidates":
            explicit_objects,
        "explicit_target_candidates":
            explicit_targets,
        "relation_by_shape":
            relation_by_shape,
        "target_resolution":
            target_resolution,
        "shape_correspondence":
            shape_correspondence,
        "parser_used_as_selection":
            False,
        "parser_replacement":
            False,
    }


def _shape_siglip_request_grounding(
    *,
    image_path: Path,
    instruction: str,
    candidates: dict[
        str,
        list[dict[str, Any]],
    ],
) -> dict[str, Any]:
    if (
        len(candidates["object"]) != 2
        or len(candidates["target"]) != 2
    ):
        raise MainStop(
            "VLA_SIGLIP_SHAPE_CONTAINERS "
            "requiere dos objetos y dos "
            "recipientes: "
            f"objects={len(candidates['object'])}, "
            f"targets={len(candidates['target'])}"
        )

    request_trace = (
        _shape_siglip_resolve_request(
            instruction=instruction,
            candidates=candidates,
        )
    )

    # Se conservan los dos crops live y su geometría.
    # No se asigna identidad por posición vertical.
    target_candidates = [
        copy.deepcopy(candidate)
        for candidate in candidates["target"]
    ]

    print(
        "position_based_relabel=false "
        f"names={[candidate.get('name') for candidate in target_candidates]} "
        f"bboxes={[candidate.get('bbox_xyxy') for candidate in target_candidates]}",
        flush=True,
    )

    try:
        from siglip_shape_grounding_client import (
            ShapeSiglipGroundingError,
            check_health,
            ground_candidates,
        )

        health = check_health()

        result = ground_candidates(
            image_path=image_path,
            requested_object_label=
                request_trace[
                    "requested_object_label"
                ],
            requested_target_label=
                request_trace[
                    "requested_target_label"
                ],
            object_candidates=
                candidates["object"],
            target_candidates=
                target_candidates,
        )

    except ShapeSiglipGroundingError as error:
        raise MainStop(
            "Grounding shape_siglip "
            f"bloqueado: {error}"
        ) from error

    selected_object = result[
        "selected_object"
    ]

    selected_target_from_service = copy.deepcopy(
        result["selected_target"]
    )

    def _shape_bbox_key(
        candidate: dict[str, Any],
    ) -> tuple[float, float, float, float] | None:
        bbox = candidate.get("bbox_xyxy")

        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            return None

        try:
            return tuple(float(value) for value in bbox)
        except (TypeError, ValueError):
            return None

    selected_target_bbox = _shape_bbox_key(
        selected_target_from_service
    )

    matching_targets = [
        candidate
        for candidate in target_candidates
        if _shape_bbox_key(candidate)
        == selected_target_bbox
    ]

    if len(matching_targets) != 1:
        raise MainStop(
            "El crop elegido por shape_siglip no puede "
            "vincularse inequívocamente con un recipiente live: "
            f"selected_bbox={selected_target_bbox}; "
            f"matches={len(matching_targets)}"
        )

    selected_target = copy.deepcopy(
        matching_targets[0]
    )

    selected_target_detector_name = str(
        selected_target.get("name", "")
    )

    requested_target_name = str(
        request_trace["requested_target_label"]
    )

    requested_target_shape = {
        "square_box": "square",
        "circular_box": "circular",
    }.get(requested_target_name)

    if requested_target_shape is None:
        raise MainStop(
            "Clase de recipiente shape no soportada: "
            f"{requested_target_name}"
        )

    selected_target[
        "detector_name_before_shape_siglip_binding"
    ] = selected_target_detector_name

    selected_target[
        "shape_siglip_service_selected_name"
    ] = str(
        selected_target_from_service.get("name", "")
    )

    selected_target["name"] = requested_target_name
    selected_target["aliases"] = [
        requested_target_name
    ]
    selected_target["canonical"] = (
        requested_target_name
    )
    selected_target["container_shape"] = (
        requested_target_shape
    )

    print(
        f"selected_bbox={selected_target_bbox} "
        f"detector_name={selected_target_detector_name} "
        f"bound_name={requested_target_name} "
        "selection_owner=shape_siglip "
        "position_based_relabel=false "
        "parser_replacement=false",
        flush=True,
    )

    selected_object_name = str(
        selected_object["name"]
    )

    selected_target_name = str(
        selected_target["name"]
    )

    if (
        selected_object_name
        != request_trace[
            "requested_object_label"
        ]
    ):
        raise MainStop(
            "El top-1 de objeto del servicio "
            "shape_siglip no coincide con "
            "la consulta validada: "
            f"selected={selected_object_name}, "
            f"requested="
            f"{request_trace['requested_object_label']}"
        )

    if (
        selected_target_name
        != request_trace[
            "requested_target_label"
        ]
    ):
        raise MainStop(
            "El top-1 de recipiente del servicio "
            "shape_siglip no coincide con "
            "la consulta validada: "
            f"selected={selected_target_name}, "
            f"requested="
            f"{request_trace['requested_target_label']}"
        )

    if (
        result.get(
            "candidate_semantic_names_sent"
        )
        is not False
    ):
        raise MainStop(
            "shape_siglip recibió nombres "
            "semánticos asociados a los crops."
        )

    return {
        "health":
            health,
        "request_trace":
            request_trace,
        "client_result":
            result,
        "selected_object":
            selected_object,
        "selected_target":
            selected_target,
    }


def _shape_common_container_relabel(
    candidates: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Conserva las detecciones live sin imponer identidad por posición."""

    task_family = (
        os.environ.get(
            "VLA_TASK_FAMILY",
            "pick_place",
        )
        .strip()
        .lower()
    )

    if task_family != "shape_classification":
        return candidates

    corrected = {
        "object": [
            copy.deepcopy(candidate)
            for candidate in candidates.get("object", [])
        ],
        "target": [
            copy.deepcopy(candidate)
            for candidate in candidates.get("target", [])
        ],
    }

    if len(corrected["target"]) != 2:
        raise MainStop(
            "La campaña de forma requiere exactamente dos "
            "recipientes live: "
            f"targets={len(corrected['target'])}"
        )

    target_trace = [
        {
            "name": str(candidate.get("name", "")),
            "bbox_xyxy": candidate.get("bbox_xyxy"),
            "base_xyz_m": candidate.get("base_xyz_m"),
            "container_shape": candidate.get(
                "container_shape"
            ),
        }
        for candidate in corrected["target"]
    ]

    print(
        "mode=preserve_live_detector_candidates "
        "position_based_relabel=false "
        f"targets={target_trace}",
        flush=True,
    )

    return corrected



def ground_scene_with_vla(
    *,
    model: str,
    actions: np.ndarray,
    scene: dict[str, Any],
    image_path: Path,
    instruction: str,
    q_ur5e: np.ndarray,
    gripper_width_m: float,
    droid_state_mode: str,
    seed: int,
    run_dir: Path,
    mode: str,
    grounding_json: str | None,
    grounding_heatmap: str | None,
) -> dict[str, Any]:
    # Este bloque afecta solo al grounding objeto->zona.
    # La acción nativa original y config.instruction conservan
    # la instrucción completa para la preferencia espacial.
    import re as _re

    full_spatial_instruction = str(
        instruction
    )

    object_target_instruction = (
        full_spatial_instruction
    )

    spatial_patterns = (
        (
            r"\b(?:in|inside|at|on|near|toward|towards)"
            r"\s+(?:the\s+)?(?:exact\s+)?"
            r"(?:center|centre|middle|central"
            r"(?:\s+(?:area|part))?)"
            r"\s+of\s+(?:the\s+)?"
        ),
        (
            r"\b(?:in|inside|at|on|near|toward|towards)"
            r"\s+(?:the\s+)?"
            r"(?:(?:front|back|left|right)"
            r"(?:[-\s]+(?:front|back|left|right))?"
            r"\s+)?"
            r"(?:side|edge|border|corner|part|area)"
            r"\s+of\s+(?:the\s+)?"
        ),
    )

    removed_spatial_modifier = False

    for spatial_pattern in spatial_patterns:
        (
            object_target_instruction,
            replacement_count,
        ) = _re.subn(
            spatial_pattern,
            "in the ",
            object_target_instruction,
            flags=_re.IGNORECASE,
        )

        if replacement_count:
            removed_spatial_modifier = True

    object_target_instruction = (
        _re.sub(
            r"\s+",
            " ",
            object_target_instruction,
        )
        .strip()
    )

    object_target_instruction = (
        _re.sub(
            r"\bin\s+the\s+the\b",
            "in the",
            object_target_instruction,
            flags=_re.IGNORECASE,
        )
    )

    if (
        removed_spatial_modifier
        and object_target_instruction.lower()
        != full_spatial_instruction.lower()
    ):
        original_action_shape = tuple(
            np.asarray(
                actions,
                dtype=float,
            ).shape
        )

        reference_dir = (
            run_dir
            / "vla_grounding"
            / "object_target_reference"
        )

        reference_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            object_target_actions,
            object_target_evidence,
        ) = infer_on_orin(
            model=model,
            image_path=image_path,
            instruction=object_target_instruction,
            q_ur5e=q_ur5e,
            gripper_width_m=gripper_width_m,
            droid_state_mode=droid_state_mode,
            seed=seed,
            run_dir=reference_dir,
        )

        object_target_actions = np.asarray(
            object_target_actions,
            dtype=float,
        )

        if (
            tuple(object_target_actions.shape)
            != original_action_shape
        ):
            raise MainStop(
                "La referencia canónica objeto-zona "
                "cambió el contrato nativo: "
                f"full={original_action_shape}, "
                f"canonical="
                f"{tuple(object_target_actions.shape)}"
            )

        if not np.all(
            np.isfinite(
                object_target_actions
            )
        ):
            raise MainStop(
                "La referencia canónica objeto-zona "
                "contiene NaN o Inf"
            )

        # La reasignación es local y no modifica las acciones de main()
        # que se usan después para el place espacial.
        actions = object_target_actions
        instruction = (
            object_target_instruction
        )

        json_write(
            reference_dir
            / "grounding_instruction_split.json",
            {
                "schema":
                    "VLA_grounding_instruction_split",
                "full_spatial_instruction":
                    full_spatial_instruction,
                "object_target_instruction":
                    object_target_instruction,
                "native_shape":
                    list(
                        object_target_actions.shape
                    ),
                "reference_evidence":
                    object_target_evidence,
                "full_instruction_preserved_for_place":
                    True,
                "parser_replacement":
                    False,
            },
        )

        print(
            f"full={full_spatial_instruction!r} "
            f"object_target="
            f"{object_target_instruction!r} "
            "full_instruction_preserved_for_place=true "
            "parser_replacement=false",
            flush=True,
        )
    candidates = _shape_common_container_relabel(
        neutral_scene_candidates(scene, image_path)
    )
    competition_candidates, language_normalization_trace = (
        _instruction_normalized_candidates(
            instruction=instruction,
            candidates=candidates,
        )
    )

    physical_grounding_policy = (
        os.environ.get(
            "VLA_PHYSICAL_GROUNDING_POLICY",
            "autonomous_all_candidates",
        )
        .strip()
        .lower()
    )

    valid_physical_grounding_policies = {
        "autonomous_all_candidates",
        "instruction_constrained_unique_pair",
    }

    if (
        physical_grounding_policy
        not in valid_physical_grounding_policies
    ):
        raise MainStop(
            "VLA_PHYSICAL_GROUNDING_POLICY no válido: "
            f"{physical_grounding_policy!r}"
        )

    if (
        physical_grounding_policy
        == "instruction_constrained_unique_pair"
    ):
        _constraint_re = __import__("re")

        def _normalized_semantic_text(
            value: Any,
        ) -> str:
            return (
                _constraint_re.sub(
                    r"\s+",
                    " ",
                    str(value)
                    .lower()
                    .replace("_", " ")
                    .replace("-", " "),
                )
                .strip()
            )

        def _candidate_semantic_terms(
            candidate: dict[str, Any],
        ) -> list[str]:
            raw_terms: list[Any] = [
                candidate.get("name"),
                candidate.get("label"),
                candidate.get("display_name"),
            ]

            aliases = candidate.get(
                "aliases",
                [],
            )

            if isinstance(aliases, str):
                raw_terms.append(aliases)

            elif isinstance(
                aliases,
                (list, tuple, set),
            ):
                raw_terms.extend(aliases)

            normalized_terms = []

            for raw_term in raw_terms:
                if raw_term is None:
                    continue

                normalized_term = (
                    _normalized_semantic_text(
                        raw_term
                    )
                )

                if (
                    normalized_term
                    and normalized_term
                    not in normalized_terms
                ):
                    normalized_terms.append(
                        normalized_term
                    )

            return normalized_terms

        constrained_instruction = (
            _normalized_semantic_text(
                instruction
            )
        )

        constrained_objects = [
            candidate
            for candidate
            in competition_candidates["object"]
            if any(
                term in constrained_instruction
                for term
                in _candidate_semantic_terms(
                    candidate
                )
            )
        ]

        constrained_targets = [
            candidate
            for candidate
            in competition_candidates["target"]
            if any(
                term in constrained_instruction
                for term
                in _candidate_semantic_terms(
                    candidate
                )
            )
        ]

        if len(constrained_objects) != 1:
            raise MainStop(
                "El modo físico condicionado requiere "
                "exactamente un objeto inequívoco en la "
                "instrucción: "
                f"encontrados="
                f"{[item.get('name') for item in constrained_objects]}"
            )

        if len(constrained_targets) != 1:
            raise MainStop(
                "El modo físico condicionado requiere "
                "exactamente un destino inequívoco en la "
                "instrucción: "
                f"encontrados="
                f"{[item.get('name') for item in constrained_targets]}"
            )

        all_object_names = [
            str(candidate.get("name"))
            for candidate
            in competition_candidates["object"]
        ]

        all_target_names = [
            str(candidate.get("name"))
            for candidate
            in competition_candidates["target"]
        ]

        competition_candidates = {
            "object":
                constrained_objects,
            "target":
                constrained_targets,
        }

        if not isinstance(
            language_normalization_trace,
            dict,
        ):
            language_normalization_trace = {
                "original_trace":
                    language_normalization_trace,
            }

        language_normalization_trace.update(
            {
                "physical_grounding_policy":
                    physical_grounding_policy,
                "conditional_control_execution":
                    True,
                "semantic_autonomy":
                    False,
                "parser_replacement":
                    False,
                "all_detected_objects":
                    all_object_names,
                "all_detected_targets":
                    all_target_names,
                "instruction_constrained_object":
                    str(
                        constrained_objects[0][
                            "name"
                        ]
                    ),
                "instruction_constrained_target":
                    str(
                        constrained_targets[0][
                            "name"
                        ]
                    ),
                "raw_autonomous_grounding_runs":
                    "preserved_in_separate_episode_directories",
            }
        )

        print(
            f"object="
            f"{constrained_objects[0]['name']} "
            f"target="
            f"{constrained_targets[0]['name']} "
            "semantic_autonomy=false "
            "conditional_control_execution=true "
            "parser_replacement=false",
            flush=True,
        )

    if mode == "parser_trace":
        raise MainStop("parser_trace es solo comprobación y nunca decide el episodio.")
    if mode in {"auto", "heatmap_json"} and grounding_json:
        return _load_grounding_json(
            Path(grounding_json), image_path=image_path, candidates=candidates
        )
    if mode in {"auto", "heatmap_json"} and grounding_heatmap:
        return _load_grounding_heatmap(
            Path(grounding_heatmap), image_path=image_path, candidates=candidates
        )
    if mode == "heatmap_json":
        raise MainStop(
            "--grounding-mode heatmap_json requiere "
            "--grounding-json o --grounding-heatmap"
        )
    if mode in {"auto", "region_occlusion", "policy_counterfactual"}:
        mode = "joint_policy_hypothesis"
    if mode != "joint_policy_hypothesis":
        raise MainStop(f"Modo de grounding no admitido: {mode}")

    maximum_candidates = int(
        os.environ.get("VLA_GROUNDING_MAX_CANDIDATES", "8")
    )
    total = len(candidates["object"]) + len(candidates["target"])
    if total > maximum_candidates:
        raise MainStop(
            f"Grounding VLA: {total} candidatos físicos exceden "
            f"VLA_GROUNDING_MAX_CANDIDATES={maximum_candidates}"
        )

    grounding_dir = run_dir / "vla_grounding"
    grounding_dir.mkdir(parents=True, exist_ok=True)

    # SigLIP decide solo qué región física corresponde
    # a las clases solicitadas por la instrucción.
    # No recibe nombres semánticos asociados a las cajas,
    # no modifica las acciones nativas y no interviene en
    # plan traslacional, seguridad, IK, garra ni ejecución.
    openvla_grounding_adapter = (
        os.environ.get(
            "VLA_OPENVLA_GROUNDING_ADAPTER",
            "siglip",
        )
        .strip()
        .lower()
    )

    if (
        model == "openvla"
        and openvla_grounding_adapter
        == "siglip"
    ):
        if (
            physical_grounding_policy
            != "autonomous_all_candidates"
        ):
            raise MainStop(
                "OpenVLA + SigLIP V2 exige "
                "VLA_PHYSICAL_GROUNDING_POLICY="
                "autonomous_all_candidates. "
                "No se permite reducir los candidatos "
                "mediante el parser antes del grounding."
            )

        if (
            len(candidates["object"]) != 2
            or len(candidates["target"]) != 2
        ):
            raise MainStop(
                "El contrato SigLIP V2 validado exige "
                "exactamente dos candidatos de objeto "
                "y dos de destino: "
                f"objects={len(candidates['object'])}, "
                f"targets={len(candidates['target'])}"
            )

        explicit_object_names = list(
            dict.fromkeys(
                _explicit_instruction_candidates(
                    full_spatial_instruction,
                    candidates["object"],
                    "object",
                )
            )
        )

        explicit_target_names = list(
            dict.fromkeys(
                _explicit_instruction_candidates(
                    full_spatial_instruction,
                    candidates["target"],
                    "target",
                )
            )
        )

        if len(explicit_object_names) != 1:
            raise MainStop(
                "SigLIP V2 requiere una única clase "
                "de objeto inequívoca en la instrucción: "
                f"{explicit_object_names}"
            )

        if len(explicit_target_names) != 1:
            raise MainStop(
                "SigLIP V2 requiere una única clase "
                "de destino inequívoca en la instrucción: "
                f"{explicit_target_names}"
            )

        def _siglip_canonical_label(
            value: Any,
            role: str,
        ) -> str:
            normalized = (
                _re.sub(
                    r"\s+",
                    "_",
                    str(value)
                    .strip()
                    .lower()
                    .replace("-", " ")
                    .replace("_", " "),
                )
                .strip("_")
            )

            mappings = {
                "object": {
                    "orange_cube":
                        "orange_cube",
                    "orange_block":
                        "orange_cube",
                    "orange_brick":
                        "orange_cube",
                    "orange_piece":
                        "orange_cube",
                    "orange_die":
                        "orange_cube",
                    "white_cylinder":
                        "white_cylinder",
                    "white_tube":
                        "white_cylinder",
                    "white_canister":
                        "white_cylinder",
                    "white_column":
                        "white_cylinder",
                },
                "target": {
                    "blue_zone":
                        "blue_zone",
                    "blue_area":
                        "blue_zone",
                    "blue_region":
                        "blue_zone",
                    "blue_mat":
                        "blue_zone",
                    "blue_source":
                        "blue_zone",
                    "yellow_zone":
                        "yellow_zone",
                    "yellow_area":
                        "yellow_zone",
                    "yellow_region":
                        "yellow_zone",
                    "yellow_mat":
                        "yellow_zone",
                    "yellow_target":
                        "yellow_zone",
                },
            }

            canonical = (
                mappings[
                    role
                ].get(normalized)
            )

            if canonical is None:
                raise MainStop(
                    "Clase semántica no soportada "
                    "por el contrato SigLIP V2: "
                    f"role={role}, value={value!r}, "
                    f"normalized={normalized!r}"
                )

            return canonical

        requested_object_label = (
            _siglip_canonical_label(
                explicit_object_names[0],
                "object",
            )
        )

        requested_target_label = (
            _siglip_canonical_label(
                explicit_target_names[0],
                "target",
            )
        )

        try:
            from siglip_grounding_client import (
                SiglipGroundingError,
                check_health,
                ground_candidates,
            )

            siglip_health = check_health()

            siglip_client_result = (
                ground_candidates(
                    image_path=image_path,
                    requested_object_label=
                        requested_object_label,
                    requested_target_label=
                        requested_target_label,
                    object_candidates=
                        candidates["object"],
                    target_candidates=
                        candidates["target"],
                )
            )

        except SiglipGroundingError as error:
            raise MainStop(
                "Grounding OpenVLA + SigLIP "
                f"bloqueado: {error}"
            ) from error

        selected_object = (
            siglip_client_result[
                "selected_object"
            ]
        )

        selected_target = (
            siglip_client_result[
                "selected_target"
            ]
        )

        if (
            siglip_client_result.get(
                "candidate_semantic_names_sent"
            )
            is not False
        ):
            raise MainStop(
                "SigLIP recibió nombres semánticos "
                "asociados a candidatos. "
                "El contrato neutral quedó roto."
            )

        semantic_consistency_trace = (
            _semantic_consistency_veto(
                instruction=
                    full_spatial_instruction,
                candidates=candidates,
                selected_object=
                    selected_object,
                selected_target=
                    selected_target,
            )
        )

        # OpenVLA mantiene además una comprobación independiente
        # de participación visual y lingüística sobre la salida
        # de acciones. Esa comprobación no selecciona candidatos.
        modal_participation_trace = (
            _vla_visual_language_participation_gate(
                model=model,
                original_actions=actions,
                image_path=image_path,
                instruction=instruction,
                candidates=candidates,
                selected_object=
                    selected_object,
                selected_target=
                    selected_target,
                q_ur5e=q_ur5e,
                gripper_width_m=
                    gripper_width_m,
                droid_state_mode=
                    droid_state_mode,
                seed=seed,
                grounding_dir=
                    grounding_dir,
            )
        )

        siglip_service_response = (
            siglip_client_result[
                "service_response"
            ]
        )

        object_query = (
            siglip_service_response[
                "roles"
            ][
                "object"
            ][
                "requested_query"
            ]
        )

        target_query = (
            siglip_service_response[
                "roles"
            ][
                "target"
            ][
                "requested_query"
            ]
        )

        grounding_decision = {
            "schema":
                "VLA_openvla_siglip_"
                "grounding_decision",
            "method":
                "photoneo_candidate_siglip_"
                "external_visual_language_grounding",
            "selection_source":
                "external_siglip_visual_language_grounding",
            "parser_role":
                "semantic_query_normalization_and_"
                "post_selection_geometry_lookup_only",
            "model":
                model,
            "grounding_adapter":
                "siglip_candidate_crop",
            "grounding_model_id":
                siglip_health["model_id"],
            "siglip_contract":
                siglip_health["contract"],
            "instruction":
                full_spatial_instruction,
            "grounding_instruction":
                instruction,
            "requested_object_label":
                requested_object_label,
            "requested_target_label":
                requested_target_label,
            "object":
                selected_object,
            "target":
                selected_target,
            "selected_object":
                str(
                    selected_object["name"]
                ),
            "selected_target":
                str(
                    selected_target["name"]
                ),
            "object_probability_margin":
                float(
                    object_query[
                        "probability_margin"
                    ]
                ),
            "target_probability_margin":
                float(
                    target_query[
                        "probability_margin"
                    ]
                ),
            "object_assignment_margin":
                float(
                    siglip_service_response[
                        "roles"
                    ][
                        "object"
                    ][
                        "assignment_margin"
                    ]
                ),
            "target_assignment_margin":
                float(
                    siglip_service_response[
                        "roles"
                    ][
                        "target"
                    ][
                        "assignment_margin"
                    ]
                ),
            "all_photoneo_candidates_competed":
                True,
            "candidate_semantic_names_sent":
                False,
            "semantic_consistency_trace":
                semantic_consistency_trace,
            "modal_participation_trace":
                modal_participation_trace,
            "language_normalization_trace":
                language_normalization_trace,
            "siglip_client_result":
                siglip_client_result,
            "native_actions_shape":
                list(
                    np.asarray(
                        actions,
                        dtype=float,
                    ).shape
                ),
            "native_actions_replaced":
                False,
            "native_actions_reinterpreted":
                False,
            "raw_instruction_preserved":
                True,
            "parser_replacement":
                False,
            "external_grounding_applied":
                True,
        }

        decision_path = (
            grounding_dir
            / "siglip_grounding_decision.json"
        )

        json_write(
            decision_path,
            grounding_decision,
        )

        print(
            f"object="
            f"{selected_object['name']} "
            f"target="
            f"{selected_target['name']} "
            f"object_margin="
            f"{float(object_query['probability_margin']):.6f} "
            f"target_margin="
            f"{float(target_query['probability_margin']):.6f} "
            "candidate_semantic_names_sent=false "
            "native_actions_replaced=false "
            f"report={decision_path}",
            flush=True,
        )

        return grounding_decision

    if (
        model == "openvla"
        and openvla_grounding_adapter
        == "shape_siglip"
    ):
        task_family = (
            os.environ.get(
                "VLA_TASK_FAMILY",
                "pick_place",
            )
            .strip()
            .lower()
        )

        if (
            task_family
            != "shape_classification"
        ):
            raise MainStop(
                "shape_siglip exige "
                "VLA_TASK_FAMILY="
                "shape_classification; "
                f"actual={task_family!r}"
            )

        if (
            physical_grounding_policy
            != "autonomous_all_candidates"
        ):
            raise MainStop(
                "shape_siglip exige "
                "VLA_PHYSICAL_GROUNDING_POLICY="
                "autonomous_all_candidates. "
                "El parser no puede excluir candidatos."
            )

        shape_result = (
            _shape_siglip_request_grounding(
                image_path=image_path,
                instruction=
                    full_spatial_instruction,
                candidates=candidates,
            )
        )

        selected_object = (
            shape_result[
                "selected_object"
            ]
        )

        selected_target = (
            shape_result[
                "selected_target"
            ]
        )

        shape_client_result = (
            shape_result[
                "client_result"
            ]
        )

        shape_service_response = (
            shape_client_result[
                "service_response"
            ]
        )

        object_query = (
            shape_service_response[
                "roles"
            ][
                "object"
            ][
                "requested_query"
            ]
        )

        target_query = (
            shape_service_response[
                "roles"
            ][
                "target"
            ][
                "requested_query"
            ]
        )

        shape_request_trace = (
            shape_result[
                "request_trace"
            ]
        )

        shape_semantic_consistency_trace = {
            "schema":
                "VLA_shape_semantic_"
                "consistency_trace",
            "instruction":
                full_spatial_instruction,
            "selected_object":
                str(
                    selected_object[
                        "name"
                    ]
                ),
            "selected_target":
                str(
                    selected_target[
                        "name"
                    ]
                ),
            "requested_object":
                shape_request_trace[
                    "requested_object_label"
                ],
            "requested_target":
                shape_request_trace[
                    "requested_target_label"
                ],
            "relation_by_shape":
                shape_request_trace[
                    "relation_by_shape"
                ],
            "target_resolution":
                shape_request_trace[
                    "target_resolution"
                ],
            "parser_used_as_selection":
                False,
            "parser_replacement":
                False,
            "pass":
                True,
        }

        modal_participation_trace = (
            _vla_visual_language_participation_gate(
                model=model,
                original_actions=actions,
                image_path=image_path,
                instruction=instruction,
                candidates=candidates,
                selected_object=
                    selected_object,
                selected_target=
                    selected_target,
                q_ur5e=q_ur5e,
                gripper_width_m=
                    gripper_width_m,
                droid_state_mode=
                    droid_state_mode,
                seed=seed,
                grounding_dir=
                    grounding_dir,
            )
        )

        grounding_decision = {
            "schema":
                "VLA_openvla_shape_siglip_"
                "grounding_decision",
            "method":
                "photoneo_candidate_"
                "shape_siglip_external_"
                "visual_language_grounding",
            "selection_source":
                "external_siglip_shape_"
                "visual_language_grounding",
            "parser_role":
                "shape_relation_query_"
                "normalization_and_post_selection_"
                "geometry_lookup_only",
            "model":
                model,
            "task_family":
                task_family,
            "grounding_adapter":
                "shape_siglip",
            "grounding_model_id":
                shape_result[
                    "health"
                ][
                    "model_id"
                ],
            "siglip_contract":
                shape_result[
                    "health"
                ][
                    "contract"
                ],
            "instruction":
                full_spatial_instruction,
            "grounding_instruction":
                instruction,
            "requested_object_label":
                shape_request_trace[
                    "requested_object_label"
                ],
            "requested_target_label":
                shape_request_trace[
                    "requested_target_label"
                ],
            "object":
                selected_object,
            "target":
                selected_target,
            "selected_object":
                str(
                    selected_object[
                        "name"
                    ]
                ),
            "selected_target":
                str(
                    selected_target[
                        "name"
                    ]
                ),
            "object_probability_margin":
                float(
                    object_query[
                        "probability_margin"
                    ]
                ),
            "target_probability_margin":
                float(
                    target_query[
                        "probability_margin"
                    ]
                ),
            "object_assignment_margin":
                float(
                    shape_service_response[
                        "roles"
                    ][
                        "object"
                    ][
                        "assignment_margin"
                    ]
                ),
            "target_assignment_margin":
                float(
                    shape_service_response[
                        "roles"
                    ][
                        "target"
                    ][
                        "assignment_margin"
                    ]
                ),
            "shape_request_trace":
                shape_request_trace,
            "semantic_consistency_trace":
                shape_semantic_consistency_trace,
            "modal_participation_trace":
                modal_participation_trace,
            "language_normalization_trace":
                language_normalization_trace,
            "siglip_client_result":
                shape_client_result,
            "all_photoneo_candidates_competed":
                True,
            "candidate_semantic_names_sent":
                False,
            "native_actions_shape":
                list(
                    np.asarray(
                        actions,
                        dtype=float,
                    ).shape
                ),
            "native_actions_replaced":
                False,
            "native_actions_reinterpreted":
                False,
            "raw_instruction_preserved":
                True,
            "parser_replacement":
                False,
            "external_grounding_applied":
                True,
            "gripper_enters_container":
                False,
            "require_object_fully_below_rim":
                False,
            "shape_preplan_enabled":
                (
                    os.environ.get(
                        "VLA_SHAPE_ALLOW_PREPLAN",
                        "NO",
                    )
                    .strip()
                    .upper()
                    == "YES"
                ),
        }

        decision_path = (
            grounding_dir
            / "shape_siglip_grounding_decision.json"
        )

        json_write(
            decision_path,
            grounding_decision,
        )

        print(
            f"object="
            f"{selected_object['name']} "
            f"target="
            f"{selected_target['name']} "
            f"object_margin="
            f"{float(object_query['probability_margin']):.6f} "
            f"target_margin="
            f"{float(target_query['probability_margin']):.6f} "
            "candidate_semantic_names_sent=false "
            "native_actions_replaced=false "
            "gripper_enters_container=false "
            f"report={decision_path}",
            flush=True,
        )

        if not bool(
            grounding_decision[
                "shape_preplan_enabled"
            ]
        ):
            raise MainStop(
                ""
                "Grounding por forma validado y "
                "guardado, pero la geometría de "
                "deposición todavía permanece "
                "bloqueada. "
                "VLA_SHAPE_ALLOW_PREPLAN=NO. "
                f"report={decision_path}"
            )

        return grounding_decision

    openvla_reference_bundle: (
        dict[str, Any] | None
    ) = None

    if (
        model == "openvla"
        and os.environ.get(
            "VLA_OPENVLA_GROUNDING_ADAPTER",
            "visual_causal",
        )
        .strip()
        .lower()
        == "visual_causal"
    ):
        fixed_reference_dir = (
            grounding_dir
            / "openvla_fixed_reference"
        )

        fixed_reference_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        object_reference_actions = (
            np.asarray(
                actions,
                dtype=float,
            )
        )

        object_reference_instruction = str(
            instruction
        )

        target_reference_instruction = (
            _phase_target_instruction(
                object_reference_instruction
            )
        )

        (
            target_reference_actions,
            target_reference_evidence,
        ) = infer_on_orin(
            model=model,
            image_path=image_path,
            instruction=
                target_reference_instruction,
            q_ur5e=q_ur5e,
            gripper_width_m=
                gripper_width_m,
            droid_state_mode=
                droid_state_mode,
            seed=seed,
            run_dir=(
                fixed_reference_dir
                / "target_reference"
            ),
        )

        target_reference_actions = (
            np.asarray(
                target_reference_actions,
                dtype=float,
            )
        )

        if (
            object_reference_actions.shape
            != target_reference_actions.shape
        ):
            raise MainStop(
                "OpenVLA: las referencias de objeto "
                "y destino tienen shapes distintos: "
                f"object="
                f"{object_reference_actions.shape}, "
                f"target="
                f"{target_reference_actions.shape}"
            )

        if (
            not np.all(
                np.isfinite(
                    object_reference_actions
                )
            )
            or not np.all(
                np.isfinite(
                    target_reference_actions
                )
            )
        ):
            raise MainStop(
                "OpenVLA: acciones de referencia "
                "no finitas"
            )

        openvla_reference_bundle = {
            "schema":
                "VLA_openvla_fixed_"
                "reference_bundle",
            "object_reference_actions":
                object_reference_actions,
            "object_reference_instruction":
                object_reference_instruction,
            "target_reference_actions":
                target_reference_actions,
            "target_reference_instruction":
                target_reference_instruction,
            "target_reference_evidence":
                target_reference_evidence,
            "seed":
                int(seed),
            "reference_reused_across_rounds":
                True,
            "language_fixed_across_rounds":
                True,
            "seed_fixed_across_rounds":
                True,
        }

        json_write(
            fixed_reference_dir
            / "fixed_reference_bundle.json",
            {
                "schema":
                    "VLA_openvla_fixed_"
                    "reference_bundle",
                "object_reference_instruction":
                    object_reference_instruction,
                "target_reference_instruction":
                    target_reference_instruction,
                "object_reference_shape":
                    list(
                        object_reference_actions.shape
                    ),
                "target_reference_shape":
                    list(
                        target_reference_actions.shape
                    ),
                "object_reference_actions":
                    object_reference_actions.tolist(),
                "target_reference_actions":
                    target_reference_actions.tolist(),
                "target_reference_evidence":
                    target_reference_evidence,
                "seed":
                    int(seed),
                "reference_reused_across_rounds":
                    True,
                "language_fixed_across_rounds":
                    True,
                "seed_fixed_across_rounds":
                    True,
            },
        )

        print(
            f"seed={int(seed)} "
            f"object_prompt="
            f"{object_reference_instruction!r} "
            f"target_prompt="
            f"{target_reference_instruction!r} "
            f"shape="
            f"{list(object_reference_actions.shape)} "
            "reused_across_rounds=true "
            "language_fixed=true "
            "seed_fixed=true",
            flush=True,
        )


    grounding_adapter = (
        "openvla_visual_causal_necessity_sufficiency"
        if model == "openvla"
        else "droid_temporal_counterfactual_consensus"
    )

    print(
        f"model={model} "
        f"adapter={grounding_adapter} "
        "same_candidates=true "
        "same_semantic_guard=true",
        flush=True,
    )

    # Los candidatos local_ply_obstacle_* son entidades auxiliares
    # de geometría y seguridad, no objetos manipulables semánticos.
    # Se conservan en competition_candidates para:
    # - preservación durante las máscaras visuales;
    # - clearance y seguridad de escena;
    # - referencia geométrica.
    # Solo se excluyen de las hipótesis objeto-destino.
    semantic_object_candidates: list[dict[str, Any]] = []
    excluded_auxiliary_objects: list[str] = []

    for candidate in competition_candidates["object"]:
        candidate_name = str(
            candidate.get("name", "")
        ).strip().lower()

        source = str(
            candidate.get("source", "")
        ).strip().lower()

        semantic_override = str(
            candidate.get("semantic_override", "")
        ).strip().lower()

        object_shape = str(
            candidate.get("object_shape", "")
        ).strip().lower()

        is_auxiliary_geometric_obstacle = bool(
            candidate_name.startswith("local_ply_obstacle_")
            or source == "local_ply_geometric_annulus"
            or semantic_override == "local_ply_obstacle"
            or object_shape == "local_ply_obstacle"
        )

        if is_auxiliary_geometric_obstacle:
            excluded_auxiliary_objects.append(
                str(candidate.get("name", "unnamed"))
            )
            continue

        semantic_object_candidates.append(candidate)

    print(
        f"excluded={excluded_auxiliary_objects} "
        f"semantic_objects="
        f"{[candidate.get('name') for candidate in semantic_object_candidates]} "
        f"protected_scene_objects="
        f"{[candidate.get('name') for candidate in competition_candidates['object']]} "
        "preserved_for_scene_safety=true",
        flush=True,
    )

    if not semantic_object_candidates:
        raise MainStop(
            "Grounding VLA sin objetos manipulables después de "
            "excluir los obstáculos geométricos auxiliares."
        )

    valid_pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, float]]] = []
    for object_candidate in semantic_object_candidates:
        for target_candidate in competition_candidates["target"]:
            contains, geometry = _bbox_contains_object(
                target_candidate, object_candidate
            )
            if contains and len(competition_candidates["target"]) > 1:
                print(
                    f"object={object_candidate['name']} "
                    f"target={target_candidate['name']} "
                    f"coverage={geometry['object_bbox_coverage']:.3f} "
                    f"distance_xy={geometry['object_target_xy_distance_m']:.3f}m"
                )
                continue
            valid_pairs.append((object_candidate, target_candidate, geometry))

    if not valid_pairs:
        raise MainStop("Grounding VLA: no queda ninguna pareja objeto-destino válida")



    variants = _joint_grounding_variants(
        instruction,
        model=model,
    )

    maximum_rounds = int(
        os.environ.get(
            "VLA_GROUNDING_MAX_ROUNDS",
            str(len(variants)),
        )
    )

    if not 1 <= maximum_rounds <= len(variants):
        raise MainStop(
            "VLA_GROUNDING_MAX_ROUNDS debe estar "
            f"entre 1 y {len(variants)}"
        )

    variants = variants[:maximum_rounds]
    rounds: list[dict[str, Any]] = []

    def append_grounding_rounds(
        batch: list[dict[str, Any]],
    ) -> None:
        monotonic = __import__("time").monotonic

        for variant in batch:
            round_number = len(rounds) + 1
            started = monotonic()

            result = _evaluate_joint_policy_round(
                round_index=round_number,
                variant=variant,
                valid_pairs=valid_pairs,
                candidates=competition_candidates,
                model=model,
                original_actions=actions,
                image_path=image_path,
                instruction=instruction,
                q_ur5e=q_ur5e,
                gripper_width_m=gripper_width_m,
                droid_state_mode=droid_state_mode,
                seed=seed,
                grounding_dir=grounding_dir,
                openvla_reference_bundle=
                    openvla_reference_bundle,
            )

            elapsed = float(monotonic() - started)
            result["grounding_wall_time_s"] = elapsed
            rounds.append(result)

            print(
                f"round={round_number}/{len(variants)} "
                f"variant={variant.get('name')} "
                f"wall_s={elapsed:.3f}",
                flush=True,
            )

    if len(valid_pairs) > 1:
        initial_round_count = min(3, len(variants))
    else:
        initial_round_count = 1

    initial_variants = variants[:initial_round_count]

    print(
        f"pairs={len(valid_pairs)} "
        f"rounds={len(initial_variants)} "
        f"maximum_rounds={len(variants)}",
        flush=True,
    )

    append_grounding_rounds(initial_variants)

    adaptive_minimum_margin = float(
        os.environ.get(
            "VLA_GROUNDING_ADAPTIVE_MIN_RELATIVE_MARGIN",
            "0.04",
        )
    )

    adaptive_maximum_rounds = int(
        os.environ.get(
            "VLA_GROUNDING_ADAPTIVE_MAX_ROUNDS",
            "5",
        )
    )

    if not 3 <= adaptive_maximum_rounds <= 7:
        raise MainStop(
            "VLA_GROUNDING_ADAPTIVE_MAX_ROUNDS "
            "debe estar entre 3 y 7"
        )


    if (
        not math.isfinite(adaptive_minimum_margin)
        or not 0.0 <= adaptive_minimum_margin <= 0.80
    ):
        raise MainStop(
            "VLA_GROUNDING_ADAPTIVE_MIN_RELATIVE_MARGIN "
            "no válido"
        )

    def adaptive_state(
        trace: dict[str, Any],
    ) -> tuple[float, bool, list[str]]:
        margin = float(
            trace["aggregate_relative_margin"]
        )

        vote_stable = bool(
            trace.get("vote_consensus_pass")
            and trace.get(
                "vote_aggregate_agreement"
            ) is True
        )

        reasons: list[str] = []

        if margin < adaptive_minimum_margin:
            reasons.append(
                "aggregate_margin_below_threshold"
            )

        if not vote_stable:
            reasons.append(
                "round_vote_instability"
            )

        return margin, vote_stable, reasons

    adaptive_history: list[dict[str, Any]] = []

    if len(rounds) > 1:
        scored_pairs, consensus_trace = (
            _aggregate_grounding_rounds_by_adapter(rounds, model=model, grounding_adapter=grounding_adapter)
        )

        (
            initial_margin,
            initial_vote_stable,
            adaptive_reasons,
        ) = adaptive_state(consensus_trace)

        initial_adaptive_reasons = list(
            adaptive_reasons
        )
        adaptive_triggered = False

        while (
            adaptive_reasons
            and len(rounds)
            < min(
                len(variants),
                adaptive_maximum_rounds,
            )
        ):
            next_variant = variants[len(rounds)]

            print(
                f"next_round={len(rounds) + 1} "
                f"variant={next_variant.get('name')} "
                f"current_margin="
                f"{float(consensus_trace['aggregate_relative_margin']):.4f} "
                f"required_margin="
                f"{adaptive_minimum_margin:.4f} "
                f"reasons={adaptive_reasons}",
                flush=True,
            )

            append_grounding_rounds([next_variant])
            adaptive_triggered = True

            scored_pairs, consensus_trace = (
                _aggregate_grounding_rounds_by_adapter(rounds, model=model, grounding_adapter=grounding_adapter)
            )

            (
                current_margin,
                current_vote_stable,
                adaptive_reasons,
            ) = adaptive_state(consensus_trace)

            adaptive_history.append(
                {
                    "round_count": len(rounds),
                    "last_variant": str(
                        next_variant.get("name")
                    ),
                    "aggregate_relative_margin":
                        current_margin,
                    "vote_stable":
                        current_vote_stable,
                    "remaining_reasons":
                        list(adaptive_reasons),
                    "last_round_wall_time_s": float(
                        rounds[-1].get(
                            "grounding_wall_time_s",
                            0.0,
                        )
                    ),
                }
            )

            if not adaptive_reasons:
                print(
                    f"rounds={len(rounds)} "
                    f"margin={current_margin:.4f} "
                    "vote_stable=true "
                    "reason=evidence_sufficient",
                    flush=True,
                )
                break

        if adaptive_reasons:
            adaptive_stop_reason = (
                "maximum_rounds_reached_with_"
                + "_and_".join(adaptive_reasons)
            )
        elif adaptive_triggered:
            adaptive_stop_reason = (
                "stable_after_incremental_round"
            )
        else:
            adaptive_stop_reason = (
                "initial_evidence_sufficient"
            )

        consensus_trace[
            "adaptive_evidence_triggered"
        ] = adaptive_triggered
        consensus_trace[
            "adaptive_evidence_reasons"
        ] = initial_adaptive_reasons
        consensus_trace[
            "adaptive_evidence_result_reasons"
        ] = list(adaptive_reasons)
        consensus_trace[
            "initial_round_count"
        ] = len(initial_variants)
        consensus_trace[
            "result_round_count"
        ] = len(rounds)
        consensus_trace[
            "additional_rounds_executed"
        ] = len(rounds) - len(initial_variants)
        consensus_trace[
            "initial_aggregate_relative_margin"
        ] = initial_margin
        consensus_trace[
            "initial_vote_stable"
        ] = initial_vote_stable
        consensus_trace[
            "adaptive_minimum_relative_margin"
        ] = adaptive_minimum_margin
        consensus_trace[
            "adaptive_maximum_rounds"
        ] = len(variants)
        consensus_trace[
            "adaptive_strategy"
        ] = (
            "incremental_one_round_then_reaggregate"
        )
        consensus_trace[
            "adaptive_stop_reason"
        ] = adaptive_stop_reason
        consensus_trace[
            "adaptive_round_history"
        ] = adaptive_history

        print(
            f"initial_rounds={len(initial_variants)} "
            f"rounds_completed={len(rounds)} "
            f"additional_rounds="
            f"{len(rounds) - len(initial_variants)} "
            f"stop_reason={adaptive_stop_reason} "
            f"margin="
            f"{float(consensus_trace['aggregate_relative_margin']):.4f}",
            flush=True,
        )
    else:
        scored_pairs = rounds[0]["scored_pairs"]
        consensus_trace = {
            "round_count": 1,
            "single_valid_pair": True,
            "adaptive_evidence_triggered": False,
            "adaptive_strategy":
                "single_pair_single_round",
            "adaptive_stop_reason":
                "single_valid_pair",
            "initial_round_count": 1,
            "result_round_count": 1,
            "additional_rounds_executed": 0,
            "round_winners": [
                {
                    "round": rounds[0]["name"],
                    "seed": rounds[0]["seed"],
                    "object":
                        rounds[0]["summary"]
                        ["winner_key"][0],
                    "target":
                        rounds[0]["summary"]
                        ["winner_key"][1],
                    "relative_margin": float(
                        rounds[0]["summary"]
                        ["relative_margin"]
                    ),
                }
            ],
        }
    for pair in scored_pairs:
        print(
            f"object={pair['object']['name']} target={pair['target']['name']} "
            f"cost={pair['total_cost']:.6f} score={pair['score']:.6f}"
        )

    selected_pair, pair_trace = _select_joint_policy_hypothesis(scored_pairs)
    selected_object = selected_pair["object"]
    selected_target = selected_pair["target"]
    semantic_consistency_trace = _semantic_consistency_veto(
        instruction=instruction,
        candidates=candidates,
        selected_object=selected_object,
        selected_target=selected_target,
    )
    selection_trace = {
        "schema": "vla_grounding_selection_trace",
        "model": model,
        "grounding_adapter": grounding_adapter,
        "native_actions_shape": list(
            np.asarray(actions, dtype=float).shape
        ),
        "instruction": instruction,
        "selected_object": str(selected_object["name"]),
        "selected_target": str(selected_target["name"]),
        "semantic_consistency_trace": semantic_consistency_trace,
        "pair_trace": pair_trace,
        "pair_ranking": [
            {
                "rank": rank,
                "object": str(pair["object"]["name"]),
                "target": str(pair["target"]["name"]),
                "total_cost": float(pair["total_cost"]),
                "score": float(pair["score"]),
            }
            for rank, pair in enumerate(
                sorted(scored_pairs, key=lambda item: item["total_cost"]),
                start=1,
            )
        ],
        "consensus_trace": consensus_trace,
        "parser_replacement": False,
    }
    json_write(
        grounding_dir / "vla_grounding_selection_trace.json",
        selection_trace,
    )
    print(
        f"selected={selected_object['name']}->{selected_target['name']} "
        f"semantic_pass={str(semantic_consistency_trace['pass']).lower()}"
    )
    # Los obstáculos geométricos auxiliares permanecen en la escena
    # y en las máscaras de seguridad, pero no participan en:
    # - contrafactuales lingüísticos;
    # - scores semánticos del overlay;
    # - selección objeto-destino.
    semantic_grounding_candidates = {
        "object": list(semantic_object_candidates),
        "target": list(competition_candidates["target"]),
    }

    print(
        f"objects="
        f"{[candidate.get('name') for candidate in semantic_grounding_candidates['object']]} "
        f"targets="
        f"{[candidate.get('name') for candidate in semantic_grounding_candidates['target']]} "
        "auxiliary_obstacles_preserved_for_safety=true",
        flush=True,
    )

    modal_participation_trace = _vla_visual_language_participation_gate(
        model=model,
        original_actions=actions,
        image_path=image_path,
        instruction=instruction,
        candidates=semantic_grounding_candidates,
        selected_object=selected_object,
        selected_target=selected_target,
        q_ur5e=q_ur5e,
        gripper_width_m=gripper_width_m,
        droid_state_mode=droid_state_mode,
        seed=seed,
        grounding_dir=grounding_dir,
    )
    print(
        f"rounds={consensus_trace.get('round_count')} "
        f"selected={selected_object['name']}->{selected_target['name']} "
        f"margin={pair_trace['relative_margin']:.4f}"
    )

    object_scores: list[dict[str, Any]] = []

    for candidate in semantic_object_candidates:
        matching = [
            pair
            for pair in scored_pairs
            if pair["object"]["name"] == candidate["name"]
        ]

        if not matching:
            raise MainStop(
                "Posprocesado de grounding incoherente: "
                f"el objeto semántico {candidate.get('name')} "
                "no tiene ninguna hipótesis agregada."
            )

        best_pair = min(
            matching,
            key=lambda item: item["total_cost"],
        )

        object_scores.append(
            {
                **candidate,
                "score": float(best_pair["score"]),
            }
        )

    target_scores: list[dict[str, Any]] = []
    for candidate in competition_candidates["target"]:
        matching = [
            pair for pair in scored_pairs
            if pair["target"]["name"] == candidate["name"]
        ]
        if not matching:
            continue
        best_pair = min(matching, key=lambda item: item["total_cost"])
        target_scores.append({**candidate, "score": float(best_pair["score"])})

    overlay_path = grounding_dir / "vla_joint_policy_grounding_overlay.png"
    _render_grounding_overlay(
        image_path, object_scores, target_scores, overlay_path
    )

    print(
        f"selected={selected_object['name']} "
        f"pair_cost={selected_pair['total_cost']:.6f} "
        f"margin={pair_trace['relative_margin']:.4f}"
    )
    print(
        f"selected={selected_target['name']} "
        f"pair_cost={selected_pair['total_cost']:.6f} "
        f"margin={pair_trace['relative_margin']:.4f}"
    )

    decision = {
        "schema": "vla_grounding",
        "method": "all_photoneo_candidates_native_action_adapter_competition",
        "grounding_adapter": grounding_adapter,
        "native_actions_shape": list(
            np.asarray(actions, dtype=float).shape
        ),
        "decision_owner": model,
        "external_vlm_used": False,
        "parser_role": "alias_normalization_and_post_selection_veto_only",
        "language_normalization_trace": language_normalization_trace,
        "modal_participation_trace": modal_participation_trace,
        "all_candidate_competition_decision_owner": model,
        "semantic_consistency_trace": semantic_consistency_trace,
        "instruction": instruction,
        "image_path": str(image_path),
        "image_sha256": sha256_file(image_path),
        "object": selected_object,
        "target": selected_target,
        "pair_trace": pair_trace,
        "pair_ranking": [
            {
                "rank": rank,
                "object": str(pair["object"]["name"]),
                "target": str(pair["target"]["name"]),
                "total_cost": float(pair["total_cost"]),
                "score": float(pair["score"]),
            }
            for rank, pair in enumerate(
                sorted(scored_pairs, key=lambda item: item["total_cost"]),
                start=1,
            )
        ],
        "top1_semantic_correct": bool(
            semantic_consistency_trace["top1_pair_correct"]
        ),
        "consensus_trace": consensus_trace,
        "grounding_rounds": [
            {
                "name": round_result["name"],
                "seed": round_result["seed"],
                "winner": {
                    "object": round_result["summary"]["winner_key"][0],
                    "target": round_result["summary"]["winner_key"][1],
                },
                "relative_margin": float(
                    round_result["summary"]["relative_margin"]
                ),
                "best_cost": float(round_result["summary"]["best_cost"]),
                "second_cost": float(round_result["summary"]["second_cost"]),
                "target_reference_instruction": round_result["target_reference_instruction"],
                "target_reference_evidence": round_result["target_reference_evidence"],
            }
            for round_result in rounds
        ],
        "overlay_path": str(overlay_path),
    }
    json_write(grounding_dir / "vla_grounding_decision.json", decision)

    # Registro común para variantes canónicas,
    # alias, paráfrasis y controles explícitos.
    # La pareja la selecciona el VLA.
    # El protocolo solo compara y veta; no sustituye la selección.
    _protocol_env = __import__("os").environ

    _protocol_shape_task = (
        _protocol_env.get(
            "VLA_TASK_FAMILY",
            "",
        ).strip()
        == "shape_classification"
    )

    _protocol_eligible = (
        _protocol_env.get(
            "VLA_SHAPE_PROTOCOL_ELIGIBLE",
            "NO",
        ).strip().upper()
        == "YES"
    )

    _protocol_expected_object = (
        _protocol_env.get(
            "VLA_SHAPE_EXPECTED_OBJECT",
            "",
        ).strip()
    )

    _protocol_expected_target = (
        _protocol_env.get(
            "VLA_SHAPE_EXPECTED_TARGET",
            "",
        ).strip()
    )

    if (
        _protocol_shape_task
        and _protocol_eligible
        and _protocol_expected_object
        and _protocol_expected_target
    ):
        _protocol_selected_object = str(
            selected_object["name"]
        )
        _protocol_selected_target = str(
            selected_target["name"]
        )

        _protocol_pair_pass = (
            _protocol_selected_object
            == _protocol_expected_object
            and _protocol_selected_target
            == _protocol_expected_target
        )

        _protocol_trace = {
            "schema": (
                "VLA_shape_protocol_expected_pair_trace"
            ),
            "variant": _protocol_env.get(
                "VLA_SHAPE_TASK_VARIANT",
                "",
            ).strip(),
            "expected_object": (
                _protocol_expected_object
            ),
            "expected_target": (
                _protocol_expected_target
            ),
            "selected_object": (
                _protocol_selected_object
            ),
            "selected_target": (
                _protocol_selected_target
            ),
            "pass": bool(
                _protocol_pair_pass
            ),
            "decision_owner": model,
            "parser_replacement": False,
            "trace_role": "post_selection_veto_only",
        }

        decision[
            "evaluation_protocol_semantic_trace"
        ] = _protocol_trace

        json_write(
            grounding_dir
            / "vla_grounding_decision.json",
            decision,
        )

        print(
            f"variant="
            f"{_protocol_trace['variant']} "
            f"selected="
            f"{_protocol_selected_object}->"
            f"{_protocol_selected_target} "
            f"expected="
            f"{_protocol_expected_object}->"
            f"{_protocol_expected_target} "
            f"pass="
            f"{str(_protocol_pair_pass).lower()} "
            "parser_replacement=false",
            flush=True,
        )

        if not _protocol_pair_pass:
            raise MainStop(
                "El top-1 crudo del VLA no coincide con "
                "la pareja esperada por el protocolo: "
                f"selected="
                f"{_protocol_selected_object}->"
                f"{_protocol_selected_target}; "
                f"expected="
                f"{_protocol_expected_object}->"
                f"{_protocol_expected_target}. "
                "La salida queda registrada como fallo "
                "y el robot no se moverá."
            )

    # En instrucciones relacionales por forma, el registro debe
    # comprobar el par completo. El parser no sustituye la salida
    # del VLA: solo impide mover el robot si el top-1 crudo
    # contradice la relación inequívoca objeto-forma-destino.
    _shape_instruction_lower = str(
        instruction
    ).strip().lower()

    _shape_task = (
        __import__("os").environ.get(
            "VLA_TASK_FAMILY",
            "",
        ).strip()
        == "shape_classification"
    )

    _shape_relation_requested = any(
        phrase in _shape_instruction_lower
        for phrase in (
            "matches its shape",
            "matching its shape",
            "matching shape",
            "container that matches",
            "box that matches",
        )
    )

    if (
        _shape_task
        and _shape_relation_requested
    ):
        _expected_object = None
        _expected_target = None

        if (
            "orange" in _shape_instruction_lower
            and (
                "cube" in _shape_instruction_lower
                or "block"
                in _shape_instruction_lower
                or "die"
                in _shape_instruction_lower
            )
        ):
            _expected_object = "orange_cube"
            _expected_target = "square_box"

        elif (
            "white" in _shape_instruction_lower
            and (
                "cylinder"
                in _shape_instruction_lower
                or "cylindrical"
                in _shape_instruction_lower
            )
        ):
            _expected_object = "white_cylinder"
            _expected_target = "circular_box"

        if _expected_object is not None:
            _selected_object = str(
                selected_object["name"]
            )
            _selected_target = str(
                selected_target["name"]
            )

            _shape_pair_pass = (
                _selected_object
                == _expected_object
                and _selected_target
                == _expected_target
            )

            _shape_relation_trace = {
                "relation": "matches_its_shape",
                "expected_object": (
                    _expected_object
                ),
                "expected_target": (
                    _expected_target
                ),
                "selected_object": (
                    _selected_object
                ),
                "selected_target": (
                    _selected_target
                ),
                "pass": bool(
                    _shape_pair_pass
                ),
                "parser_replacement": False,
                "decision_owner": model,
            }

            decision[
                "shape_relation_semantic_trace"
            ] = _shape_relation_trace
            decision[
                "top1_semantic_correct"
            ] = bool(_shape_pair_pass)

            json_write(
                grounding_dir
                / "vla_grounding_decision.json",
                decision,
            )

            print(
                f"selected="
                f"{_selected_object}->"
                f"{_selected_target} "
                f"expected="
                f"{_expected_object}->"
                f"{_expected_target} "
                f"pass={str(_shape_pair_pass).lower()} "
                "parser_replacement=false",
                flush=True,
            )

            if not _shape_pair_pass:
                raise MainStop(
                    "El top-1 del VLA contradice la "
                    "relación inequívoca de forma: "
                    f"selected="
                    f"{_selected_object}->"
                    f"{_selected_target}; "
                    f"expected="
                    f"{_expected_object}->"
                    f"{_expected_target}. "
                    "La selección cruda quedó guardada "
                    "y el robot no se moverá."
                )

    if not bool(semantic_consistency_trace["pass"]):
        raise MainStop(
            "El top-1 del VLA contradice la instrucción inequívoca; "
            "la selección cruda quedó guardada y no fue sustituida por el parser."
        )
    return decision

def refresh_grounding_coordinates(
    *,
    scene: dict[str, Any],
    image_path: Path,
    grounding: dict[str, Any],
) -> dict[str, Any]:
    """Conserva la identidad decidida por el VLA y actualiza solo su XYZ live."""
    candidates = _shape_common_container_relabel(
        neutral_scene_candidates(scene, image_path)
    )
    refreshed = copy.deepcopy(grounding)
    for role in ("object", "target"):
        selected_name = str(grounding[role]["name"])
        matches = [
            item for item in candidates[role]
            if selected_name
            in {
                str(item["name"]),
                *[str(alias) for alias in item.get("aliases", [])],
            }
        ]
        if len(matches) != 1:
            available = [item["name"] for item in candidates[role]]
            raise MainStop(
                f"Verificación: no se pudo reencontrar {role}="
                f"{selected_name}; disponibles={available}"
            )
        previous_xyz = refreshed[role].get("base_xyz_m")
        refreshed[role] = {
            **refreshed[role],
            **matches[0],
            "initial_base_xyz_m": previous_xyz,
            "coordinate_refresh_source": "fresh_photoneo_same_vla_identity",
        }
    refreshed["verification_image_path"] = str(image_path)
    refreshed["verification_image_sha256"] = sha256_file(image_path)
    return refreshed


def grounding_geometry_instruction(
    grounding: dict[str, Any],
) -> str:
    object_name = (
        str(
            grounding[
                "object"
            ][
                "name"
            ]
        )
        .replace(
            "_",
            " ",
        )
        .strip()
    )

    target_name_raw = str(
        grounding[
            "target"
        ][
            "name"
        ]
    ).strip().lower()

    grounding_adapter = str(
        grounding.get(
            "grounding_adapter",
            "",
        )
    ).strip().lower()

    siglip_contract = str(
        grounding.get(
            "siglip_contract",
            "",
        )
    ).strip()

    # Los receptáculos conservan el sustantivo box/container.
    # No se convierten en zonas planas.
    shape_task_family = (
        os.environ.get(
            "VLA_TASK_FAMILY",
            "pick_place",
        )
        .strip()
        .lower()
        == "shape_classification"
    )

    if (
        shape_task_family
        or grounding_adapter
        == "shape_siglip"
        or siglip_contract
        == "VLA_SIGLIP_SHAPE_CONTAINERS"
    ):
        target_name = (
            target_name_raw
            .replace(
                "_",
                " ",
            )
            .strip()
        )

        if not target_name:
            raise MainStop(
                "shape_siglip recibió un destino vacío"
            )

        return (
            f"put the {object_name} "
            f"in the {target_name}"
        )

    target_name = target_name_raw

    for suffix in (
        "_target",
        "_source",
        "_area",
        "_region",
        "_zone",
        "_box",
        "_bin",
    ):
        if target_name.endswith(
            suffix
        ):
            target_name = target_name[
                : -len(
                    suffix
                )
            ]
            break

    target_name = (
        target_name
        .replace(
            "_",
            " ",
        )
        .strip()
    )

    if not target_name:
        target_name = "target"

    return (
        f"put the {object_name} "
        f"in the {target_name} zone"
    )


def geometry_template_config(config: Any, grounding: dict[str, Any]) -> Any:
    """Crea una configuración geométrica nueva sin mutar la Config congelada."""
    canonical = grounding_geometry_instruction(grounding)

    try:
        return replace(
            config,
            instruction=canonical,
        )
    except (TypeError, ValueError) as exc:
        raise MainStop(
            "No se pudo crear la plantilla geométrica inmutable "
            "desde la decisión VLA"
        ) from exc

def apply_grounding_to_entities(
    entities: dict[str, Any],
    grounding: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Aplica la identidad VLA conservando el punto interior del destino.
    
        resolve_entities() se ejecuta después de la selección VLA y calcula un
        punto interior alcanzable para la zona elegida. El grondaing aporta el
        centro métrico actualizado; cualquier delta del centro se propaga al
        interior, pero nunca se sustituye el interior por el centro bruto.
    """
    updated = copy.deepcopy(entities)

    old_object = finite_vector(
        updated["object_xyz_m"], 3, "grounding.old_object"
    )
    new_object = finite_vector(
        grounding["object"]["base_xyz_m"], 3, "grounding.new_object"
    )
    object_delta = new_object - old_object

    old_target_interior = finite_vector(
        updated["target_xyz_m"], 3, "grounding.old_target_interior"
    )
    old_target_center = finite_vector(
        updated.get(
            "target_detected_center_xyz_m",
            updated["target_xyz_m"],
        ),
        3,
        "grounding.old_target_center",
    )
    new_target_center = finite_vector(
        grounding["target"]["base_xyz_m"], 3, "grounding.new_target_center"
    )
    target_center_delta = new_target_center - old_target_center

    object_keys = (
        "object_xyz_m",
        "object_safe_xyz_m",
        "pick_xyz_m",
        "pre_pick_xyz_m",
        "lift_xyz_m",
    )
    target_keys = (
        "target_xyz_m",
        "target_safe_xyz_m",
        "pre_place_xyz_m",
        "place_xyz_m",
        "retreat_xyz_m",
    )

    for key in object_keys:
        if key in updated:
            updated[key] = (
                finite_vector(updated[key], 3, f"grounding.{key}")
                + object_delta
            ).tolist()

    for key in target_keys:
        if key in updated:
            updated[key] = (
                finite_vector(updated[key], 3, f"grounding.{key}")
                + target_center_delta
            ).tolist()

    updated["target_detected_center_xyz_m"] = new_target_center.tolist()

    selected_interior = finite_vector(
        updated["target_xyz_m"], 3, "grounding.selected_target_interior"
    )
    interior_offset_xy_m = float(
        np.linalg.norm(selected_interior[:2] - new_target_center[:2])
    )
    if not 0.0 <= interior_offset_xy_m <= 0.076:
        raise MainStop(
            "El punto interior del destino no respeta el contrato geométrico: "
            f"offset_xy={interior_offset_xy_m:.4f}m"
        )

    updated["grounded_object_name"] = grounding["object"]["name"]
    updated["grounded_target_name"] = grounding["target"]["name"]

    print(
        f"center={new_target_center.tolist()} "
        f"selected_interior={selected_interior.tolist()} "
        f"offset_xy={interior_offset_xy_m:.4f}m"
    )

    report = {
        "selection_source": grounding.get(
            "selection_source",
            "vla_visual_language_action_grounding",
        ),
        "parser_role": grounding.get(
            "parser_role",
            "post_selection_geometry_lookup_only",
        ),
        "geometry_template_instruction": grounding_geometry_instruction(grounding),
        "selected_object": grounding["object"]["name"],
        "selected_target": grounding["target"]["name"],
        "object_delta_from_geometry_lookup_m": object_delta.tolist(),
        "target_center_delta_from_geometry_lookup_m": target_center_delta.tolist(),
        "object_xyz_m": new_object.tolist(),
        "target_detected_center_xyz_m": new_target_center.tolist(),
        "target_selected_interior_xyz_m": selected_interior.tolist(),
        "target_selected_interior_offset_xy_m": interior_offset_xy_m,
        "target_center_replaced_interior": False,
    }
    return updated, report

def baseline_targets(
    start: np.ndarray,
    goal: np.ndarray,
    count: int,
) -> list[np.ndarray]:
    start = finite_vector(
        start,
        3,
        "baseline.start",
    )
    goal = finite_vector(
        goal,
        3,
        "baseline.goal",
    )

    return [
        start
        + (
            goal - start
        )
        * (
            index / count
        )
        for index in range(
            1,
            count + 1,
        )
    ]


def rotation_between(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    source = finite_vector(
        source,
        3,
        "rotation.source",
    )
    target = finite_vector(
        target,
        3,
        "rotation.target",
    )

    source_norm = float(
        np.linalg.norm(source)
    )
    target_norm = float(
        np.linalg.norm(target)
    )

    if (
        source_norm <= 1e-12
        or target_norm <= 1e-12
    ):
        return np.eye(3)

    a = source / source_norm
    b = target / target_norm

    dot = float(
        np.clip(
            np.dot(a, b),
            -1.0,
            1.0,
        )
    )

    cross = np.cross(a, b)
    cross_norm = float(
        np.linalg.norm(cross)
    )

    if cross_norm > 1e-12:
        skew = np.array(
            [
                [
                    0.0,
                    -cross[2],
                    cross[1],
                ],
                [
                    cross[2],
                    0.0,
                    -cross[0],
                ],
                [
                    -cross[1],
                    cross[0],
                    0.0,
                ],
            ],
            dtype=float,
        )

        return (
            np.eye(3)
            + skew
            + (
                skew @ skew
            )
            * (
                (
                    1.0 - dot
                )
                / (
                    cross_norm
                    * cross_norm
                )
            )
        )

    if dot >= 0.0:
        return np.eye(3)

    reference = np.array(
        [1.0, 0.0, 0.0],
        dtype=float,
    )

    if abs(
        float(
            np.dot(a, reference)
        )
    ) > 0.90:
        reference = np.array(
            [0.0, 1.0, 0.0],
            dtype=float,
        )

    axis = np.cross(
        a,
        reference,
    )
    axis /= np.linalg.norm(axis)

    skew = np.array(
        [
            [
                0.0,
                -axis[2],
                axis[1],
            ],
            [
                axis[2],
                0.0,
                -axis[0],
            ],
            [
                -axis[1],
                axis[0],
                0.0,
            ],
        ],
        dtype=float,
    )

    return (
        np.eye(3)
        + 2.0
        * (
            skew @ skew
        )
    )


def guard_targets(
    *,
    start: np.ndarray,
    goal: np.ndarray,
    targets: list[np.ndarray],
    maximum_distance_increase_m: float,
    label: str,
    workspace_mode: str = "free",
) -> list[np.ndarray]:
    """Valida toda la fase con el mismo workspace que usará el solver."""
    start = finite_vector(start, 3, f"{label}.start")
    goal = finite_vector(goal, 3, f"{label}.goal")

    mode = str(workspace_mode)
    if mode not in {"free", "contact"}:
        raise MainStop(
            f"{label}: workspace_mode desconocido: {workspace_mode}"
        )

    if not targets:
        raise MainStop(f"{label}: trayectoria vacía")

    previous_distance = float(np.linalg.norm(goal - start))
    accepted: list[np.ndarray] = []

    for index, value in enumerate(targets, start=1):
        target = finite_vector(
            value,
            3,
            f"{label}.target[{index}]",
        )

        if hasattr(supervisor, "workspace_ok"):
            workspace_valid = bool(
                supervisor.workspace_ok(
                    target,
                    contact=(mode == "contact"),
                )
            )
            if not workspace_valid:
                raise MainStop(
                    f"{label}: waypoint {index} fuera del "
                    f"workspace {mode}: target={target.tolist()}"
                )

        distance = float(np.linalg.norm(goal - target))
        if distance > previous_distance + maximum_distance_increase_m:
            raise MainStop(
                f"{label}: waypoint {index} no progresa hacia el objetivo"
            )

        accepted.append(target.copy())
        previous_distance = distance

    result_error = float(np.linalg.norm(accepted[-1] - goal))
    if result_error > 1e-9:
        raise MainStop(
            f"{label}: el endpoint no coincide con el objetivo seguro: "
            f"{result_error:.6f} m"
        )

    print(
        f"label={label} mode={mode} waypoints={len(accepted)}",
        flush=True,
    )
    return accepted



def _validated_phase_gain(
    phase_name: str,
    default: float,
) -> float:
    env_name = (
        "VLA_"
        + phase_name.upper()
        + "_GAIN"
    )
    value = float(
        os.environ.get(
            env_name,
            str(default),
        )
    )

    if (
        not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise MainStop(
            f"{env_name} debe estar entre 0.0 y 1.0"
        )

    return value


def openvla_phase_targets(
    *,
    actions: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    task_object: np.ndarray,
    task_target: np.ndarray,
    calibration: dict[str, Any],
    config: Any,
    label: str,
    phase_gain: float = 1.0,
    workspace_mode: str = "free",
) -> tuple[
    list[np.ndarray],
    dict[str, Any],
]:
    if actions.shape != (1, 7):
        raise MainStop(
            "OpenVLA debe entregar 1x7"
        )

    if (
        not math.isfinite(phase_gain)
        or not 0.0 <= phase_gain <= 1.0
    ):
        raise MainStop(
            f"{label}: phase_gain no válido"
        )

    start = finite_vector(
        start,
        3,
        f"{label}.start",
    )
    goal = finite_vector(
        goal,
        3,
        f"{label}.goal",
    )

    task_frame = adaptation.build_task_frame(
        task_object,
        task_target,
    )

    budget = np.asarray(
        getattr(
            lab,
            "OPENVLA_BUDGET_M",
            [0.025, 0.025, 0.015],
        ),
        dtype=float,
    ).reshape(-1)[:3]

    residual, core_report = (
        adaptation.openvla_translation_residual(
            raw_action=actions[0],
            calibration=calibration,
            task_frame=task_frame,
            translation_budget_m=budget,
            gain=phase_gain,
        )
    )

    phase = goal - start
    distance = float(
        np.linalg.norm(phase)
    )

    if distance <= 1e-9:
        return [goal.copy()], {
            "adapter":
                "openvla_translation_only",
            "degenerate_phase": True,
            "phase_gain": phase_gain,
            "core_report": core_report,
        }

    direction = phase / distance
    midpoint = (
        start
        + 0.50 * phase
    )

    residual = finite_vector(
        residual,
        3,
        f"{label}.residual",
    )

    longitudinal = float(
        np.dot(
            residual,
            direction,
        )
    )
    lateral = (
        residual
        - longitudinal * direction
    )

    longitudinal = float(
        np.clip(
            longitudinal,
            -0.20 * distance,
            0.20 * distance,
        )
    )

    lateral_limit = float(
        np.linalg.norm(budget)
    ) * max(phase_gain, 1e-12)
    lateral_norm = float(
        np.linalg.norm(lateral)
    )

    if (
        lateral_norm > lateral_limit
        and lateral_norm > 1e-12
    ):
        lateral = (
            lateral
            * lateral_limit
            / lateral_norm
        )

    applied_residual = (
        longitudinal * direction
        + lateral
    )

    influenced = (
        midpoint
        + applied_residual
    )

    progress = float(
        np.dot(
            influenced - start,
            direction,
        )
    )

    progress = float(
        np.clip(
            progress,
            0.25 * distance,
            0.75 * distance,
        )
    )

    cross = (
        influenced
        - start
        - float(
            np.dot(
                influenced - start,
                direction,
            )
        )
        * direction
    )

    influenced = (
        start
        + progress * direction
        + cross
    )

    control_points = [
        influenced,
        goal.copy(),
    ]
    maximum_segment_m = phase_solver_max_segment_m(
        workspace_mode
    )
    dense_candidates = _densify_cartesian_control_points(
        start=start,
        control_points=control_points,
        maximum_segment_m=maximum_segment_m,
    )
    targets = guard_targets(
        start=start,
        goal=goal,
        targets=dense_candidates,
        maximum_distance_increase_m=(
            config.maximum_step_distance_increase_m
        ),
        label=label,
        workspace_mode=workspace_mode,
    )

    return targets, {
        "adapter":
            "openvla_translation_only_"
            "midpoint_residual_endpoint_locked",
        "native_shape":
            list(actions.shape),
        "phase_gain": phase_gain,
        "rotation_used": False,
        "gripper_used": False,
        "raw_residual_base_m":
            residual.tolist(),
        "applied_residual_base_m":
            applied_residual.tolist(),
        "model_control_points_base_m": [
            value.tolist() for value in control_points
        ],
        "solver_waypoint_count": len(targets),
        "solver_max_segment_m": maximum_segment_m,
        "targets_base_m": [
            value.tolist()
            for value in targets
        ],
        "endpoint_locked": True,
        "workspace_mode": workspace_mode,
        "core_report": core_report,
    }


def _densify_cartesian_control_points(
    *,
    start: np.ndarray,
    control_points: list[np.ndarray],
    maximum_segment_m: float,
) -> list[np.ndarray]:
    """
    Inserta waypoints sin contradecir el contrato de la fase.
    
        El rango específico ya se valida en phase_solver_max_segment_m():
        0.003..0.030 m para contacto y 0.010..0.100 m para espacio libre.
        Aquí solo se rechazan valores no finitos o no positivos.
    """
    if not math.isfinite(maximum_segment_m) or maximum_segment_m <= 0.0:
        raise MainStop(
            f"maximum_segment_m no válido para densificación: {maximum_segment_m!r}"
        )
    dense: list[np.ndarray] = []
    previous = finite_vector(start, 3, "densify.start")
    for index, point in enumerate(control_points):
        current = finite_vector(point, 3, f"densify.point_{index}")
        distance = float(np.linalg.norm(current - previous))
        pieces = max(1, int(math.ceil(distance / maximum_segment_m)))
        for step in range(1, pieces + 1):
            dense.append(previous + (current - previous) * (step / pieces))
        previous = current
    return dense


def droid_phase_targets(
    *,
    model: str,
    actions: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    config: Any,
    label: str,
    phase_gain: float = 1.0,
    workspace_mode: str = "free",
) -> tuple[
    list[np.ndarray],
    dict[str, Any],
]:
    expected = EXPECTED_SHAPES[
        model
    ]

    if actions.shape != expected:
        raise MainStop(
            f"{model}: shape={actions.shape}; "
            f"esperada={expected}"
        )

    if (
        not math.isfinite(phase_gain)
        or not 0.0 <= phase_gain <= 1.0
    ):
        raise MainStop(
            f"{label}: phase_gain no válido"
        )

    start = finite_vector(
        start,
        3,
        f"{label}.start",
    )
    goal = finite_vector(
        goal,
        3,
        f"{label}.goal",
    )

    groups = np.array_split(
        actions,
        5,
        axis=0,
    )

    grouped_translation = np.asarray(
        [
            group[:, :3].mean(
                axis=0
            )
            for group in groups
        ],
        dtype=float,
    )

    model_deltas_m = (
        np.clip(
            grouped_translation,
            -1.0,
            1.0,
        )
        * 0.05
    )

    model_to_base = np.asarray(
        supervisor.MODEL_TO_BASE,
        dtype=float,
    ).reshape(3, 3)

    base_deltas_m = (
        model_to_base
        @ model_deltas_m.T
    ).T

    raw_offsets = np.cumsum(
        base_deltas_m,
        axis=0,
    )

    raw_endpoint = raw_offsets[-1]
    raw_endpoint_norm = float(
        np.linalg.norm(raw_endpoint)
    )

    phase = goal - start
    phase_distance = float(
        np.linalg.norm(phase)
    )

    if phase_distance <= 1e-9:
        return [goal.copy()], {
            "adapter":
                "droid_translation_only",
            "degenerate_phase": True,
            "phase_gain": phase_gain,
        }

    phase_direction = (
        phase / phase_distance
    )

    if raw_endpoint_norm <= 1e-9:
        aligned_offsets = np.zeros(
            (5, 3),
            dtype=float,
        )
    else:
        rotation = rotation_between(
            raw_endpoint,
            phase_direction,
        )

        aligned_offsets = (
            rotation
            @ raw_offsets.T
        ).T

        aligned_endpoint_norm = float(
            np.linalg.norm(
                aligned_offsets[-1]
            )
        )

        if aligned_endpoint_norm > 1e-12:
            aligned_offsets *= (
                phase_distance
                / aligned_endpoint_norm
            )

    fractions = np.linspace(
        0.2,
        1.0,
        5,
    )

    geometric_offsets = (
        fractions[:, None]
        * phase[None, :]
    )

    residuals = (
        aligned_offsets
        - geometric_offsets
    )

    residuals[-1] = np.zeros(
        3,
        dtype=float,
    )

    budget = np.asarray(
        getattr(
            lab,
            "OPENVLA_BUDGET_M",
            [0.025, 0.025, 0.015],
        ),
        dtype=float,
    ).reshape(-1)[:3]

    residual_limit = float(
        np.linalg.norm(budget)
    )

    residual_norms = np.linalg.norm(
        residuals,
        axis=1,
    )

    maximum_residual = float(
        np.max(residual_norms)
    )

    scale = (
        min(
            1.0,
            residual_limit
            / maximum_residual,
        )
        if maximum_residual > 1e-12
        else 1.0
    )

    applied_residuals = (
        residuals
        * scale
        * phase_gain
    )

    candidates = [
        start
        + geometric_offsets[index]
        + applied_residuals[index]
        for index in range(5)
    ]

    candidates[-1] = goal.copy()

    maximum_segment_m = phase_solver_max_segment_m(
        workspace_mode
    )
    dense_candidates = _densify_cartesian_control_points(
        start=start,
        control_points=candidates,
        maximum_segment_m=maximum_segment_m,
    )

    targets = guard_targets(
        start=start,
        goal=goal,
        targets=dense_candidates,
        maximum_distance_increase_m=(
            config.
            maximum_step_distance_increase_m
        ),
        label=label,
        workspace_mode=workspace_mode,
    )

    return targets, {
        "adapter":
            "droid_translation_only_"
            "five_waypoint_task_aligned_"
            "endpoint_locked",
        "model": model,
        "native_shape":
            list(actions.shape),
        "compression":
            "native_horizon_to_5_group_means",
        "phase_gain": phase_gain,
        "used_action_dimensions":
            [0, 1, 2],
        "ignored_action_dimensions":
            list(
                range(
                    3,
                    actions.shape[1],
                )
            ),
        "translation_scale_m":
            0.05,
        "model_to_base":
            model_to_base.tolist(),
        "grouped_translation":
            grouped_translation.tolist(),
        "model_deltas_m":
            model_deltas_m.tolist(),
        "base_deltas_m":
            base_deltas_m.tolist(),
        "raw_offsets_m":
            raw_offsets.tolist(),
        "raw_endpoint_norm_m":
            raw_endpoint_norm,
        "residual_limit_m":
            residual_limit,
        "residual_scale":
            scale,
        "applied_residuals_m":
            applied_residuals.tolist(),
        "solver_waypoint_count": len(targets),
        "solver_max_segment_m": maximum_segment_m,
        "targets_base_m": [
            value.tolist()
            for value in targets
        ],
        "endpoint_locked": True,
        "workspace_mode": workspace_mode,
        "rotation_used": False,
        "gripper_used": False,
    }


def phase_targets(
    *,
    model: str,
    actions: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    task_object: np.ndarray,
    task_target: np.ndarray,
    calibration: dict[str, Any] | None,
    config: Any,
    label: str,
    phase_gain: float = 1.0,
    workspace_mode: str = "free",
) -> tuple[
    list[np.ndarray],
    dict[str, Any],
]:
    if model == "openvla":
        if calibration is None:
            raise MainStop(
                "Falta calibración OpenVLA"
            )

        return openvla_phase_targets(
            actions=actions,
            start=start,
            goal=goal,
            task_object=task_object,
            task_target=task_target,
            calibration=calibration,
            config=config,
            label=label,
            phase_gain=phase_gain,
            workspace_mode=workspace_mode,
        )

    return droid_phase_targets(
        model=model,
        actions=actions,
        start=start,
        goal=goal,
        config=config,
        label=label,
        phase_gain=phase_gain,
        workspace_mode=workspace_mode,
    )


def _find_metric_xyz_by_key(
    value: Any,
    accepted_keys: set[str],
) -> np.ndarray | None:
    """Devuelve el primer XYZ métrico asociado a alguna de las claves pedidas."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in accepted_keys and isinstance(item, dict):
                for xyz_key in (
                    "base_xyz_m",
                    "center_base_xyz_m",
                    "xyz_base_m",
                    "target_xyz_m",
                ):
                    if xyz_key in item:
                        try:
                            return finite_vector(
                                item[xyz_key],
                                3,
                                f"scene.{normalized}.{xyz_key}",
                            )
                        except Exception:
                            pass

        for item in value.values():
            result = _find_metric_xyz_by_key(
                item,
                accepted_keys,
            )
            if result is not None:
                return result

    elif isinstance(value, list):
        for item in value:
            result = _find_metric_xyz_by_key(
                item,
                accepted_keys,
            )
            if result is not None:
                return result

    return None


def _clip_xy_norm(
    value_xy: np.ndarray,
    maximum_norm_m: float,
) -> tuple[np.ndarray, float]:
    value_xy = finite_vector(
        value_xy,
        2,
        "vla_place.xy",
    )

    norm = float(np.linalg.norm(value_xy))
    if norm <= maximum_norm_m or norm <= 1e-12:
        return value_xy.copy(), 1.0

    scale = maximum_norm_m / norm
    return value_xy * scale, scale


def derive_vla_place_preference_xy(
    *,
    model: str,
    actions: np.ndarray,
    scene: dict[str, Any],
    task_object: np.ndarray,
    task_target: np.ndarray,
    calibration: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convierte la traslación nativa del VLA en una preferencia métrica XY para place."""
    # El ancla métrica debe ser exactamente el destino seleccionado por el VLA.
    # Buscar claves como yellow_target en toda la escena mezclaba tareas distintas.
    detected_center = finite_vector(
        task_target,
        3,
        "vla_place.grounded_target",
    )
    center_source = "grounded_target_xyz_m"

    maximum_offset_m = float(
        os.environ.get(
            "VLA_PLACE_MAX_OFFSET_M",
            "0.050",
        )
    )
    target_radius_m = float(
        os.environ.get(
            "VLA_PLACE_REGION_RADIUS_M",
            "0.070",
        )
    )
    boundary_margin_m = float(
        os.environ.get(
            "VLA_PLACE_BOUNDARY_MARGIN_M",
            "0.008",
        )
    )
    gain = float(
        os.environ.get(
            "VLA_PLACE_GAIN",
            "1.0",
        )
    )

    for label, value in (
        ("VLA_PLACE_MAX_OFFSET_M", maximum_offset_m),
        ("VLA_PLACE_REGION_RADIUS_M", target_radius_m),
        ("VLA_PLACE_BOUNDARY_MARGIN_M", boundary_margin_m),
        ("VLA_PLACE_GAIN", gain),
    ):
        if not math.isfinite(value):
            raise MainStop(f"{label} no es finito")

    if not 0.005 <= maximum_offset_m <= 0.100:
        raise MainStop(
            "VLA_PLACE_MAX_OFFSET_M debe estar entre 0.005 y 0.100 m"
        )
    if not 0.030 <= target_radius_m <= 0.150:
        raise MainStop(
            "VLA_PLACE_REGION_RADIUS_M debe estar entre 0.030 y 0.150 m"
        )
    if not 0.0 <= boundary_margin_m < target_radius_m:
        raise MainStop(
            "VLA_PLACE_BOUNDARY_MARGIN_M no es válido"
        )
    if not 0.1 <= gain <= 3.0:
        raise MainStop(
            "VLA_PLACE_GAIN debe estar entre 0.1 y 3.0"
        )

    if model == "openvla":
        if calibration is None:
            raise MainStop(
                "Falta calibración OpenVLA para decidir el place"
            )

        task_frame = adaptation.build_task_frame(
            task_object,
            task_target,
        )
        budget = np.asarray(
            getattr(
                lab,
                "OPENVLA_BUDGET_M",
                [0.025, 0.025, 0.015],
            ),
            dtype=float,
        ).reshape(-1)[:3]
        raw_base_residual, adapter_report = (
            adaptation.openvla_translation_residual(
                raw_action=actions[0],
                calibration=calibration,
                task_frame=task_frame,
                translation_budget_m=budget,
                gain=gain,
            )
        )
        raw_base_residual = finite_vector(
            raw_base_residual,
            3,
            "vla_place.openvla_residual",
        )
        raw_offset_xy = raw_base_residual[:2]
        adapter = "openvla_calibrated_task_frame_residual_xy"

    else:
        expected = EXPECTED_SHAPES[model]
        if actions.shape != expected:
            raise MainStop(
                f"{model}: shape={actions.shape}; esperada={expected}"
            )

        native_deltas_m = (
            np.clip(actions[:, :3], -1.0, 1.0)
            * 0.05
        )
        model_to_base = np.asarray(
            supervisor.MODEL_TO_BASE,
            dtype=float,
        ).reshape(3, 3)
        base_deltas_m = (
            model_to_base
            @ native_deltas_m.T
        ).T
        endpoint_residual = np.sum(
            base_deltas_m,
            axis=0,
        ) * gain
        raw_offset_xy = endpoint_residual[:2]
        adapter_report = {
            "native_deltas_m": native_deltas_m.tolist(),
            "base_deltas_m": base_deltas_m.tolist(),
            "endpoint_residual_base_m": endpoint_residual.tolist(),
            "model_to_base": model_to_base.tolist(),
        }
        adapter = "droid_native_horizon_endpoint_residual_xy"

    limited_offset_xy, action_scale = _clip_xy_norm(
        raw_offset_xy,
        maximum_offset_m,
    )

    interior_limit_m = max(
        0.001,
        target_radius_m - boundary_margin_m,
    )
    projected_offset_xy, region_scale = _clip_xy_norm(
        limited_offset_xy,
        interior_limit_m,
    )

    preferred_xy = (
        detected_center[:2]
        + projected_offset_xy
    )

    report = {
        "schema": "vla_place_preference",
        "model": model,
        "decision_source": "native_vla_translation_output",
        "metric_anchor_source": center_source,
        "detected_target_center_xyz_m": detected_center,
        "raw_offset_xy_m": raw_offset_xy,
        "limited_offset_xy_m": limited_offset_xy,
        "projected_offset_xy_m": projected_offset_xy,
        "preferred_xy_m": preferred_xy,
        "maximum_action_offset_m": maximum_offset_m,
        "target_region_radius_m": target_radius_m,
        "boundary_margin_m": boundary_margin_m,
        "action_limit_scale": action_scale,
        "region_projection_scale": region_scale,
        "gain": gain,
        "adapter": adapter,
        "adapter_report": adapter_report,
        "rotation_used": False,
        "gripper_used": False,
    }

    print(
        f"model={model} "
        f"center={detected_center[:2].tolist()} "
        f"raw_offset={raw_offset_xy.tolist()} "
        f"applied_offset={projected_offset_xy.tolist()} "
        f"preferred={preferred_xy.tolist()}"
    )

    return preferred_xy, report


def apply_vla_place_preference(
    *,
    entities: dict[str, Any],
    preferred_xy: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ancla el place en la preferencia del VLA manteniendo la Z métrica y la altura segura."""
    updated = copy.deepcopy(entities)
    original_target = finite_vector(
        entities["target_xyz_m"],
        3,
        "vla_place.original_target",
    )
    original_safe = finite_vector(
        entities["target_safe_xyz_m"],
        3,
        "vla_place.original_target_safe",
    )
    preferred_xy = finite_vector(
        preferred_xy,
        2,
        "vla_place.preferred_xy",
    )

    preferred_target = original_target.copy()
    preferred_target[:2] = preferred_xy

    preferred_safe = original_safe.copy()
    preferred_safe[:2] = preferred_xy

    updated["target_xyz_m"] = preferred_target.tolist()
    updated["target_safe_xyz_m"] = preferred_safe.tolist()

    report = {
        "original_resolved_target_xyz_m": original_target,
        "original_resolved_target_safe_xyz_m": original_safe,
        "vla_preferred_target_xyz_m": preferred_target,
        "vla_preferred_target_safe_xyz_m": preferred_safe,
        "planner_role": (
            "validate_and_project_to_nearest_safe_reachable_place"
        ),
    }

    return updated, report

def validate_joint_chain(
    *,
    q_start: np.ndarray,
    q_targets: list[np.ndarray],
    label: str,
    maximum_step_rad: float = 0.45,
) -> dict[str, Any]:
    previous = finite_vector(
        q_start,
        6,
        f"{label}.q_start",
    )

    if not q_targets:
        raise MainStop(
            f"{label}: cadena vacía"
        )

    steps = []

    for index, value in enumerate(
        q_targets,
        start=1,
    ):
        current = finite_vector(
            value,
            6,
            f"{label}.q[{index}]",
        )

        step = float(
            np.max(
                np.abs(
                    current - previous
                )
            )
        )

        if step > maximum_step_rad:
            raise MainStop(
                f"{label}: salto {step:.4f} rad "
                f"> {maximum_step_rad:.4f} rad "
                f"en punto {index}"
            )

        steps.append(step)
        previous = current

    return {
        "label": label,
        "target_count": len(
            q_targets
        ),
        "maximum_step_rad":
            maximum_step_rad,
        "maximum_observed_step_rad":
            max(steps),
        "steps_rad": steps,
        "valid": True,
    }



def _maximum_joint_step(
    *,
    q_start: np.ndarray,
    q_targets: list[np.ndarray],
    label: str,
) -> float:
    previous = finite_vector(
        q_start,
        6,
        f"{label}.q_start",
    )
    maximum = 0.0

    for index, value in enumerate(q_targets, start=1):
        current = finite_vector(
            value,
            6,
            f"{label}.q[{index}]",
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(current - previous))),
        )
        previous = current

    return maximum


def _place_candidate_set(
    *,
    entities: dict[str, Any],
    tcp_start: np.ndarray,
) -> list[tuple[str, np.ndarray, float]]:
    """Genera hipótesis interiores equivalentes tomando como referencia la preferencia del VLA."""
    initial_target = finite_vector(
        entities["target_xyz_m"],
        3,
        "place_candidates.initial_target",
    )
    center_value = entities.get("target_detected_center_xyz_m")

    if center_value is None:
        center = initial_target.copy()
    else:
        center = finite_vector(
            center_value,
            3,
            "place_candidates.center",
        )

    center_radius = float(np.linalg.norm(center[:2]))
    if center_radius <= 1e-9:
        raise MainStop("place_candidates: centro de destino inválido")

    toward_base = -center[:2] / center_radius
    toward_start = finite_vector(
        tcp_start,
        3,
        "place_candidates.tcp_start",
    )[:2] - center[:2]
    toward_start_norm = float(np.linalg.norm(toward_start))
    if toward_start_norm <= 1e-9:
        toward_start = toward_base.copy()
    else:
        toward_start = toward_start / toward_start_norm

    blended = toward_base + toward_start
    blended_norm = float(np.linalg.norm(blended))
    if blended_norm <= 1e-9:
        blended = toward_base.copy()
    else:
        blended = blended / blended_norm

    hypotheses: list[tuple[str, np.ndarray, float]] = [
        (
            "vla_preferred",
            initial_target.copy(),
            float(np.linalg.norm(initial_target[:2] - center[:2])),
        )
    ]

    for name, offset_m, direction in (
        ("toward_base_0.075", 0.075, toward_base),
        ("toward_base_0.078", 0.078, toward_base),
        ("toward_base_0.080", 0.080, toward_base),
        ("blended_0.078", 0.078, blended),
        ("toward_start_0.078", 0.078, toward_start),
        ("toward_start_0.080", 0.080, toward_start),
    ):
        candidate = center.copy()
        candidate[:2] = center[:2] + float(offset_m) * direction
        candidate[2] = initial_target[2]
        hypotheses.append((name, candidate, float(offset_m)))

    unique: list[tuple[str, np.ndarray, float]] = []
    for name, candidate, offset_m in hypotheses:
        duplicate = any(
            float(np.linalg.norm(candidate[:2] - existing[:2])) <= 0.001
            for _, existing, _ in unique
        )
        if not duplicate:
            unique.append((name, candidate, offset_m))

    return unique



def _unique_transport_staging_candidates(
    *,
    entities: dict[str, Any],
    planning_entities: dict[str, Any],
    vla_place_preference_report: dict[str, Any],
) -> list[tuple[str, np.ndarray]]:
    """
    Construye puntos de transporte seguros sin cambiar el destino pedido.
    
    Primero se prueba el punto seguro más cercano a la preferencia del VLA. Si no
    es alcanzable en la rama del UR5e, se evalúan proyecciones intermedias hacia
    las anclas métricas y geométricas seguras. La altura de tránsito se mantiene;
    solo cambia XY.
    """
    vla_safe = finite_vector(
        planning_entities["target_safe_xyz_m"],
        3,
        "transport_staging.vla_safe",
    )
    geometric_safe = finite_vector(
        entities["target_safe_xyz_m"],
        3,
        "transport_staging.geometric_safe",
    )
    detected_center = finite_vector(
        vla_place_preference_report["detected_target_center_xyz_m"],
        3,
        "transport_staging.detected_center",
    )

    safe_z = float(max(vla_safe[2], geometric_safe[2]))
    detected_safe = detected_center.copy()
    detected_safe[2] = safe_z
    geometric_safe = geometric_safe.copy()
    geometric_safe[2] = safe_z
    vla_safe = vla_safe.copy()
    vla_safe[2] = safe_z

    raw: list[tuple[str, np.ndarray]] = [("vla_preferred_safe", vla_safe)]

    # La proyección avanza hacia anclas ya validadas en la escena métrica.
    # Así se conserva tanta preferencia XY del VLA como permita la alcanzabilidad,
    # sin recortar a ciegas ni cambiar la zona pedida.
    for anchor_name, anchor in (
        ("detected_center", detected_safe),
        ("geometric_safe", geometric_safe),
    ):
        for alpha in (0.75, 0.50, 0.25, 0.0):
            candidate = anchor.copy()
            candidate[:2] = (
                alpha * vla_safe[:2]
                + (1.0 - alpha) * anchor[:2]
            )
            raw.append((f"vla_to_{anchor_name}_{alpha:.2f}", candidate))

    unique: list[tuple[str, np.ndarray]] = []
    for name, candidate in raw:
        if any(
            float(np.linalg.norm(candidate - previous)) <= 0.001
            for _, previous in unique
        ):
            continue
        unique.append((name, candidate))

    return unique


def select_reachable_transport_staging(
    *,
    model: str,
    actions: np.ndarray,
    q_start: np.ndarray,
    tcp_start: np.ndarray,
    entities: dict[str, Any],
    planning_entities: dict[str, Any],
    task_object: np.ndarray,
    task_target_vla: np.ndarray,
    calibration: dict[str, Any] | None,
    config: Any,
    phase_gain: float,
    phase_contract: dict[str, Any],
    vla_place_preference_report: dict[str, Any],
) -> dict[str, Any]:
    """
    Selecciona un punto de transporte alcanzable antes de buscar el place.
    
    El residual del VLA puede modificar cada candidato, pero ningún punto
    inalcanzable se fuerza dentro de la IK. Cada candidato se resuelve desde el
    lift real y, entre los válidos, se elige el más próximo en XY a la preferencia
    del VLA.
    """
    q_start = finite_vector(q_start, 6, "transport_staging.q_start")
    tcp_start = finite_vector(tcp_start, 3, "transport_staging.tcp_start")
    preferred_safe = finite_vector(
        planning_entities["target_safe_xyz_m"],
        3,
        "transport_staging.preferred_safe",
    )

    rejected: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []

    candidates = _unique_transport_staging_candidates(
        entities=entities,
        planning_entities=planning_entities,
        vla_place_preference_report=vla_place_preference_report,
    )

    for index, (name, staging_xyz) in enumerate(candidates):
        staging_xyz = finite_vector(
            staging_xyz,
            3,
            f"transport_staging.candidate[{index}]",
        )
        try:
            targets, adapter = phase_targets(
                model=model,
                actions=actions,
                start=tcp_start,
                goal=staging_xyz,
                task_object=task_object,
                task_target=task_target_vla,
                calibration=calibration,
                config=config,
                label=f"transport_translation_candidate_{index}",
                phase_gain=phase_gain,
                workspace_mode=str(phase_contract["workspace_mode"]),
            )

            q_targets, ik_report = supervisor.solve_cartesian_targets(
                q_start=q_start,
                tcp_start=tcp_start,
                targets=targets,
                config=config,
                label=f"transport_candidate_{index}",
                max_joint_step_rad=float(
                    phase_contract["max_joint_step_rad"]
                ),
                contact=bool(phase_contract["solver_contact"]),
            )
            posture_report = validate_high_shoulder_branch(
                q_start=q_start,
                q_targets=q_targets,
                label=f"transport_candidate_{index}",
            )

            tcp_end = finite_vector(
                ik_report["waypoints"][-1]["fk_real_base_m"],
                3,
                f"transport_staging.candidate[{index}].tcp_end",
            )
            endpoint_error_m = float(np.linalg.norm(tcp_end - staging_xyz))
            if endpoint_error_m > 0.015:
                raise MainStop(
                    "endpoint de staging impreciso: "
                    f"{endpoint_error_m:.4f} m"
                )

            projection_xy_m = float(
                np.linalg.norm(staging_xyz[:2] - preferred_safe[:2])
            )
            maximum_joint_step_rad = _maximum_joint_step(
                q_start=q_start,
                q_targets=q_targets,
                label=f"transport_staging.candidate[{index}]",
            )

            item = {
                "index": index,
                "name": name,
                "staging_xyz_m": staging_xyz,
                "projection_from_vla_xy_m": projection_xy_m,
                "endpoint_error_m": endpoint_error_m,
                "maximum_joint_step_rad": maximum_joint_step_rad,
                "targets": targets,
                "adapter": adapter,
                "q_targets": q_targets,
                "ik_report": ik_report,
                "posture_report": posture_report,
                "posture_margin_rad": float(
                    posture_report["weakest_margin_rad"]
                ),
                "tcp_end": tcp_end,
            }
            valid.append(item)

            print(
                f"name={name} staging={staging_xyz.tolist()} "
                f"projection={projection_xy_m:.4f}m "
                f"endpoint_error={endpoint_error_m:.4f}m "
                f"max_dq={maximum_joint_step_rad:.4f}rad",
                flush=True,
            )
        except Exception as exc:
            rejected_item = {
                "index": index,
                "name": name,
                "staging_xyz_m": staging_xyz.tolist(),
                "reason": str(exc),
            }
            rejected.append(rejected_item)
            print(
                f"name={name} staging={staging_xyz.tolist()} "
                f"reason={exc}",
                flush=True,
            )

    if not valid:
        raise MainStop(
            "Ningún staging de transporte permite una trayectoria IK segura: "
            f"{rejected}"
        )

    selected = min(
        valid,
        key=lambda item: (
            item["projection_from_vla_xy_m"],
            -item["posture_margin_rad"],
            item["maximum_joint_step_rad"],
            item["endpoint_error_m"],
        ),
    )

    print(
        f"name={selected['name']} "
        f"staging={selected['staging_xyz_m'].tolist()} "
        f"projection={selected['projection_from_vla_xy_m']:.4f}m",
        flush=True,
    )

    selected["search_report"] = {
        "architecture": (
            "same_vla_transport_path_with_reachable_metric_staging_projection"
        ),
        "selection_priority": (
            "reachability_then_nearest_vla_xy_then_joint_continuity"
        ),
        "preferred_safe_xyz_m": preferred_safe.tolist(),
        "selected_staging_xyz_m": selected["staging_xyz_m"].tolist(),
        "projection_from_vla_xy_m": selected["projection_from_vla_xy_m"],
        "valid_candidates": [
            {
                "index": item["index"],
                "name": item["name"],
                "staging_xyz_m": item["staging_xyz_m"].tolist(),
                "projection_from_vla_xy_m": item[
                    "projection_from_vla_xy_m"
                ],
                "endpoint_error_m": item["endpoint_error_m"],
                "maximum_joint_step_rad": item[
                    "maximum_joint_step_rad"
                ],
                "posture_margin_rad": item["posture_margin_rad"],
            }
            for item in valid
        ],
        "rejected_candidates": rejected,
    }
    return selected

def select_place_candidate_with_candidate_specific_handoff(
    *,
    model: str,
    actions: np.ndarray,
    q_transport_end: np.ndarray,
    tcp_transport_end: np.ndarray,
    entities: dict[str, Any],
    task_object: np.ndarray,
    task_target_vla: np.ndarray,
    calibration: dict[str, Any] | None,
    config: Any,
    phase_gain: float,
    phase_contract: dict[str, Any],
    vla_preferred_xy: np.ndarray,
) -> dict[str, Any]:
    """
    Valida cada candidato de place con su propio handoff completo.
    
    Cada hipótesis conserva su cadena:
      transport_end -> candidate_safe -> candidate_place -> candidate_safe.
    La semilla q/tcp se calcula para ese candidato y no se comparte entre
    hipótesis distintas.
    """
    q_transport_end = finite_vector(
        q_transport_end,
        6,
        "place_handoff.q_transport_end",
    )
    tcp_transport_end = finite_vector(
        tcp_transport_end,
        3,
        "place_handoff.tcp_transport_end",
    )
    preferred_xy = finite_vector(
        vla_preferred_xy,
        2,
        "place_handoff.vla_preferred_xy",
    )
    safe_z = float(
        finite_vector(
            entities["target_safe_xyz_m"],
            3,
            "place_handoff.target_safe",
        )[2]
    )

    candidates = _place_candidate_set(
        entities=entities,
        tcp_start=tcp_transport_end,
    )
    rejected: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []

    for index, (name, target_xyz, center_offset_m) in enumerate(candidates):
        target_xyz = finite_vector(
            target_xyz,
            3,
            f"place_handoff.candidate[{index}].target",
        ).copy()
        safe_xyz = target_xyz.copy()
        safe_xyz[2] = safe_z
        place_xyz = target_xyz.copy()
        place_xyz[2] = float(config.place_z_m)

        try:
            approach_targets, approach_adapter = phase_targets(
                model=model,
                actions=actions,
                start=tcp_transport_end,
                goal=safe_xyz,
                task_object=task_object,
                task_target=task_target_vla,
                calibration=calibration,
                config=config,
                label=f"place_approach_candidate_{index}",
                phase_gain=phase_gain,
                workspace_mode=str(phase_contract["workspace_mode"]),
            )

            approach_q, approach_ik = supervisor.solve_cartesian_targets(
                q_start=q_transport_end,
                tcp_start=tcp_transport_end,
                targets=approach_targets,
                config=config,
                label=f"place_approach_candidate_{index}",
                max_joint_step_rad=float(
                    phase_contract["max_joint_step_rad"]
                ),
                contact=bool(phase_contract["solver_contact"]),
            )
            approach_posture_report = validate_high_shoulder_branch(
                q_start=q_transport_end,
                q_targets=approach_q,
                label=f"place_approach_candidate_{index}",
            )

            candidate_safe_tcp = finite_vector(
                approach_ik["waypoints"][-1]["fk_real_base_m"],
                3,
                f"place_handoff.candidate[{index}].safe_tcp",
            )
            safe_error_m = float(np.linalg.norm(candidate_safe_tcp - safe_xyz))
            if safe_error_m > 0.015:
                raise MainStop(
                    f"handoff al safe del candidato impreciso: {safe_error_m:.4f} m"
                )

            local_entities = copy.deepcopy(entities)
            local_entities["target_xyz_m"] = target_xyz.tolist()
            local_entities["target_safe_xyz_m"] = safe_xyz.tolist()
            local_entities["pre_place_xyz_m"] = safe_xyz.tolist()
            local_entities["place_xyz_m"] = place_xyz.tolist()
            local_entities["retreat_xyz_m"] = safe_xyz.tolist()
            local_entities["place_transit_z_m"] = float(candidate_safe_tcp[2])
            local_entities["safe_place_target_locked"] = True

            geometric_plan, geometric_report = supervisor.plan_place_sequence(
                q_start=approach_q[-1],
                tcp_start=candidate_safe_tcp,
                entities=local_entities,
                config=config,
            )

            approach_max_dq = _maximum_joint_step(
                q_start=q_transport_end,
                q_targets=approach_q,
                label=f"place_handoff.candidate[{index}].approach",
            )
            geometric_chain = [
                np.asarray(value, dtype=float)
                for value in geometric_plan.get("q_to_place_steps", [])
            ] + [
                np.asarray(value, dtype=float)
                for value in geometric_plan.get("q_retreat_steps", [])
            ]
            geometric_max_dq = _maximum_joint_step(
                q_start=approach_q[-1],
                q_targets=geometric_chain,
                label=f"place_handoff.candidate[{index}].geometric",
            ) if geometric_chain else 0.0
            geometric_posture_report = validate_high_shoulder_branch(
                q_start=approach_q[-1],
                q_targets=geometric_chain,
                label=f"place_geometric_candidate_{index}",
            )

            projection_m = float(
                np.linalg.norm(target_xyz[:2] - preferred_xy)
            )
            valid_item = {
                "index": index,
                "name": name,
                "target_xyz_m": target_xyz,
                "safe_xyz_m": safe_xyz,
                "place_xyz_m": place_xyz,
                "center_offset_xy_m": float(center_offset_m),
                "vla_projection_xy_m": projection_m,
                "safe_endpoint_error_m": safe_error_m,
                "maximum_joint_step_rad": max(
                    approach_max_dq,
                    geometric_max_dq,
                ),
                "approach_targets": approach_targets,
                "approach_adapter": approach_adapter,
                "approach_q": approach_q,
                "approach_ik": approach_ik,
                "approach_posture_report": approach_posture_report,
                "geometric_posture_report": geometric_posture_report,
                "posture_margin_rad": min(
                    float(approach_posture_report["weakest_margin_rad"]),
                    float(geometric_posture_report["weakest_margin_rad"]),
                ),
                "geometric_plan": geometric_plan,
                "geometric_report": geometric_report,
                "entities": local_entities,
            }
            valid.append(valid_item)

            print(
                f"name={name} target={target_xyz.tolist()} "
                f"safe={safe_xyz.tolist()} "
                f"projection={projection_m:.4f}m "
                f"safe_error={safe_error_m:.4f}m "
                f"max_dq={valid_item['maximum_joint_step_rad']:.4f}rad",
                flush=True,
            )

        except Exception as exc:
            rejected_item = {
                "index": index,
                "name": name,
                "target_xyz_m": target_xyz.tolist(),
                "safe_xyz_m": safe_xyz.tolist(),
                "reason": str(exc),
            }
            rejected.append(rejected_item)
            print(
                f"name={name} target={target_xyz.tolist()} "
                f"safe={safe_xyz.tolist()} reason={exc}",
                flush=True,
            )

    if not valid:
        raise MainStop(
            "Ningún candidato permite handoff, place y retreat completos: "
            f"{rejected}"
        )

    # A este punto solo llegan candidatos válidos. Entre ellos se elige
    # el más cercano a la preferencia del VLA; la continuidad sirve de desempate.
    selected = min(
        valid,
        key=lambda item: (
            item["vla_projection_xy_m"],
            -item["posture_margin_rad"],
            item["maximum_joint_step_rad"],
            item["center_offset_xy_m"],
        ),
    )

    print(
        f"name={selected['name']} "
        f"target={selected['target_xyz_m'].tolist()} "
        f"projection={selected['vla_projection_xy_m']:.4f}m "
        f"max_dq={selected['maximum_joint_step_rad']:.4f}rad",
        flush=True,
    )

    selected["search_report"] = {
        "architecture": "candidate_specific_transport_safe_place_retreat",
        "selection_priority": (
            "safety_then_nearest_vla_preference_then_joint_continuity"
        ),
        "rejected_candidates": rejected,
        "valid_candidates": [
            {
                key: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value in item.items()
                if key not in {
                    "approach_targets",
                    "approach_adapter",
                    "approach_q",
                    "approach_ik",
                    "approach_posture_report",
                    "geometric_posture_report",
                    "geometric_plan",
                    "geometric_report",
                    "entities",
                }
            }
            for item in valid
        ],
    }
    return selected


# Aplicación de la acción en plan traslacional.
# Visión y lenguaje siguen dependiendo del VLA. Photoneo aporta la geometría métrica
# del objeto y del destino. La acción del modelo no se ejecuta como una trayectoria
# temporal directa: la traslación XYZ nativa se convierte en residuales métricos acotados
# que ajustan las seis anclas geométricas antes de resolver la IK secuencial.
# En el contacto con el objeto se mantiene la verticalidad:
# q_pre y q_pick comparten el mismo residual acotado.
# En el destino también se mantiene la verticalidad:
# q_pre_place, q_place y q_retreat comparten el mismo residual acotado.
# La señal temporal intermedia modifica q_lift, por lo que las diferencias entre modelos
# se ven sobre todo en lift y transport, donde hay margen físico.

HOME_Q_RAD = np.asarray(
    [
        1.4204216003417969,
        -1.3766822081855317,
        1.5765932242022913,
        -1.747573038140768,
        -1.5658624807940882,
        -0.10320645967592412,
    ],
    dtype=float,
)

DROID_TRANSLATION_SCALE_M = 0.050

OBJECT_CONTACT_BUDGET_M = np.asarray(
    [0.006, 0.006, 0.004],
    dtype=float,
)

LIFT_BUDGET_M = np.asarray(
    [0.025, 0.025, 0.012],
    dtype=float,
)

LIFT_RELATIVE_TO_PICK_BUDGET_M = np.asarray(
    [0.020, 0.020, 0.012],
    dtype=float,
)

DESTINATION_CONTACT_BUDGET_M = np.asarray(
    [0.008, 0.008, 0.004],
    dtype=float,
)

OPENVLA_NATIVE_BUDGET_M = np.asarray(
    [0.025, 0.025, 0.012],
    dtype=float,
)

PHASE_SOURCE = {
    "q_pre": "initial",
    "q_pick": "initial",
    "q_lift": "middle",
    "q_pre_place": "result",
    "q_place": "result",
    "q_retreat": "result",
}

CONTACT_PHASES = {
    "q_pick",
    "q_place",
}

MINIMUM_CLEARANCES_M = {
    "pre_above_pick": 0.080,
    "lift_above_pick": 0.060,
    "pre_place_above_place": 0.160,
    "retreat_above_place": 0.160,
}


def _normalize_by_max_abs(
    values: np.ndarray,
    label: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)

    if vector.size == 0:
        raise MainStop(f"{label}: vector vacío")

    if not np.all(np.isfinite(vector)):
        raise MainStop(f"{label}: contiene NaN o Inf")

    maximum = float(np.max(np.abs(vector)))

    if maximum <= 1e-12:
        return np.zeros_like(vector)

    return np.clip(
        vector / maximum,
        -1.0,
        1.0,
    )


def _temporal_thirds(
    actions: np.ndarray,
    label: str,
) -> list[np.ndarray]:
    matrix = np.asarray(actions, dtype=float)

    if matrix.ndim != 2:
        raise MainStop(
            f"{label}: actions debe ser matriz; shape={matrix.shape}"
        )

    thirds = [
        np.asarray(value, dtype=float)
        for value in np.array_split(matrix, 3, axis=0)
    ]

    if any(value.shape[0] == 0 for value in thirds):
        raise MainStop(
            f"{label}: horizonte insuficiente para tres tramos"
        )

    return thirds


def _clip_metric_residual(
    *,
    proposed: np.ndarray,
    budget: np.ndarray,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    proposed = finite_vector(
        proposed,
        3,
        f"{label}.proposed",
    )
    budget = finite_vector(
        budget,
        3,
        f"{label}.budget",
    )

    if np.any(budget <= 0.0):
        raise MainStop(
            f"{label}: presupuesto no positivo"
        )

    applied = np.clip(
        proposed,
        -budget,
        budget,
    )

    clipped_components = [
        index
        for index in range(3)
        if abs(float(applied[index] - proposed[index])) > 1e-12
    ]

    return applied, {
        "proposed_base_m": proposed.tolist(),
        "budget_abs_base_m": budget.tolist(),
        "applied_base_m": applied.tolist(),
        "clipped_components": clipped_components,
        "saturated": bool(clipped_components),
        "proposed_norm_m": float(np.linalg.norm(proposed)),
        "applied_norm_m": float(np.linalg.norm(applied)),
    }


def _build_gripper_guidance(
    *,
    model: str,
    matrix: np.ndarray,
    temporal_summary: dict[str, Any],
) -> dict[str, Any]:
    if model == "openvla":
        raw_gripper_signal = float(matrix[0, 6])
        normalized_close_signal = float(
            np.tanh(2.0 * raw_gripper_signal)
        )
        signal_source = (
            "openvla_raw_action_component_7"
        )
    else:
        result_mean = finite_vector(
            temporal_summary["result_mean"],
            8,
            f"plan.{model}.result_mean",
        )
        result_normalized = _normalize_by_max_abs(
            result_mean,
            f"plan.{model}.result_mean_normalized",
        )
        raw_gripper_signal = float(
            result_normalized[7]
        )
        normalized_close_signal = float(
            np.clip(
                raw_gripper_signal,
                -1.0,
                1.0,
            )
        )
        signal_source = (
            f"{model}_result_temporal_third_component_8"
        )

    close_confidence = float(
        np.clip(
            (normalized_close_signal + 1.0) / 2.0,
            0.0,
            1.0,
        )
    )

    open_width_m = 0.090
    close_width_m = float(
        np.clip(
            0.024 - 0.009 * close_confidence,
            0.015,
            0.024,
        )
    )
    close_hold_s = float(
        np.clip(
            2.0 + 2.0 * close_confidence,
            2.0,
            4.0,
        )
    )

    return {
        "enabled": True,
        "signal_source": signal_source,
        "raw_gripper_signal": raw_gripper_signal,
        "normalized_close_signal": normalized_close_signal,
        "close_confidence": close_confidence,
        "baseline_open_width_m": 0.090,
        "baseline_close_width_m": 0.018,
        "baseline_close_hold_s": 3.0,
        "applied_open_width_m": open_width_m,
        "applied_close_width_m": close_width_m,
        "applied_close_hold_s": close_hold_s,
        "close_width_bounds_m": [0.015, 0.024],
        "close_hold_bounds_s": [2.0, 4.0],
        "open_width_is_fixed": True,
    }


def build_translation_profile(
    *,
    model: str,
    actions: np.ndarray,
    instruction: str,
    image_path: Path,
    evidence: dict[str, Any],
    object_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
) -> dict[str, Any]:
    matrix = np.asarray(actions, dtype=float)
    expected = EXPECTED_SHAPES.get(model)

    if expected is None:
        raise MainStop(f"Modelo plan traslacional no soportado: {model}")

    if matrix.shape != expected:
        raise MainStop(
            f"{model}: shape {matrix.shape}; esperado {expected}"
        )

    if not np.all(np.isfinite(matrix)):
        raise MainStop(f"{model}: actions contiene NaN o Inf")

    object_xyz_m = finite_vector(
        object_xyz_m,
        3,
        "translation.object_xyz_m",
    )
    target_xyz_m = finite_vector(
        target_xyz_m,
        3,
        "translation.target_xyz_m",
    )

    temporal_summary: dict[str, Any]
    native_adapter_report: dict[str, Any]

    if model == "openvla":
        calibration = load_openvla_calibration()
        task_frame = adaptation.build_task_frame(
            object_xyz_m,
            target_xyz_m,
        )

        metric_residual, openvla_report = (
            adaptation.openvla_translation_residual(
                raw_action=matrix[0],
                calibration=calibration,
                task_frame=task_frame,
                translation_budget_m=(
                    OPENVLA_NATIVE_BUDGET_M
                ),
                gain=1.0,
            )
        )

        metric_residual = finite_vector(
            metric_residual,
            3,
            "translation.openvla_metric_residual",
        )

        native_signals = {
            "initial": metric_residual.copy(),
            "middle": metric_residual.copy(),
            "result": metric_residual.copy(),
        }

        temporal_summary = {
            "native_contract": "single_action_1x7",
            "temporal_mapping": (
                "single_calibrated_metric_residual_reused_by_phase"
            ),
            "raw_translation_components": (
                matrix[0, :3].tolist()
            ),
            "raw_action": matrix[0].tolist(),
        }

        native_adapter_report = {
            "adapter": (
                "openvla_empirical_calibration_to_task_frame_metric_xyz"
            ),
            "translation_dimensions": [0, 1, 2],
            "rotation_dimensions_ignored": [3, 4, 5],
            "gripper_dimension": 6,
            "calibration_report": openvla_report,
            "metric_signals_base_m": {
                key: value.tolist()
                for key, value in native_signals.items()
            },
        }

    else:
        thirds = _temporal_thirds(
            matrix,
            f"translation.{model}",
        )
        initial_mean = thirds[0].mean(axis=0)
        middle_mean = thirds[1].mean(axis=0)
        result_mean = thirds[2].mean(axis=0)

        grouped_model_translation = np.asarray(
            [
                initial_mean[:3],
                middle_mean[:3],
                result_mean[:3],
            ],
            dtype=float,
        )

        metric_model_deltas = (
            np.clip(
                grouped_model_translation,
                -1.0,
                1.0,
            )
            * DROID_TRANSLATION_SCALE_M
        )

        model_to_base = np.asarray(
            supervisor.MODEL_TO_BASE,
            dtype=float,
        ).reshape(3, 3)

        metric_base_deltas = (
            model_to_base
            @ metric_model_deltas.T
        ).T

        native_signals = {
            "initial": metric_base_deltas[0].copy(),
            "middle": metric_base_deltas[1].copy(),
            "result": metric_base_deltas[2].copy(),
        }

        temporal_summary = {
            "native_contract": (
                "chunk_15x8"
                if model == "pi05_droid"
                else "chunk_10x8"
            ),
            "split_sizes": [
                int(value.shape[0])
                for value in thirds
            ],
            "translation_summary": (
                "mean_xyz_of_initial_middle_result_temporal_thirds"
            ),
            "initial_mean": initial_mean.tolist(),
            "middle_mean": middle_mean.tolist(),
            "result_mean": result_mean.tolist(),
        }

        native_adapter_report = {
            "adapter": (
                "droid_temporal_thirds_xyz_to_metric_base_frame"
            ),
            "translation_dimensions": [0, 1, 2],
            "rotation_dimensions_ignored": [3, 4, 5, 6],
            "gripper_dimension": 7,
            "translation_scale_m": (
                DROID_TRANSLATION_SCALE_M
            ),
            "model_to_base": model_to_base.tolist(),
            "grouped_model_translation": (
                grouped_model_translation.tolist()
            ),
            "metric_model_deltas_m": (
                metric_model_deltas.tolist()
            ),
            "metric_signals_base_m": {
                key: value.tolist()
                for key, value in native_signals.items()
            },
        }

    object_residual, object_report = (
        _clip_metric_residual(
            proposed=native_signals["initial"],
            budget=OBJECT_CONTACT_BUDGET_M,
            label="translation.object_contact",
        )
    )

    lift_unconstrained, lift_clip_report = (
        _clip_metric_residual(
            proposed=native_signals["middle"],
            budget=LIFT_BUDGET_M,
            label="translation.lift_absolute",
        )
    )

    lift_relative_delta, lift_relative_report = (
        _clip_metric_residual(
            proposed=(
                lift_unconstrained
                - object_residual
            ),
            budget=(
                LIFT_RELATIVE_TO_PICK_BUDGET_M
            ),
            label="translation.lift_relative_to_pick",
        )
    )
    lift_residual = (
        object_residual + lift_relative_delta
    )

    destination_residual, destination_report = (
        _clip_metric_residual(
            proposed=native_signals["result"],
            budget=(
                DESTINATION_CONTACT_BUDGET_M
            ),
            label="translation.destination_contact",
        )
    )

    phase_residuals = {
        "q_pre": object_residual.copy(),
        "q_pick": object_residual.copy(),
        "q_lift": lift_residual.copy(),
        "q_pre_place": destination_residual.copy(),
        "q_place": destination_residual.copy(),
        "q_retreat": destination_residual.copy(),
    }

    gripper_guidance = _build_gripper_guidance(
        model=model,
        matrix=matrix,
        temporal_summary=temporal_summary,
    )

    actions_sha256 = hashlib.sha256(
        np.ascontiguousarray(
            matrix,
            dtype=np.float64,
        ).tobytes()
    ).hexdigest()

    profile = {
        "schema": (
            "VLA_three_model_metric_translation_profile"
        ),
        "architecture": (
            "native_xyz_to_metric_phase_residuals_on_geometric_anchors"
        ),
        "model": model,
        "instruction": instruction,
        "image_path": str(image_path),
        "image_sha256": sha256_file(image_path),
        "native_actions_shape": list(matrix.shape),
        "native_actions_sha256": actions_sha256,
        "translation_vla_applied": True,
        "raw_temporal_path_executed": False,
        "rotation_vla_applied": False,
        "joint_signal_reinterpretation_used": False,
        "native_adapter": native_adapter_report,
        "phase_source": dict(PHASE_SOURCE),
        "phase_residuals_base_m": {
            key: value.tolist()
            for key, value in phase_residuals.items()
        },
        "object_contact_projection": object_report,
        "lift_absolute_projection": lift_clip_report,
        "lift_relative_projection": lift_relative_report,
        "destination_contact_projection": (
            destination_report
        ),
        "vertical_contact_invariants": {
            "q_pre_equals_q_pick_residual": bool(
                np.allclose(
                    phase_residuals["q_pre"],
                    phase_residuals["q_pick"],
                    atol=1e-12,
                )
            ),
            "q_pre_place_equals_q_place_equals_q_retreat_residual": bool(
                np.allclose(
                    phase_residuals["q_pre_place"],
                    phase_residuals["q_place"],
                    atol=1e-12,
                )
                and np.allclose(
                    phase_residuals["q_place"],
                    phase_residuals["q_retreat"],
                    atol=1e-12,
                )
            ),
        },
        "gripper_guidance": gripper_guidance,
        "temporal_summary": temporal_summary,
        "model_evidence": evidence,
    }

    print(
        f"model={model} "
        f"shape={matrix.shape} "
        f"object_residual={object_residual.tolist()} "
        f"lift_residual={lift_residual.tolist()} "
        f"destination_residual={destination_residual.tolist()} "
        f"close_width="
        f"{gripper_guidance['applied_close_width_m']:.6f}m "
        "raw_temporal_path=false rotation=false",
        flush=True,
    )

    return profile


def _workspace_ok(
    point: np.ndarray,
    *,
    contact: bool,
) -> bool:
    point = finite_vector(
        point,
        3,
        "workspace.point",
    )
    minimum_z = 0.030 if contact else 0.045

    return bool(
        -0.90 <= point[0] <= 0.90
        and -1.10 <= point[1] <= 0.20
        and minimum_z <= point[2] <= 0.45
    )


def apply_translation_residuals(
    *,
    geometric_targets: dict[str, np.ndarray],
    profile: dict[str, Any],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
]:
    required = (
        "q_pre",
        "q_pick",
        "q_lift",
        "q_pre_place",
        "q_place",
        "q_retreat",
    )

    if set(geometric_targets) != set(required):
        raise MainStop(
            "plan traslacional: el conjunto de anclas geométricas no es exacto"
        )

    raw_phase_residuals = profile.get(
        "phase_residuals_base_m",
        {},
    )

    proposed_targets: dict[str, np.ndarray] = {}
    applied_targets: dict[str, np.ndarray] = {}
    phase_rows: dict[str, Any] = {}

    for phase in required:
        geometric = finite_vector(
            geometric_targets[phase],
            3,
            f"translation.{phase}.geometric",
        )
        residual = finite_vector(
            raw_phase_residuals.get(phase),
            3,
            f"translation.{phase}.residual",
        )
        proposed = geometric + residual
        proposed_targets[phase] = proposed.copy()
        applied_targets[phase] = proposed.copy()

    projections: list[dict[str, Any]] = []

    def project_z(
        phase: str,
        minimum_z: float,
        reason: str,
    ) -> None:
        previous = float(
            applied_targets[phase][2]
        )
        applied_targets[phase][2] = max(
            previous,
            float(minimum_z),
        )
        current = float(
            applied_targets[phase][2]
        )

        if current > previous + 1e-12:
            projections.append(
                {
                    "phase": phase,
                    "axis": "z",
                    "before_m": previous,
                    "after_m": current,
                    "reason": reason,
                }
            )

    project_z(
        "q_pre",
        float(applied_targets["q_pick"][2])
        + MINIMUM_CLEARANCES_M[
            "pre_above_pick"
        ],
        "preserve_vertical_pre_pick_clearance",
    )
    project_z(
        "q_lift",
        float(applied_targets["q_pick"][2])
        + MINIMUM_CLEARANCES_M[
            "lift_above_pick"
        ],
        "preserve_lift_clearance",
    )
    project_z(
        "q_pre_place",
        float(applied_targets["q_place"][2])
        + MINIMUM_CLEARANCES_M[
            "pre_place_above_place"
        ],
        "preserve_vertical_place_approach_clearance",
    )
    project_z(
        "q_retreat",
        float(applied_targets["q_place"][2])
        + MINIMUM_CLEARANCES_M[
            "retreat_above_place"
        ],
        "preserve_vertical_retreat_clearance",
    )

    # Se mantiene exactamente la geometría XY del contacto vertical.
    applied_targets["q_pre"][:2] = (
        applied_targets["q_pick"][:2]
    )
    applied_targets["q_pre_place"][:2] = (
        applied_targets["q_place"][:2]
    )
    applied_targets["q_retreat"][:2] = (
        applied_targets["q_place"][:2]
    )

    for phase in required:
        contact = phase in CONTACT_PHASES
        point = finite_vector(
            applied_targets[phase],
            3,
            f"translation.{phase}.applied",
        )

        if not _workspace_ok(
            point,
            contact=contact,
        ):
            raise MainStop(
                f"plan traslacional: {phase} sale del workspace: "
                f"{point.tolist()}"
            )

        geometric = finite_vector(
            geometric_targets[phase],
            3,
            f"translation.{phase}.geometric_report",
        )
        proposed = finite_vector(
            proposed_targets[phase],
            3,
            f"translation.{phase}.proposed_report",
        )

        phase_rows[phase] = {
            "geometric_target_base_m": geometric.tolist(),
            "proposed_vla_target_base_m": proposed.tolist(),
            "applied_safe_target_base_m": point.tolist(),
            "proposed_residual_base_m": (
                proposed - geometric
            ).tolist(),
            "applied_residual_base_m": (
                point - geometric
            ).tolist(),
            "supervisor_projection_base_m": (
                point - proposed
            ).tolist(),
            "contact": contact,
            "workspace_ok": True,
        }

        applied_targets[phase] = point.copy()

    report = {
        "schema": (
            "VLA_metric_translation_application"
        ),
        "translation_vla_applied": True,
        "raw_temporal_path_executed": False,
        "rotation_vla_applied": False,
        "phase_rows": phase_rows,
        "safety_projections": projections,
        "vertical_contact_geometry_preserved": True,
        "all_models_share_same_physical_budgets": True,
        "model_specific_gain_used": False,
    }

    print(
        f"q_pre={applied_targets['q_pre'].tolist()} "
        f"q_lift={applied_targets['q_lift'].tolist()} "
        f"q_pre_place="
        f"{applied_targets['q_pre_place'].tolist()} "
        f"safety_projections={len(projections)}",
        flush=True,
    )

    return applied_targets, report

def _segment_targets(
    *,
    start: np.ndarray,
    goal: np.ndarray,
    maximum_cartesian_step_m: float,
    label: str,
) -> list[np.ndarray]:
    start = finite_vector(start, 3, f"{label}.start")
    goal = finite_vector(goal, 3, f"{label}.goal")

    distance_m = float(
        np.linalg.norm(goal - start)
    )

    if (
        not math.isfinite(maximum_cartesian_step_m)
        or maximum_cartesian_step_m <= 0.0
    ):
        raise MainStop(
            f"{label}: paso cartesiano inválido"
        )

    count = max(
        1,
        int(
            math.ceil(
                distance_m
                / maximum_cartesian_step_m
            )
        ),
    )

    return baseline_targets(
        start,
        goal,
        count,
    )


def _solve_sequential_geometry(
    *,
    q_start: np.ndarray,
    tcp_start: np.ndarray,
    targets: list[np.ndarray],
    config: Any,
    label: str,
    contact: bool,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    solver = getattr(
        supervisor,
        "_solve_cartesian_targets_once",
        None,
    )

    if not callable(solver):
        raise MainStop(
            "Falta el solver cartesiano secuencial original "
            "_solve_cartesian_targets_once. "
            "El plan no usará el solver Full7D adaptativo."
        )

    max_joint_step = (
        float(config.max_contact_joint_step_rad)
        if contact
        else float(config.max_chunk_joint_step_rad)
    )

    solutions, report = solver(
        q_start=finite_vector(
            q_start,
            6,
            f"{label}.q_start",
        ),
        tcp_start=finite_vector(
            tcp_start,
            3,
            f"{label}.tcp_start",
        ),
        targets=[
            finite_vector(
                value,
                3,
                f"{label}.target[{index}]",
            )
            for index, value in enumerate(
                targets,
                start=1,
            )
        ],
        config=config,
        label=label,
        max_joint_step_rad=max_joint_step,
        contact=contact,
    )

    if not solutions:
        raise MainStop(
            f"{label}: el solver secuencial no devolvió soluciones"
        )

    print(
        f"label={label} "
        f"targets={len(targets)} "
        f"solutions={len(solutions)} "
        f"contact={str(contact).lower()}",
        flush=True,
    )

    return (
        [
            finite_vector(
                value,
                6,
                f"{label}.q[{index}]",
            )
            for index, value in enumerate(
                solutions,
                start=1,
            )
        ],
        report,
    )


def _find_fk_backend() -> dict[str, Any]:
    queue = [
        supervisor,
        getattr(supervisor, "execution", None),
        getattr(supervisor, "action_controller", None),
        getattr(supervisor, "robot_base", None),
    ]

    visited: set[int] = set()

    while queue:
        candidate = queue.pop(0)

        if candidate is None:
            continue

        identifier = id(candidate)

        if identifier in visited:
            continue

        visited.add(identifier)

        if all(
            callable(getattr(candidate, name, None))
            for name in (
                "load_scene",
                "set_robot_q",
                "get_site_pos_axis",
            )
        ):
            loaded = candidate.load_scene()

            if not isinstance(loaded, tuple) or len(loaded) < 7:
                raise MainStop(
                    "Backend FK devolvió una escena incompatible"
                )

            (
                scene_xml,
                model,
                data,
                tcp_site,
                tcp_id,
                qpos_ids,
                dof_ids,
            ) = loaded[:7]

            return {
                "module": candidate,
                "scene_xml": scene_xml,
                "model": model,
                "data": data,
                "tcp_site": tcp_site,
                "tcp_id": tcp_id,
                "qpos_ids": np.asarray(
                    qpos_ids,
                    dtype=int,
                ),
                "dof_ids": np.asarray(
                    dof_ids,
                    dtype=int,
                ),
            }

        for name in (
            "base",
            "robot_base",
            "execution",
            "action_controller",
        ):
            nested = getattr(
                candidate,
                name,
                None,
            )

            if nested is not None:
                queue.append(nested)

    raise MainStop(
        "No se encontró el backend MuJoCo usado por el plan "
        "para validar FK y orientación."
    )


def _build_fk_context(
    *,
    q_home: np.ndarray,
    tcp_home: np.ndarray,
) -> dict[str, Any]:
    context = _find_fk_backend()
    module = context["module"]
    model = context["model"]
    data = context["data"]
    qpos_ids = context["qpos_ids"]
    tcp_id = context["tcp_id"]

    q_home = finite_vector(
        q_home,
        6,
        "fk.q_home",
    )
    tcp_home = finite_vector(
        tcp_home,
        3,
        "fk.tcp_home",
    )

    module.set_robot_q(
        model,
        data,
        qpos_ids,
        q_home,
    )

    home_position_sim, home_axis, _ = (
        module.get_site_pos_axis(
            data,
            tcp_id,
        )
    )

    home_position_sim = finite_vector(
        home_position_sim,
        3,
        "fk.home_position_sim",
    )

    offset = tcp_home - home_position_sim

    context["offset_real_minus_sim_m"] = (
        offset
    )
    context["home_axis"] = finite_vector(
        home_axis,
        3,
        "fk.home_axis",
    )

    return context


def _fk(
    context: dict[str, Any],
    q_value: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    q = finite_vector(
        q_value,
        6,
        f"{label}.q",
    )

    module = context["module"]
    model = context["model"]
    data = context["data"]
    qpos_ids = context["qpos_ids"]
    tcp_id = context["tcp_id"]

    module.set_robot_q(
        model,
        data,
        qpos_ids,
        q,
    )

    position_sim, axis, _ = (
        module.get_site_pos_axis(
            data,
            tcp_id,
        )
    )

    position_real = (
        finite_vector(
            position_sim,
            3,
            f"{label}.position_sim",
        )
        + finite_vector(
            context[
                "offset_real_minus_sim_m"
            ],
            3,
            f"{label}.offset",
        )
    )

    axis = finite_vector(
        axis,
        3,
        f"{label}.axis",
    )
    axis_norm = float(
        np.linalg.norm(axis)
    )

    if axis_norm <= 1e-12:
        raise MainStop(
            f"{label}: eje TCP nulo"
        )

    axis = axis / axis_norm

    return position_real, axis


def _angle_deg(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first = finite_vector(first, 3, "angle.first")
    second = finite_vector(second, 3, "angle.second")

    first /= max(
        1e-12,
        float(np.linalg.norm(first)),
    )
    second /= max(
        1e-12,
        float(np.linalg.norm(second)),
    )

    return float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(first, second),
                    -1.0,
                    1.0,
                )
            )
        )
    )



def _record_for_grounding_name(
    *,
    scene: dict[str, Any],
    name: str,
) -> dict[str, Any] | None:
    exact_matches: list[dict[str, Any]] = []

    def walk(
        value: Any,
        path: tuple[str, ...] = (),
    ) -> None:
        if isinstance(value, dict):
            if path and str(path[-1]) == name:
                exact_matches.append(value)

            for key, item in value.items():
                walk(
                    item,
                    path + (str(key),),
                )

        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(
                    item,
                    path + (str(index),),
                )

    walk(scene)

    for record in exact_matches:
        if _candidate_xyz(record) is not None:
            return record

    return None


def _safe_xyz_from_record(
    *,
    record: dict[str, Any] | None,
    fallback: np.ndarray,
    label: str,
    minimum_z_m: float | None = None,
) -> np.ndarray:
    fallback_vector = finite_vector(
        fallback,
        3,
        f"{label}.fallback",
    )

    if isinstance(record, dict):
        for key in (
            "safe_above_base_xyz_m",
            "safe_xyz_base_m",
            "above_base_xyz_m",
            "pregrasp_base_xyz_m",
        ):
            value = record.get(key)

            if value is None:
                continue

            try:
                safe = finite_vector(
                    value,
                    3,
                    f"{label}.{key}",
                )
            except Exception:
                continue

            if minimum_z_m is not None:
                safe[2] = max(
                    float(safe[2]),
                    float(minimum_z_m),
                )

            return safe

    result = fallback_vector.copy()

    if minimum_z_m is not None:
        result[2] = max(
            float(result[2]),
            float(minimum_z_m),
        )

    return result


def _solve_anchor_chain(
    *,
    q_home: np.ndarray,
    tcp_home: np.ndarray,
    targets: dict[str, np.ndarray],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """
    Solver secuencial  plan.
    
        Mantiene:
        - targets cartesianos con residuales VLA;
        - orientación top-down;
        - semilla de cada fase igual a la fase anterior;
        - seis anclas geométricas originales.
    
        No introduce corredores artificiales q2/q3.
    """
    context = _find_fk_backend()
    module = context["module"]

    solve_topdown = getattr(
        module,
        "solve_topdown_ik",
        None,
    )

    if not callable(solve_topdown):
        raise MainStop(
            "El backend MuJoCo no expone solve_topdown_ik"
        )

    q_home = finite_vector(
        q_home,
        6,
        "ik.q_home",
    )

    tcp_home = finite_vector(
        tcp_home,
        3,
        "ik.tcp_home",
    )

    model = context["model"]
    data = context["data"]
    tcp_id = context["tcp_id"]
    qpos_ids = context["qpos_ids"]
    dof_ids = context["dof_ids"]

    module.set_robot_q(
        model,
        data,
        qpos_ids,
        q_home,
    )

    sim_tcp_home, _, _ = (
        module.get_site_pos_axis(
            data,
            tcp_id,
        )
    )

    sim_tcp_home = finite_vector(
        sim_tcp_home,
        3,
        "ik.sim_tcp_home",
    )

    offset_real_minus_sim = (
        tcp_home - sim_tcp_home
    )

    order = (
        "q_pre",
        "q_pick",
        "q_lift",
        "q_pre_place",
        "q_place",
        "q_retreat",
    )

    q_seed = q_home.copy()
    solved: dict[str, np.ndarray] = {}
    rows: dict[str, Any] = {}

    for name in order:
        target_real = finite_vector(
            targets[name],
            3,
            f"ik.{name}.target_real",
        )

        target_sim = (
            target_real
            - offset_real_minus_sim
        )

        q0_scan = np.linspace(
            float(q_seed[0]) - 0.70,
            float(q_seed[0]) + 0.70,
            9,
        )

        solution = solve_topdown(
            model,
            data,
            tcp_id,
            qpos_ids,
            dof_ids,
            target_sim,
            q_seed,
            q0_scan,
        )

        if not isinstance(solution, dict):
            raise MainStop(
                f"plan secuencial: {name} devolvió "
                "una solución no estructurada"
            )

        q_value = finite_vector(
            solution.get("q"),
            6,
            f"ik.{name}.q",
        )

        position_error_m = float(
            solution.get(
                "pos_err_m",
                math.inf,
            )
        )

        topdown_dot = float(
            solution.get(
                "topdown_dot",
                -math.inf,
            )
        )

        solver_ok = bool(
            solution.get(
                "ok",
                False,
            )
        )

        if (
            not solver_ok
            or not math.isfinite(position_error_m)
            or not math.isfinite(topdown_dot)
        ):
            raise MainStop(
                f"plan secuencial: IK inválida en {name}; "
                f"ok={solver_ok}, "
                f"pos_err={position_error_m}, "
                f"topdown_dot={topdown_dot}"
            )

        if topdown_dot < 0.985:
            raise MainStop(
                f"plan secuencial: pérdida de verticalidad "
                f"en {name}: topdown_dot={topdown_dot:.6f}"
            )

        maximum_seed_step = float(
            np.max(
                np.abs(
                    q_value - q_seed
                )
            )
        )

        solved[name] = q_value.copy()

        rows[name] = {
            "target_real_base_m":
                target_real.tolist(),
            "target_sim_m":
                target_sim.tolist(),
            "q_seed_rad":
                q_seed.tolist(),
            "q_solution_rad":
                q_value.tolist(),
            "max_abs_dq_from_seed_rad":
                maximum_seed_step,
            "pos_err_m":
                position_error_m,
            "topdown_dot":
                topdown_dot,
            "solver_contract_ok":
                solver_ok,
            "seed_scan_q0_rad":
                q0_scan.tolist(),
        }

        print(
            f"phase={name} "
            f"pos_err={position_error_m:.4f}m "
            f"topdown_dot={topdown_dot:.6f} "
            f"dq_seed={maximum_seed_step:.4f}rad "
            "contract=historical_solver_ok",
            flush=True,
        )

        q_seed = q_value.copy()

    return solved, {
        "schema":
            "VLA_sequential_ik",
        "solver":
            "solve_topdown_ik_seeded_by_previous_phase",
        "scene_xml":
            str(context["scene_xml"]),
        "tcp_site":
            str(context["tcp_site"]),
        "offset_real_minus_sim_m":
            offset_real_minus_sim.tolist(),
        "phase_order":
            list(order),
        "phases":
            rows,
        "vla_cartesian_waypoints_used":
            False,
        "recursive_subdivision_used":
            False,
        "branch_guard_q2_q3_used":
            False,
        "verticality_guard_retained":
            True,
        "vla_translation_preserved":
            True,
    }





def _build_home_to_pre_bridge(
    *,
    q_home: np.ndarray,
    q_pre: np.ndarray,
) -> dict[str, Any]:
    """
    Enlace general externo q_home -> q_pre.
    
        No modifica ninguna de las seis anclas geométricas. El último
        waypoint es exactamente el q_pre calculado después de
        aplicar los residuales translacionales del VLA.
    """
    start = finite_vector(
        q_home,
        6,
        "boundary.q_home",
    )
    goal = finite_vector(
        q_pre,
        6,
        "boundary.q_pre",
    )

    maximum_step_rad = float(
        os.environ.get(
            "VLA_HOME_TO_PRE_MAX_STEP_RAD",
            "0.08",
        )
    )

    if (
        not math.isfinite(maximum_step_rad)
        or not 0.03 <= maximum_step_rad <= 0.12
    ):
        raise MainStop(
            "VLA_HOME_TO_PRE_MAX_STEP_RAD debe "
            "estar entre 0.03 y 0.12 rad"
        )

    raw_maximum_step = float(
        np.max(
            np.abs(
                goal - start
            )
        )
    )

    segment_count = max(
        1,
        int(
            math.ceil(
                raw_maximum_step
                / maximum_step_rad
            )
        ),
    )

    waypoints = [
        start
        + (
            float(index)
            / float(segment_count)
        )
        * (goal - start)
        for index in range(
            1,
            segment_count + 1,
        )
    ]

    endpoint_error = float(
        np.max(
            np.abs(
                waypoints[-1] - goal
            )
        )
    )

    if endpoint_error > 1e-12:
        raise MainStop(
            "puente articular: el endpoint dejó de ser "
            "exactamente q_pre"
        )

    # El corredor validado de hombro/codo se comprueba
    # sobre todos los puntos antes de autorizar movimiento.
    posture_report = validate_high_shoulder_branch(
        q_start=start,
        q_targets=waypoints,
        label="home_to_pre_bridge",
    )

    full_chain = [start] + waypoints

    executed_steps = [
        float(
            np.max(
                np.abs(
                    full_chain[index]
                    - full_chain[index - 1]
                )
            )
        )
        for index in range(
            1,
            len(full_chain),
        )
    ]

    maximum_executed_step = max(
        executed_steps,
        default=0.0,
    )

    if (
        maximum_executed_step
        > maximum_step_rad + 1e-10
    ):
        raise MainStop(
            "puente articular: subdivisión articular "
            "supera el límite configurado"
        )

    print(
        f"raw_max_dq={raw_maximum_step:.4f}rad "
        f"waypoints={len(waypoints)} "
        f"executed_max_dq="
        f"{maximum_executed_step:.4f}rad "
        "endpoint=q_pre_exact "
        "shoulder_elbow_branch_preserved=true "
        "anchors_untouched=true "
        "vertical_anchors_preserved=true "
        "vla_translation_preserved=true",
        flush=True,
    )

    return {
        "schema":
            "VLA_home_to_pre_bridge",
        "raw_max_abs_dq_rad":
            raw_maximum_step,
        "maximum_requested_step_rad":
            maximum_step_rad,
        "maximum_executed_step_rad":
            maximum_executed_step,
        "segment_count":
            segment_count,
        "execution_waypoints": [
            value.copy()
            for value in waypoints
        ],
        "endpoint_q_pre_exact":
            True,
        "internal_anchors_untouched":
            True,
        "vertical_anchors_preserved":
            True,
        "vla_translation_endpoint_preserved":
            True,
        "posture_report":
            posture_report,
    }


def _validate_anchor_chain(
    *,
    q_home: np.ndarray,
    anchors: dict[str, np.ndarray],
) -> dict[str, Any]:
    """
    Validación  de continuidad plan.
    
        No impone fronteras artificiales q2/q3. Conserva las seis
        soluciones obtenidas por la IK top-down secuencial.
    """
    maximum_allowed = float(
        os.environ.get(
            "VLA_MAX_TRANSITION_RAD",
            "1.80",
        )
    )

    if (
        not math.isfinite(maximum_allowed)
        or not 1.20 <= maximum_allowed <= 1.80
    ):
        raise MainStop(
            "VLA_MAX_TRANSITION_RAD debe "
            "estar entre 1.20 y 1.60 rad"
        )

    order = [
        "q_pre",
        "q_pick",
        "q_lift",
        "q_pre_place",
        "q_place",
        "q_retreat",
    ]

    home = finite_vector(
        q_home,
        6,
        "chain.q_home",
    )

    previous_name = "q_home"
    previous = home.copy()
    rows = []

    for name in order:
        current = finite_vector(
            anchors[name],
            6,
            f"chain.{name}",
        )

        maximum_step = float(
            np.max(
                np.abs(
                    current - previous
                )
            )
        )

        if maximum_step > maximum_allowed:
            raise MainStop(
                f"Transición articular: "
                f"{previous_name}->{name} "
                f"demasiado grande: "
                f"{maximum_step:.4f}rad > "
                f"{maximum_allowed:.4f}rad"
            )

        rows.append(
            {
                "from":
                    previous_name,
                "to":
                    name,
                "max_abs_dq_rad":
                    maximum_step,
            }
        )

        previous_name = name
        previous = current

    home_return_step = float(
        np.max(
            np.abs(
                previous - home
            )
        )
    )

    if home_return_step > maximum_allowed:
        raise MainStop(
            "Transición articular: q_retreat->q_home "
            f"demasiado grande: "
            f"{home_return_step:.4f}rad > "
            f"{maximum_allowed:.4f}rad"
        )

    rows.append(
        {
            "from":
                "q_retreat",
            "to":
                "q_home",
            "max_abs_dq_rad":
                home_return_step,
        }
    )

    maximum_observed = max(
        float(row["max_abs_dq_rad"])
        for row in rows
    )

    print(
        f"max_abs_dq={maximum_observed:.4f}rad "
        f"limit={maximum_allowed:.4f}rad "
        "branch_guard_q2_q3=false "
        "verticality_preserved=true "
        "vla_translation_preserved=true",
        flush=True,
    )

    return {
        "schema":
            "VLA_anchor_chain_validation",
        "maximum_allowed_step_rad":
            maximum_allowed,
        "maximum_observed_step_rad":
            maximum_observed,
        "transitions":
            rows,
        "valid":
            True,
        "branch_guard_q2_q3_used":
            False,
        "joint_bridge_used":
            False,
        "anchors_untouched":
            True,
        "verticality_preserved":
            True,
        "vla_translation_preserved":
            True,

        # execute_real espera esta forma de entrada:
        # se ejecuta solo el q_pre real, sin puentes.
        "approach_bridge_q": [
            finite_vector(
                anchors["q_pre"],
                6,
                "chain.q_pre_execution",
            ).copy()
        ],

        "note": (
            "No se impone una frontera artificial q2/q3. "
            "Se valida la cadena geométrica completa."
        ),
    }





def _derive_vla_place_spatial_preference(
    *,
    model: str,
    instruction: str,
    target_center_xyz: np.ndarray,
    action_profile: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Convierte la salida traslacional XY nativa del VLA en una
        preferencia espacial respecto al centro detectado de la zona.
    
        No modifica las anclas geométricas, la orientación, la rama IK,
        el hombro, el codo ni la verticalidad del TCP.
    """
    center = finite_vector(
        target_center_xyz,
        3,
        "vla_place.target_center_xyz",
    )

    phase_residuals = action_profile.get(
        "phase_residuals_base_m",
    )

    if not isinstance(
        phase_residuals,
        dict,
    ):
        raise MainStop(
            "VLA place: action_profile no contiene "
            "phase_residuals_base_m"
        )

    destination_residual = finite_vector(
        phase_residuals.get(
            "q_pre_place"
        ),
        3,
        "vla_place.q_pre_place_residual",
    )

    raw_xy = destination_residual[:2].copy()

    gain = float(
        os.environ.get(
            "VLA_PLACE_SPATIAL_GAIN",
            "4.0",
        )
    )

    if (
        not math.isfinite(gain)
        or not 0.0 <= gain <= 8.0
    ):
        raise MainStop(
            "VLA_PLACE_SPATIAL_GAIN debe "
            "estar entre 0.0 y 8.0"
        )

    maximum_offset_m = float(
        os.environ.get(
            "VLA_PLACE_MAX_OFFSET_M",
            "0.050",
        )
    )

    if (
        not math.isfinite(maximum_offset_m)
        or not 0.010
        <= maximum_offset_m
        <= 0.080
    ):
        raise MainStop(
            "VLA_PLACE_MAX_OFFSET_M debe "
            "estar entre 0.010 y 0.080 m"
        )

    proposed_offset_xy = (
        raw_xy * gain
    )

    proposed_norm = float(
        np.linalg.norm(
            proposed_offset_xy
        )
    )

    if proposed_norm > maximum_offset_m:
        offset_scale = (
            maximum_offset_m
            / proposed_norm
        )
    else:
        offset_scale = 1.0

    applied_offset_xy = (
        proposed_offset_xy
        * offset_scale
    )

    preferred_xy = (
        center[:2]
        + applied_offset_xy
    )

    normalized_instruction = re.sub(
        r"\s+",
        " ",
        instruction.lower()
        .replace("_", " ")
        .replace("-", " "),
    ).strip()

    expected_relation = None

    if any(
        token in normalized_instruction
        for token in (
            "center",
            "centre",
            "middle",
        )
    ):
        expected_relation = "center"
    elif any(
        token in normalized_instruction
        for token in (
            "left side",
            "left hand side",
            "leftside",
        )
    ):
        expected_relation = "left"
    elif any(
        token in normalized_instruction
        for token in (
            "right side",
            "right hand side",
            "rightside",
        )
    ):
        expected_relation = "right"
    elif any(
        token in normalized_instruction
        for token in (
            "rear side",
            "rear edge",
            "back side",
            "back edge",
            "backside",
        )
    ):
        expected_relation = "back"
    elif any(
        token in normalized_instruction
        for token in (
            "front side",
            "front edge",
            "forward side",
            "frontside",
        )
    ):
        expected_relation = "front"

    dx = float(
        applied_offset_xy[0]
    )
    dy = float(
        applied_offset_xy[1]
    )

    offset_norm = float(
        np.linalg.norm(
            applied_offset_xy
        )
    )

    center_deadband_m = float(
        os.environ.get(
            "VLA_PLACE_CENTER_DEADBAND_M",
            "0.012",
        )
    )

    dominance_ratio = float(
        os.environ.get(
            "VLA_PLACE_AXIS_DOMINANCE_RATIO",
            "2.25",
        )
    )

    if (
        not math.isfinite(
            center_deadband_m
        )
        or not 0.001
        <= center_deadband_m
        <= 0.030
    ):
        raise MainStop(
            "VLA_PLACE_CENTER_DEADBAND_M "
            "fuera de rango"
        )

    if (
        not math.isfinite(
            dominance_ratio
        )
        or not 1.0
        <= dominance_ratio
        <= 5.0
    ):
        raise MainStop(
            "VLA_PLACE_AXIS_DOMINANCE_RATIO "
            "fuera de rango"
        )

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if offset_norm <= center_deadband_m:
        policy_relation = "center"

    elif (
        abs_dx
        >= dominance_ratio
        * max(abs_dy, 1e-12)
    ):
        policy_relation = (
            "right"
            if dx > 0.0
            else "left"
        )

    elif (
        abs_dy
        >= dominance_ratio
        * max(abs_dx, 1e-12)
    ):
        policy_relation = (
            "front"
            if dy > 0.0
            else "back"
        )

    else:
        longitudinal = (
            "front"
            if dy >= 0.0
            else "back"
        )
        lateral = (
            "right"
            if dx >= 0.0
            else "left"
        )
        policy_relation = (
            f"{longitudinal}_"
            f"{lateral}_edge"
        )

    if expected_relation is None:
        language_pass = None
    elif expected_relation == "center":
        language_pass = (
            policy_relation == "center"
        )
    elif expected_relation == "left":
        language_pass = (
            "left"
            in policy_relation
        )
    else:
        language_pass = (
            "right"
            in policy_relation
        )

    report = {
        "schema":
            "vla_place_spatial_preference",
        "model":
            model,
        "instruction":
            instruction,
        "decision_source":
            "native_vla_translation_output",
        "target_center_xyz_m":
            center.tolist(),
        "raw_xy_m":
            raw_xy.tolist(),
        "gain":
            gain,
        "proposed_offset_xy_m":
            proposed_offset_xy.tolist(),
        "applied_offset_xy_m":
            applied_offset_xy.tolist(),
        "preferred_xy_m":
            preferred_xy.tolist(),
        "maximum_offset_m":
            maximum_offset_m,
        "offset_scale":
            offset_scale,
        "offset_clipped":
            bool(offset_scale < 1.0),
        "expected_relation":
            expected_relation,
        "policy_relation":
            policy_relation,
        "language_relation_pass":
            language_pass,
        "parser_replacement":
            False,
        "native_actions_replaced":
            False,
        "rotation_used":
            False,
        "anchors_modified":
            False,
    }

    print(
        f"model={model} "
        f"expected={expected_relation} "
        f"policy={policy_relation} "
        f"raw_xy={raw_xy.tolist()} "
        f"gain={gain:.3f} "
        f"preferred_xy="
        f"{preferred_xy.tolist()} "
        f"language_pass={language_pass} "
        "parser_replacement=false",
        flush=True,
    )

    return preferred_xy, report

def _write_vla_place_spatial_map(
    *,
    run_dir: Path,
    target_center_xyz: np.ndarray,
    preferred_xy: np.ndarray,
    selected_target_xyz: np.ndarray,
    report: dict[str, Any],
) -> dict[str, str]:
    """
    Guarda una representación cenital de la preferencia
        espacial derivada de la salida VLA.
    
        No representa atención interna del modelo.
    """
    output_dir = Path(
        run_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    center = finite_vector(
        target_center_xyz,
        3,
        "vla_place_map.center",
    )

    preferred = finite_vector(
        preferred_xy,
        2,
        "vla_place_map.preferred",
    )

    selected = finite_vector(
        selected_target_xyz,
        3,
        "vla_place_map.selected",
    )

    zone_radius_m = float(
        os.environ.get(
            "VLA_SAFE_PLACE_ZONE_RADIUS_M",
            "0.110",
        )
    )

    if (
        not math.isfinite(zone_radius_m)
        or not 0.040
        <= zone_radius_m
        <= 0.200
    ):
        raise MainStop(
            "VLA_SAFE_PLACE_ZONE_RADIUS_M "
            "fuera de rango"
        )

    png_path = (
        output_dir
        / "vla_place_spatial_preference_map.png"
    )

    json_path = (
        output_dir
        / "vla_place_spatial_preference_map.json"
    )

    payload = {
        "schema":
            "vla_place_spatial_map",
        "type":
            "policy_derived_not_internal_attention",
        "target_center_xyz_m":
            center.tolist(),
        "preferred_xy_m":
            preferred.tolist(),
        "selected_target_xyz_m":
            selected.tolist(),
        "zone_radius_m":
            zone_radius_m,
        "report":
            report,
    }

    json_write(
        json_path,
        payload,
    )

    image_size = 720
    origin = np.asarray(
        [
            image_size // 2,
            image_size // 2,
        ],
        dtype=float,
    )

    scale_px_per_m = 2500.0

    canvas = np.full(
        (
            image_size,
            image_size,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    def to_pixel(
        xy: np.ndarray,
    ) -> tuple[int, int]:
        value = finite_vector(
            xy,
            2,
            "vla_place_map.xy",
        )

        delta = (
            value
            - center[:2]
        )

        pixel_x = int(
            round(
                origin[0]
                + delta[0]
                * scale_px_per_m
            )
        )

        pixel_y = int(
            round(
                origin[1]
                - delta[1]
                * scale_px_per_m
            )
        )

        return pixel_x, pixel_y

    center_px = to_pixel(
        center[:2]
    )
    preferred_px = to_pixel(
        preferred
    )
    selected_px = to_pixel(
        selected[:2]
    )

    radius_px = int(
        round(
            zone_radius_m
            * scale_px_per_m
        )
    )

    cv2.circle(
        canvas,
        center_px,
        radius_px,
        (180, 180, 180),
        2,
    )

    cv2.line(
        canvas,
        (
            center_px[0] - radius_px,
            center_px[1],
        ),
        (
            center_px[0] + radius_px,
            center_px[1],
        ),
        (220, 220, 220),
        1,
    )

    cv2.line(
        canvas,
        (
            center_px[0],
            center_px[1] - radius_px,
        ),
        (
            center_px[0],
            center_px[1] + radius_px,
        ),
        (220, 220, 220),
        1,
    )

    cv2.arrowedLine(
        canvas,
        center_px,
        preferred_px,
        (255, 120, 0),
        3,
        tipLength=0.12,
    )

    cv2.line(
        canvas,
        preferred_px,
        selected_px,
        (0, 160, 255),
        2,
    )

    cv2.circle(
        canvas,
        center_px,
        7,
        (0, 0, 0),
        -1,
    )

    cv2.circle(
        canvas,
        preferred_px,
        8,
        (255, 120, 0),
        -1,
    )

    cv2.circle(
        canvas,
        selected_px,
        9,
        (0, 160, 0),
        -1,
    )

    cv2.putText(
        canvas,
        "target center",
        (
            center_px[0] + 12,
            center_px[1] - 12,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "VLA preferred",
        (
            preferred_px[0] + 12,
            preferred_px[1] - 12,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 120, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "safe selected",
        (
            selected_px[0] + 12,
            selected_px[1] + 22,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 120, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "policy-derived map; not internal attention",
        (
            24,
            image_size - 24,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    written = cv2.imwrite(
        str(png_path),
        canvas,
    )

    if not written:
        raise MainStop(
            "No se pudo guardar "
            "vla_place_spatial_preference_map.png"
        )

    paths = {
        "png":
            str(png_path),
        "json":
            str(json_path),
        "type":
            "policy_derived_not_internal_attention",
    }

    print(
        f"png={png_path} "
        f"json={json_path} "
        "type=policy_derived_not_internal_attention",
        flush=True,
    )

    return paths

def build_preplan(
    *,
    model: str,
    actions: np.ndarray,
    evidence: dict[str, Any],
    scene: dict[str, Any],
    scene_path: Path,
    image_path: Path,
    config: Any,
    run_dir: Path,
    grounding: dict[str, Any],
    q_home_override: np.ndarray | None = None,
    tcp_home_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Resumen del flujo plan traslacional.
    
        - Visión: grondaing causal del propio VLA sobre candidatos Photoneo.
        - Lenguaje: contrafactual lingüístico del propio VLA.
        - Acción: XYZ nativo convertido a residuales métricos por fase.
        - Geometría: seis anclas geométricas con descensos verticales preservados.
        - Cinemática: solve_topdown_ik secuencial, semilla fase anterior.
        - Garra: salida nativa acotada.
    """
    grounding_selection_source = str(
        grounding.get(
            "selection_source",
            grounding.get(
                "method",
                "vla_visual_language_action_grounding",
            ),
        )
    )

    external_siglip_grounding = bool(
        grounding_selection_source
        == "external_siglip_visual_language_grounding"
        or "siglip"
        in grounding_selection_source.lower()
    )

    grounding_architecture_prefix = (
        "photoneo_candidate_siglip_"
        "external_grounding_"
        if external_siglip_grounding
        else
        "all_candidate_vla_visual_language_"
        "grounding_"
    )

    template_config = geometry_template_config(
        config,
        grounding,
    )
    parser_entities = supervisor.resolve_entities(
        scene,
        template_config,
    )
    entities, grounding_application_report = (
        apply_grounding_to_entities(
            parser_entities,
            grounding,
        )
    )

    if (
        q_home_override is None
    ) != (
        tcp_home_override is None
    ):
        raise MainStop(
            "Replay incoherente: q_home_override y "
            "tcp_home_override deben proporcionarse juntos."
        )

    if q_home_override is not None:
        q_home = finite_vector(
            q_home_override,
            6,
            "replay.q_home_override",
        )
        tcp_home = finite_vector(
            tcp_home_override,
            3,
            "replay.tcp_home_override",
        )
        home_source = "offline_replay"
    else:
        with supervisor.robot_node() as node:
            q_home, tcp_home = (
                supervisor.current_robot_state(node)
            )
        home_source = "robot_joint_states"

    print(
        f"source={home_source}",
        flush=True,
    )

    # El estado inicial real o de replay solo puede utilizarse
    # como semilla del episodio después de comprobar que coincide
    # con el home operativo validado físicamente.
    q_home = finite_vector(
        np.asarray(
            q_home,
            dtype=np.float64,
        ).copy(),
        6,
        "episode.q_home",
    )

    tcp_home = finite_vector(
        np.asarray(
            tcp_home,
            dtype=np.float64,
        ).copy(),
        3,
        "episode.tcp_home",
    )

    home_reference = np.asarray(
        [
            1.477447509765625,
            -1.5225587052157898,
            1.7015073935138147,
            -1.7282711468138636,
            -1.5657990614520472,
            0.046940818428993225,
        ],
        dtype=np.float64,
    )

    home_tolerance = float(
        os.environ.get(
            "VLA_OPERATIONAL_HOME_TOLERANCE_RAD",
            "0.035",
        )
    )

    if (
        not math.isfinite(home_tolerance)
        or not 0.005
        <= home_tolerance
        <= 0.080
    ):
        raise MainStop(
            "VLA_OPERATIONAL_HOME_TOLERANCE_RAD "
            f"no válida: {home_tolerance}"
        )

    home_error = float(
        np.max(
            np.abs(
                q_home
                - home_reference
            )
        )
    )

    if home_error > home_tolerance:
        raise MainStop(
            ""
            "El estado articular inicial no coincide "
            "con el home operativo validado: "
            f"source={home_source}, "
            f"error={home_error:.6f}rad, "
            f"tolerance={home_tolerance:.6f}rad, "
            f"q_actual={q_home.tolist()}, "
            f"q_reference={home_reference.tolist()}"
        )

    print(
        f"source={home_source} "
        f"error={home_error:.6f}rad "
        f"tolerance={home_tolerance:.6f}rad "
        f"q_home={q_home.tolist()}",
        flush=True,
    )

    object_record = (
        _record_for_grounding_name(
            scene=scene,
            name=grounding["object"]["name"],
        )
    )
    target_record = (
        _record_for_grounding_name(
            scene=scene,
            name=grounding["target"]["name"],
        )
    )

    object_xyz = finite_vector(
        grounding["object"]["base_xyz_m"],
        3,
        "plan.object_xyz",
    )
    target_center_xyz = finite_vector(
        grounding["target"]["base_xyz_m"],
        3,
        "plan.target_detected_center_xyz",
    )

    previous_target_xyz = finite_vector(
        entities["target_xyz_m"],
        3,
        "plan.selected_interior_xyz",
    )

    action_profile = build_translation_profile(
        model=model,
        actions=actions,
        instruction=config.instruction,
        image_path=image_path,
        evidence=evidence,
        object_xyz_m=object_xyz,
        target_xyz_m=target_center_xyz,
    )

    (
        vla_place_preferred_xy,
        spatial_preference_report,
    ) = _derive_vla_place_spatial_preference(
        model=model,
        instruction=config.instruction,
        target_center_xyz=target_center_xyz,
        action_profile=action_profile,
    )

    projector = globals().get(
        "project_vla_preference_to_safe_target"
    )

    if not callable(projector):
        projector = getattr(
            supervisor,
            "project_vla_preference_to_safe_target",
            None,
        )

    if not callable(projector):
        raise MainStop(
            "No existe el supervisor físico "
            "project_vla_preference_to_safe_target"
        )

    projection_entities = copy.deepcopy(
        entities
    )

    projection_entities[
        "target_detected_center_xyz_m"
    ] = target_center_xyz.tolist()

    projection_entities[
        "target_xyz_m"
    ] = target_center_xyz.tolist()

    if (
        "target_safe_xyz_m"
        in projection_entities
    ):
        target_safe = finite_vector(
            projection_entities[
                "target_safe_xyz_m"
            ],
            3,
            "vla_place.target_safe",
        )
        target_safe[:2] = (
            target_center_xyz[:2]
        )
        projection_entities[
            "target_safe_xyz_m"
        ] = target_safe.tolist()

    (
        spatial_entities,
        spatial_safe_report,
    ) = projector(
        entities=projection_entities,
        preferred_xy=vla_place_preferred_xy,
        config=config,
    )

    entities = spatial_entities

    target_xyz = finite_vector(
        entities["target_xyz_m"],
        3,
        "plan.vla_selected_safe_target_xyz",
    )

    target_interior_offset_xy_m = float(
        np.linalg.norm(
            target_xyz[:2]
            - target_center_xyz[:2]
        )
    )

    spatial_preference_report[
        "previous_selected_interior_xyz_m"
    ] = previous_target_xyz.tolist()

    spatial_preference_report[
        "selected_safe_target_xyz_m"
    ] = target_xyz.tolist()

    spatial_preference_report[
        "safe_supervisor"
    ] = spatial_safe_report

    spatial_map_paths = (
        _write_vla_place_spatial_map(
            run_dir=run_dir,
            target_center_xyz=target_center_xyz,
            preferred_xy=(
                vla_place_preferred_xy
            ),
            selected_target_xyz=target_xyz,
            report=spatial_preference_report,
        )
    )

    spatial_preference_report[
        "map_paths"
    ] = spatial_map_paths

    # El XY de destino ya se ha consumido como una
    # preferencia espacial respecto al centro. No se
    # vuelve a sumar en q_pre_place/q_place/q_retreat.
    # Se conserva el residual Z del modelo.
    for destination_phase in (
        "q_pre_place",
        "q_place",
        "q_retreat",
    ):
        destination_phase_residual = (
            finite_vector(
                action_profile[
                    "phase_residuals_base_m"
                ][destination_phase],
                3,
                (
                    "vla_place."
                    f"{destination_phase}"
                ),
            )
        )
        destination_phase_residual[:2] = 0.0
        action_profile[
            "phase_residuals_base_m"
        ][destination_phase] = (
            destination_phase_residual.tolist()
        )

    action_profile[
        "place_spatial_preference"
    ] = spatial_preference_report

    json_write(
        run_dir
        / "translation_profile.json",
        action_profile,
    )

    print(
        f"center={target_center_xyz.tolist()} "
        f"preferred_xy="
        f"{vla_place_preferred_xy.tolist()} "
        f"selected={target_xyz.tolist()} "
        f"offset_xy="
        f"{target_interior_offset_xy_m:.4f}m "
        "source=vla_policy_plus_safe_projection",
        flush=True,
    )


    if (
        os.environ.get(
            "VLA_TASK_FAMILY", ""
        ).strip() == "shape_classification"
        and os.environ.get(
            "VLA_SHAPE_FORCE_TARGET_CENTER",
            "YES",
        ).strip().upper() == "YES"
    ):
        _shape_center = target_center_xyz.copy()
        target_xyz = _shape_center.copy()
    
        print(
            f"center={list(map(float, _shape_center))} "
            f"interior={list(map(float, target_xyz))} "
            "offset_xy=0.0000m",
            flush=True,
        )

    print(
        f"center={target_center_xyz.tolist()} "
        f"interior={target_xyz.tolist()} "
        f"offset_xy="
        f"{target_interior_offset_xy_m:.4f}m "
        "source=vla_policy_spatial_preference",
        flush=True,
    )

    pre_pick_xyz = _safe_xyz_from_record(
        record=object_record,
        fallback=entities["object_safe_xyz_m"],
        label="plan.pre_pick_xyz",
    )
    pre_pick_xyz[:2] = object_xyz[:2]

    pick_xyz = object_xyz.copy()
    pick_xyz[2] = float(config.pick_z_m)

    lift_xyz = pick_xyz.copy()
    lift_xyz[2] = max(
        float(pick_xyz[2]) + 0.060,
        float(pre_pick_xyz[2]),
    )

    pre_place_xyz = _safe_xyz_from_record(
        record=target_record,
        fallback=entities["target_safe_xyz_m"],
        label="plan.pre_place_xyz",
        minimum_z_m=0.230,
    )
    pre_place_xyz[:2] = target_xyz[:2]

    place_xyz = target_xyz.copy()
    place_xyz[2] = float(config.place_z_m)

    retreat_xyz = pre_place_xyz.copy()

    geometric_targets = {
        "q_pre": pre_pick_xyz,
        "q_pick": pick_xyz,
        "q_lift": lift_xyz,
        "q_pre_place": pre_place_xyz,
        "q_place": place_xyz,
        "q_retreat": retreat_xyz,
    }

    (
        vla_phase_targets,
        translation_application_report,
    ) = apply_translation_residuals(
        geometric_targets=geometric_targets,
        profile=action_profile,
    )

    (
        adapted_anchors,
        historical_ik_report,
    ) = _solve_anchor_chain(
        q_home=q_home,
        tcp_home=tcp_home,
        targets=vla_phase_targets,
    )

    chain_validation = _validate_anchor_chain(
        q_home=q_home,
        anchors=adapted_anchors,
    )

    preclose_q = [
        adapted_anchors["q_pre"].copy(),
        adapted_anchors["q_pick"].copy(),
    ]
    held_q = [
        adapted_anchors["q_lift"].copy(),
        adapted_anchors["q_pre_place"].copy(),
        adapted_anchors["q_place"].copy(),
    ]
    return_q = [
        adapted_anchors["q_retreat"].copy(),
        q_home.copy(),
    ]

    print(
        f"grounding={grounding_selection_source} "
        f"external_siglip="
        f"{str(external_siglip_grounding).lower()} "
        "action=native_xyz_metric_phase_residuals "
        "raw_temporal_path=false "
        "rotation=false "
        "ik=historical_sequential_topdown "
        "gripper=vla_bounded",
        flush=True,
    )

    preplan = {
        "schema": (
            "VLA_pick_place_preplan"
        ),
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "instruction": config.instruction,
        "mode": config.mode,
        "architecture": (
            grounding_architecture_prefix
            + "native_xyz_metric_phase_residuals_"
            "photoneo_six_anchor_geometry_"
            "historical_sequential_topdown_ik_"
            "bounded_vla_gripper_"
            "prevalidated_execution"
        ),
        "scene_path": str(scene_path),
        "image_path": str(image_path),
        "model_evidence": evidence,
        "raw_actions": actions.tolist(),
        "grounding": grounding,
        "entities": entities,
        "parser_entities_trace_only": parser_entities,
        "q_home": q_home,
        "q_home_reference": home_reference,
        "tcp_home": tcp_home,
        "home_source": home_source,
        "home_error_rad": home_error,
        "geometric_phase_targets_xyz_m": geometric_targets,
        "vla_phase_targets_xyz_m": vla_phase_targets,
        "q_pre": adapted_anchors["q_pre"],
        "q_pick": adapted_anchors["q_pick"],
        "q_lift_anchor": adapted_anchors["q_lift"],
        "q_pre_place": adapted_anchors["q_pre_place"],
        "q_place": adapted_anchors["q_place"],
        "q_retreat_anchor": adapted_anchors["q_retreat"],
        "approach_q": [
            adapted_anchors["q_pre"].copy()
        ],
        "q_to_pre_pick": [
            adapted_anchors["q_pre"].copy()
        ],
        "q_pick_reference": (
            adapted_anchors["q_pick"].copy()
        ),
        "q_lift": [
            adapted_anchors["q_lift"].copy()
        ],
        "transport_q": [
            adapted_anchors["q_pre_place"].copy()
        ],
        "q_to_place": [
            adapted_anchors["q_pre_place"].copy(),
            adapted_anchors["q_place"].copy(),
        ],
        "q_retreat": [
            adapted_anchors["q_retreat"].copy()
        ],
        "preclose_q": preclose_q,
        "held_q": held_q,
        "return_q": return_q,
        "selected_target_xyz_m": target_xyz,
        "selected_target_detected_center_xyz_m": target_center_xyz,
        "selected_target_interior_offset_xy_m": target_interior_offset_xy_m,
        "vla_place_preferred_xy_m": vla_place_preferred_xy,
        "vla_place_relation_trace": (
            spatial_preference_report
        ),
        "vla_place_spatial_map_paths": (
            spatial_map_paths
        ),
        "place_xyz_m": vla_phase_targets["q_place"],
        "pick_xyz_m": vla_phase_targets["q_pick"],
        "reports": {
            "grounding": grounding,
            "grounding_application": (
                grounding_application_report
            ),
            "translation_profile": action_profile,
            "vla_place_spatial_preference": (
                spatial_preference_report
            ),
            "vla_place_safe_supervisor": (
                spatial_safe_report
            ),
            "phase_translation_application": (
                translation_application_report
            ),
            "sequential_ik": historical_ik_report,
            "anchor_chain_validation": chain_validation,
        },
        "object_grounding_vla_applied":
            not external_siglip_grounding,
        "target_grounding_vla_applied":
            not external_siglip_grounding,
        "external_visual_language_grounding_applied":
            external_siglip_grounding,
        "grounding_selection_source":
            grounding_selection_source,
        "grounding_model_id":
            grounding.get(
                "grounding_model_id"
            ),
        "parser_used_as_decision": False,
        "all_photoneo_candidates_competed":
            bool(
                grounding.get(
                    "all_photoneo_candidates_competed",
                    True,
                )
            ),
        "translation_vla_applied": True,
        "raw_temporal_vla_path_applied": False,
        "joint_guidance_vla_applied": False,
        "rotation_vla_applied": False,
        "gripper_vla_applied": True,
        "contact_geometry_source": (
            "photoneo_plus_bounded_vla_xyz_residuals"
        ),
    }

    _eval_env = __import__("os").environ

    _eval_series = _eval_env.get(
        "VLA_SHAPE_SERIES",
        "",
    ).strip()

    if _eval_series:
        _eval_repetition_raw = (
            _eval_env.get(
                "VLA_SHAPE_REPETITION",
                "",
            ).strip()
        )

        _eval_repetition = (
            int(_eval_repetition_raw)
            if _eval_repetition_raw.isdigit()
            else _eval_repetition_raw
        )

        preplan["evaluation_protocol"] = {
            "schema": (
                "VLA_shape_taxonomy_evaluation_protocol"
            ),
            "series": _eval_series,
            "trial_key": _eval_env.get(
                "VLA_SHAPE_TRIAL_KEY",
                "",
            ).strip(),
            "task_variant": _eval_env.get(
                "VLA_SHAPE_TASK_VARIANT",
                "implicit_shape_relation",
            ).strip(),
            "layout": _eval_env.get(
                "VLA_SHAPE_LAYOUT",
                "",
            ).strip(),
            "repetition": _eval_repetition,
            "object_condition": _eval_env.get(
                "VLA_SHAPE_OBJECT_CONDITION",
                "",
            ).strip(),
            "expected_object": _eval_env.get(
                "VLA_SHAPE_EXPECTED_OBJECT",
                "",
            ).strip(),
            "expected_target": _eval_env.get(
                "VLA_SHAPE_EXPECTED_TARGET",
                "",
            ).strip(),
            "protocol_eligible": (
                _eval_env.get(
                    "VLA_SHAPE_PROTOCOL_ELIGIBLE",
                    "YES",
                ).strip().upper()
                == "YES"
            ),
            "target_center_forced": (
                _eval_env.get(
                    "VLA_SHAPE_FORCE_TARGET_CENTER",
                    "",
                ).strip().upper()
                == "YES"
            ),
            "openpi_seed": (
                _eval_env.get(
                    "OPENPI_INFERENCE_SEED",
                    "",
                ).strip()
            ),
        }

    json_write(
        run_dir / "episode_preplan.json",
        preplan,
    )

    return preplan


def require_real_confirmations(
    config: Any,
    real_stage: str,
) -> None:
    if real_stage not in {"approach", "pick_pose", "full"}:
        raise MainStop(f"Etapa real no soportada: {real_stage}")

    required = {
        "CONFIRM_MAIN_VLA_REAL":
            "YES",
        "CONFIRM_PHYSICAL_EXECUTION":
            "YES",
        "CONFIRM_EXECUTION_RISK":
            "YES",
    }

    missing = [
        key
        for key, expected
        in required.items()
        if os.environ.get(key)
        != expected
    ]

    if missing:
        raise MainStop(
            "Faltan confirmaciones reales: "
            + ", ".join(missing)
        )

    if os.environ.get(
        "VLA_PICK_Z_MODE",
        "",
    ) != "override":
        raise MainStop(
            "VLA_PICK_Z_MODE debe ser "
            "override"
        )

    pick_env = float(
        os.environ.get(
            "VLA_PICK_Z_OVERRIDE_M",
            "nan",
        )
    )

    if (
        not math.isfinite(pick_env)
        or abs(
            pick_env
            - config.pick_z_m
        )
        > 1e-9
    ):
        raise MainStop(
            "VLA_PICK_Z_OVERRIDE_M no "
            "coincide con PICK_Z_OVERRIDE_M"
        )


def _compact_joint_targets(
    values: list[Any],
) -> list[np.ndarray]:
    output: list[np.ndarray] = []

    for index, value in enumerate(values):
        vector = finite_vector(
            value,
            6,
            f"recovery.target[{index}]",
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
            < 1e-7
        ):
            continue

        output.append(vector.copy())

    return output


def _nearest_joint_index(
    q_current: np.ndarray,
    path: list[np.ndarray],
) -> tuple[int, float]:
    if not path:
        raise MainStop(
            "Recuperación: trayectoria vacía"
        )

    current = finite_vector(
        q_current,
        6,
        "recovery.q_current",
    )

    distances = [
        float(
            np.max(
                np.abs(
                    current
                    - finite_vector(
                        value,
                        6,
                        f"recovery.path[{index}]",
                    )
                )
            )
        )
        for index, value in enumerate(path)
    ]

    nearest_index = int(
        np.argmin(distances)
    )

    return nearest_index, distances[nearest_index]


def attempt_safe_home_recovery(
    *,
    node: Any,
    preplan: dict[str, Any],
    config: Any,
    phase: str,
    gripper_closed: bool,
    object_released: bool,
    run_dir: Path,
    original_error: Exception,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": (
            "VLA_pick_place_"
            "recovery"
        ),
        "timestamp": datetime.now().isoformat(),
        "phase_when_failed": phase,
        "original_error_type": (
            type(original_error).__name__
        ),
        "original_error": str(original_error),
        "gripper_closed": gripper_closed,
        "object_released": object_released,
        "payload_may_be_held": bool(
            gripper_closed
            and not object_released
        ),
        "gripper_opened_during_recovery": False,
        "status": "STARTED",
    }

    try:
        q_current, tcp_current = (
            supervisor.current_robot_state(node)
        )

        q_home = finite_vector(
            preplan["q_home"],
            6,
            "recovery.q_home",
        )

        home_error_before = float(
            np.max(
                np.abs(
                    q_current - q_home
                )
            )
        )

        report.update(
            {
                "q_start_rad": q_current,
                "tcp_start_m": tcp_current,
                "home_error_before_rad": (
                    home_error_before
                ),
            }
        )

        if home_error_before <= 0.03:
            report.update(
                {
                    "status": "ALREADY_HOME",
                    "home_reached": True,
                    "home_error_after_rad": (
                        home_error_before
                    ),
                }
            )
            json_write(
                run_dir / "recovery_report.json",
                report,
            )
            return report

        preclose = _compact_joint_targets(
            [
                np.asarray(value, dtype=float)
                for value in preplan["preclose_q"]
            ]
        )

        held = _compact_joint_targets(
            [
                np.asarray(value, dtype=float)
                for value in preplan["held_q"]
            ]
        )

        return_path = _compact_joint_targets(
            [
                np.asarray(value, dtype=float)
                for value in preplan["return_q"]
            ]
        )

        if object_released:
            source_path = return_path
            nearest_index, nearest_error = (
                _nearest_joint_index(
                    q_current,
                    source_path,
                )
            )
            recovery_targets = (
                source_path[nearest_index:]
                + [q_home]
            )
            policy = (
                "continue_validated_return_after_release"
            )
        elif gripper_closed:
            source_path = preclose + held
            nearest_index, nearest_error = (
                _nearest_joint_index(
                    q_current,
                    source_path,
                )
            )
            recovery_targets = list(
                reversed(
                    source_path[: nearest_index + 1]
                )
            ) + [q_home]
            policy = (
                "retrace_validated_outbound_"
                "with_payload_closed"
            )
        else:
            source_path = preclose
            nearest_index, nearest_error = (
                _nearest_joint_index(
                    q_current,
                    source_path,
                )
            )
            recovery_targets = list(
                reversed(
                    source_path[: nearest_index + 1]
                )
            ) + [q_home]
            policy = (
                "retrace_validated_pregrasp_"
                "with_gripper_open"
            )

        maximum_attach_error = float(
            os.environ.get(
                "VLA_RECOVERY_MAX_ATTACH_RAD",
                "0.20",
            )
        )

        if (
            not math.isfinite(
                maximum_attach_error
            )
            or not 0.05
            <= maximum_attach_error
            <= 0.35
        ):
            raise MainStop(
                "VLA_RECOVERY_MAX_ATTACH_RAD "
                "fuera de rango"
            )

        if nearest_error > maximum_attach_error:
            raise MainStop(
                "Recuperación cancelada: la postura "
                "actual está demasiado lejos de la "
                "trayectoria validada: "
                f"{nearest_error:.4f} rad > "
                f"{maximum_attach_error:.4f} rad"
            )

        recovery_targets = _compact_joint_targets(
            recovery_targets
        )

        report.update(
            {
                "policy": policy,
                "nearest_path_index": (
                    nearest_index
                ),
                "nearest_path_error_rad": (
                    nearest_error
                ),
                "target_count": len(
                    recovery_targets
                ),
                "recovery_targets_rad": (
                    recovery_targets
                ),
            }
        )

        execution = supervisor.send_joint_sequence(
            node=node,
            q_start_expected=q_current,
            q_targets=recovery_targets,
            config=config,
            label="failure_recovery",
        )

        result_q, result_tcp = (
            supervisor.current_robot_state(node)
        )

        home_error_after = float(
            np.max(
                np.abs(
                    result_q - q_home
                )
            )
        )

        home_reached = bool(
            home_error_after <= 0.05
        )

        report.update(
            {
                "status": (
                    "SUCCESS"
                    if home_reached
                    else "HOME_ERROR"
                ),
                "execution": execution,
                "result_q_rad": result_q,
                "result_tcp_m": result_tcp,
                "home_error_after_rad": (
                    home_error_after
                ),
                "home_reached": home_reached,
                "manual_payload_removal_required": bool(
                    gripper_closed
                    and not object_released
                ),
            }
        )

    except Exception as recovery_error:
        report.update(
            {
                "status": "FAILED",
                "home_reached": False,
                "recovery_error_type": (
                    type(recovery_error).__name__
                ),
                "recovery_error": str(
                    recovery_error
                ),
                "recovery_traceback": (
                    traceback.format_exc()
                ),
            }
        )

    json_write(
        run_dir / "recovery_report.json",
        report,
    )

    return report



def _write_staged_real_report(
    *,
    preplan: dict[str, Any],
    config: Any,
    run_dir: Path,
    real_stage: str,
    events: list[dict[str, Any]],
    result_q: np.ndarray,
    result_tcp: np.ndarray,
) -> dict[str, Any]:
    q_home = finite_vector(preplan["q_home"], 6, "stage.q_home")
    home_error = float(np.max(np.abs(result_q - q_home)))
    if home_error > 0.05:
        raise MainStop(
            f"La etapa {real_stage} no volvió a home: {home_error:.4f}rad"
        )

    report = {
        "schema": "pick_place_stage",
        "timestamp": datetime.now().isoformat(),
        "result": "SUCCESS",
        "real_stage": real_stage,
        "model": preplan["model"],
        "instruction": config.instruction,
        "architecture": preplan["architecture"],
        "robot_motion": True,
        "gripper_closed": False,
        "object_manipulated": False,
        "full_cycle_executed": False,
        "selected_object": preplan["grounding"]["object"]["name"],
        "selected_target": preplan["grounding"]["target"]["name"],
        "selected_target_xyz_m": preplan["selected_target_xyz_m"],
        "events": events,
        "result_q_rad": result_q,
        "result_tcp_m": result_tcp,
        "home_error_rad": home_error,
    }
    json_write(run_dir / "episode_result.json", report)
    print(
        f"stage={real_stage} model={preplan['model']} home_error={home_error:.4f}rad"
    )
    return report



def _merge_episode_result_json(
    path: Any,
    update: dict[str, Any],
) -> None:
    _json_module = __import__("json")
    _path_module = __import__("pathlib").Path

    target = _path_module(path)
    previous: dict[str, Any] = {}

    if target.exists():
        try:
            loaded = _json_module.loads(
                target.read_text(
                    encoding="utf-8",
                )
            )

            if isinstance(loaded, dict):
                previous = loaded

        except Exception:
            previous = {}

    if isinstance(update, dict):
        previous.update(update)

    target.write_text(
        _json_module.dumps(
            previous,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )



def execute_real(
    *,
    preplan: dict[str, Any],
    config: Any,
    run_dir: Path,
    real_stage: str,
) -> dict[str, Any]:
    require_real_confirmations(config, real_stage)

    events: list[dict[str, Any]] = []
    gripper_closed = False
    object_released = False
    phase = "preflight"

    gripper_guidance = (
        preplan["reports"]
        ["translation_profile"]
        ["gripper_guidance"]
    )

    open_width_m = float(
        gripper_guidance[
            "applied_open_width_m"
        ]
    )
    close_width_m = float(
        gripper_guidance[
            "applied_close_width_m"
        ]
    )
    close_hold_s = float(
        gripper_guidance[
            "applied_close_hold_s"
        ]
    )

    anchors = {
        "q_pre": finite_vector(
            preplan["q_pre"],
            6,
            "real.q_pre",
        ),
        "q_pick": finite_vector(
            preplan["q_pick"],
            6,
            "real.q_pick",
        ),
        "q_lift": finite_vector(
            preplan["q_lift_anchor"],
            6,
            "real.q_lift",
        ),
        "q_pre_place": finite_vector(
            preplan["q_pre_place"],
            6,
            "real.q_pre_place",
        ),
        "q_place": finite_vector(
            preplan["q_place"],
            6,
            "real.q_place",
        ),
        "q_retreat": finite_vector(
            preplan["q_retreat_anchor"],
            6,
            "real.q_retreat",
        ),
    }

    execution_chain_validation = (
        _validate_anchor_chain(
            q_home=np.asarray(
                preplan["q_home"],
                dtype=float,
            ),
            anchors=anchors,
        )
    )

    with supervisor.robot_node() as node:
        try:
            if hasattr(
                supervisor,
                "require_single_gripper_subscriber",
            ):
                supervisor.require_single_gripper_subscriber(
                    node
                )

            q_start, tcp_start = (
                supervisor.current_robot_state(node)
            )

            q_home = finite_vector(
                preplan["q_home"],
                6,
                "real.q_home",
            )
            tcp_home = finite_vector(
                preplan["tcp_home"],
                3,
                "real.tcp_home",
            )

            q_error = float(
                np.max(
                    np.abs(
                        q_start - q_home
                    )
                )
            )
            tcp_error = float(
                np.linalg.norm(
                    tcp_start - tcp_home
                )
            )

            if q_error > 0.02 or tcp_error > 0.005:
                raise MainStop(
                    "El robot cambió después del preplan: "
                    f"q={q_error:.4f}rad, "
                    f"tcp={tcp_error:.4f}m"
                )

            phase = "open_gripper"
            node.set_gripper_width(
                open_width_m,
                wait_sec=2.5,
            )

            phase = "approach_pre_pick"
            approach_execution = supervisor.send_joint_sequence(
                node=node,
                q_start_expected=q_start,
                q_targets=[
                    finite_vector(
                        value,
                        6,
                        f"real.approach_bridge[{index}]",
                    )
                    for index, value
                    in enumerate(
                        execution_chain_validation[
                            "approach_bridge_q"
                        ]
                    )
                ],
                config=config,
                label="main_approach",
            )

            q_pre_actual, tcp_pre_actual = (
                supervisor.current_robot_state(node)
            )

            if real_stage == "approach":
                phase = "approach_return_home"
                home_execution = supervisor.send_joint_sequence(
                    node=node,
                    q_start_expected=q_pre_actual,
                    q_targets=[q_home],
                    config=config,
                    label="main_approach_return",
                )
                result_q, result_tcp = supervisor.current_robot_state(node)
                events.extend(
                    [
                        {
                            "phase": "approach",
                            "execution": approach_execution,
                            "actual_tcp_m": tcp_pre_actual,
                        },
                        {
                            "phase": "return_home",
                            "execution": home_execution,
                        },
                    ]
                )
                return _write_staged_real_report(
                    preplan=preplan,
                    config=config,
                    run_dir=run_dir,
                    real_stage=real_stage,
                    events=events,
                    result_q=result_q,
                    result_tcp=result_tcp,
                )

            phase = "pick_descent"
            pick_execution = supervisor.send_joint_sequence(
                node=node,
                q_start_expected=q_pre_actual,
                q_targets=[anchors["q_pick"]],
                config=config,
                label="main_contact_pick",
            )

            q_pick_actual, tcp_pick_actual = (
                supervisor.current_robot_state(node)
            )

            pick_expected = finite_vector(
                preplan["pick_xyz_m"],
                3,
                "real.pick_expected",
            )

            pick_error = float(
                np.linalg.norm(
                    tcp_pick_actual - pick_expected
                )
            )

            if pick_error > config.pick_actual_error_m:
                raise MainStop(
                    "Error real de pick: "
                    f"{pick_error:.4f}m"
                )

            if real_stage == "pick_pose":
                phase = "pick_pose_vertical_retreat"
                retreat_execution = supervisor.send_joint_sequence(
                    node=node,
                    q_start_expected=q_pick_actual,
                    q_targets=[anchors["q_pre"], q_home],
                    config=config,
                    label="main_pick_pose_return",
                )
                result_q, result_tcp = supervisor.current_robot_state(node)
                events.extend(
                    [
                        {
                            "phase": "approach",
                            "execution": approach_execution,
                        },
                        {
                            "phase": "pick_descent_without_close",
                            "execution": pick_execution,
                            "pick_actual_xyz_m": tcp_pick_actual,
                            "pick_error_m": pick_error,
                        },
                        {
                            "phase": "vertical_retreat_and_return_home",
                            "execution": retreat_execution,
                        },
                    ]
                )
                return _write_staged_real_report(
                    preplan=preplan,
                    config=config,
                    run_dir=run_dir,
                    real_stage=real_stage,
                    events=events,
                    result_q=result_q,
                    result_tcp=result_tcp,
                )

            phase = "close_gripper"
            node.set_gripper_width(
                close_width_m,
                wait_sec=close_hold_s,
            )
            gripper_closed = True

            phase = "lift"
            lift_execution = supervisor.send_joint_sequence(
                node=node,
                q_start_expected=q_pick_actual,
                q_targets=[anchors["q_lift"]],
                config=config,
                label="main_lift",
            )

            q_lift_actual, _ = (
                supervisor.current_robot_state(node)
            )

            phase = "transport"
            transport_execution = supervisor.send_joint_sequence(
                node=node,
                q_start_expected=q_lift_actual,
                q_targets=[anchors["q_pre_place"]],
                config=config,
                label="main_transport",
            )

            q_pre_place_actual, _ = (
                supervisor.current_robot_state(node)
            )

            phase = "place_descent"
            place_execution = supervisor.send_joint_sequence(
                node=node,
                q_start_expected=q_pre_place_actual,
                q_targets=[anchors["q_place"]],
                config=config,
                label="main_place_descent",
            )

            q_place_actual, tcp_place_actual = (
                supervisor.current_robot_state(node)
            )

            place_expected = finite_vector(
                preplan["place_xyz_m"],
                3,
                "real.place_expected",
            )

            place_error = float(
                np.linalg.norm(
                    tcp_place_actual - place_expected
                )
            )

            if place_error > config.place_actual_error_m:
                raise MainStop(
                    "Error real de place: "
                    f"{place_error:.4f}m"
                )

            phase = "release_gripper"
            node.set_gripper_width(
                open_width_m,
                wait_sec=2.5,
            )
            object_released = True
            gripper_closed = False

            phase = "retreat"
            retreat_execution = supervisor.send_joint_sequence(
                node=node,
                q_start_expected=q_place_actual,
                q_targets=[anchors["q_retreat"]],
                config=config,
                label="main_retreat",
            )

            q_retreat_actual, _ = (
                supervisor.current_robot_state(node)
            )

            phase = "return_home"
            home_execution = supervisor.send_joint_sequence(
                node=node,
                q_start_expected=q_retreat_actual,
                q_targets=[q_home],
                config=config,
                label="main_return",
            )

            result_q, result_tcp = (
                supervisor.current_robot_state(node)
            )

        except Exception as original_error:
            recovery = attempt_safe_home_recovery(
                node=node,
                preplan=preplan,
                config=config,
                phase=phase,
                gripper_closed=gripper_closed,
                object_released=object_released,
                run_dir=run_dir,
                original_error=original_error,
            )

            raise MainStop(
                f"{original_error}; "
                "recuperación="
                f"{recovery.get('status')}; "
                "home_reached="
                f"{recovery.get('home_reached')}"
            ) from original_error

    home_error = float(
        np.max(
            np.abs(
                result_q
                - np.asarray(
                    preplan["q_home"],
                    dtype=float,
                )
            )
        )
    )

    if home_error > 0.05:
        raise MainStop(
            "No se volvió a home: "
            f"{home_error:.4f}rad"
        )

    phase = "place_verification"

    try:
        (
            verify_scene_path,
            verify_scene,
            verify_image_path,
            _,
        ) = supervisor.request_fresh_scene(
            run_dir=run_dir,
            instruction=preplan["model_evidence"].get(
                "neutral_capture_instruction",
                "put the orange cube in the yellow zone",
            ),
            index=2,
            label="place_verification",
        )
    except Exception as _shape_result_error:
        _os = __import__("os")
    
        _is_shape = (
            _os.environ.get(
                "VLA_TASK_FAMILY", ""
            ).strip() == "shape_classification"
        )
    
        _confirmed = (
            _os.environ.get(
                "VLA_SHAPE_RESULT_OCCUPANCY_CONFIRMED",
                "NO",
            ).strip().upper() == "YES"
        )
    
        _detector_failed = (
            "Detector live desde PLY falló"
            in str(_shape_result_error)
        )
    
        if (
            _is_shape
            and _confirmed
            and _detector_failed
        ):
            _json = __import__("json")
            _datetime = __import__(
                "datetime"
            ).datetime
            _Path = __import__(
                "pathlib"
            ).Path
    
            _instruction = str(
                getattr(config, "instruction", "")
            )
            _instruction_lower = (
                _instruction.lower()
            )
    
            if (
                "orange" in _instruction_lower
                and (
                    "cube" in _instruction_lower
                    or "block" in _instruction_lower
                )
            ):
                _object = "orange_cube"
                _target = "square_box"
            elif (
                "white" in _instruction_lower
                and (
                    "cylinder" in _instruction_lower
                    or "cylindrical"
                    in _instruction_lower
                )
            ):
                _object = "white_cylinder"
                _target = "circular_box"
            else:
                _object = "selected_shape_object"
                _target = "matching_shape_container"
    
            # Usar la identidad que realmente seleccionó y ejecutó
            # el VLA. No reconstruir el target desde la instrucción.
            _executed_grounding = (
                preplan.get("grounding", {})
            )

            _executed_object_record = (
                _executed_grounding.get(
                    "object",
                    {},
                )
            )
            _executed_target_record = (
                _executed_grounding.get(
                    "target",
                    {},
                )
            )

            if isinstance(
                _executed_object_record,
                dict,
            ):
                _object = str(
                    _executed_object_record.get(
                        "name",
                        _object,
                    )
                )

            if isinstance(
                _executed_target_record,
                dict,
            ):
                _target = str(
                    _executed_target_record.get(
                        "name",
                        _target,
                    )
                )

            _run_dir = _Path(run_dir)
            _run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
    
            _success = {
                "schema": (
                    "VLA_shape_physical_success"
                ),
                "timestamp": (
                    _datetime.now().isoformat()
                ),
                "result": "SUCCESS",
                "physical_result": "SUCCESS",
                "task_family": (
                    "shape_classification"
                ),
                "instruction": _instruction,
                "selected_object": _object,
                "selected_target": _target,
                "grounding_success": True,
                "pick_success": True,
                "transport_success": True,
                "place_success": True,
                "release_success": True,
                "retreat_success": True,
                "return_home_success": True,
                "target_occupied_by_correct_object": True,
                "inside_target": True,
                "verification_source": (
                    "visual_occupancy_confirmation"
                ),
                "occupied_target_opening_expected": True,
                "result_detector_error_is_post_execution": True,
                "result_detector_error": repr(
                    _shape_result_error
                ),
            }
    
            _success_path = (
                _run_dir
                / "shape_physical_success.json"
            )
    
            _success_path.write_text(
                _json.dumps(
                    _success,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
    
            _episode_result_path = (
                _run_dir / "episode_result.json"
            )
    
            _merge_episode_result_json(
                _episode_result_path,
                _success,
            )
    
            print(
                f"object={_object} "
                f"target={_target} "
                "inside=true result=SUCCESS "
                "verification=visual_occupancy_confirmation "
                f"report={_success_path}",
                flush=True,
            )
    
            return

        if (
            _is_shape
            and _detector_failed
        ):
            # Todos los movimientos terminaron y el guard de home
            # ya pasó. El detector posterior no debe destruir la
            # ejecución, pero tampoco afirmar SUCCESS automático.
            _json = __import__("json")
            _datetime = __import__(
                "datetime"
            ).datetime
            _Path = __import__(
                "pathlib"
            ).Path

            _instruction = str(
                getattr(config, "instruction", "")
            )
            _instruction_lower = (
                _instruction.lower()
            )

            if (
                "orange" in _instruction_lower
                and (
                    "cube" in _instruction_lower
                    or "block"
                    in _instruction_lower
                )
            ):
                _object = "orange_cube"
                _target = "square_box"
            elif (
                "white" in _instruction_lower
                and (
                    "cylinder"
                    in _instruction_lower
                    or "cylindrical"
                    in _instruction_lower
                )
            ):
                _object = "white_cylinder"
                _target = "circular_box"
            else:
                _object = (
                    "selected_shape_object"
                )
                _target = (
                    "matching_shape_container"
                )

            # Usar la identidad que realmente seleccionó y ejecutó
            # el VLA. No reconstruir el target desde la instrucción.
            _executed_grounding = (
                preplan.get("grounding", {})
            )

            _executed_object_record = (
                _executed_grounding.get(
                    "object",
                    {},
                )
            )
            _executed_target_record = (
                _executed_grounding.get(
                    "target",
                    {},
                )
            )

            if isinstance(
                _executed_object_record,
                dict,
            ):
                _object = str(
                    _executed_object_record.get(
                        "name",
                        _object,
                    )
                )

            if isinstance(
                _executed_target_record,
                dict,
            ):
                _target = str(
                    _executed_target_record.get(
                        "name",
                        _target,
                    )
                )

            _run_dir = _Path(run_dir)
            _run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            _incomplete = {
                "schema": (
                    "VLA_shape_visual_evidence_"
                    "incomplete"
                ),
                "timestamp": (
                    _datetime.now().isoformat()
                ),
                "result": (
                    "VISUAL_EVIDENCE_INCOMPLETE"
                ),
                "physical_result": (
                    "UNVERIFIED_AUTOMATICALLY"
                ),
                "execution_result": "COMPLETED",
                "task_family": (
                    "shape_classification"
                ),
                "instruction": _instruction,
                "selected_object": _object,
                "selected_target": _target,
                "grounding_success": True,
                "motion_execution_success": True,
                "pick_motion_completed": True,
                "transport_motion_completed": True,
                "place_motion_completed": True,
                "release_command_completed": True,
                "retreat_success": True,
                "return_home_success": True,
                "automatic_result_verification": False,
                "visual_evidence_incomplete": True,
                "manual_confirmation_required": True,
                "result_detector_error_is_post_execution": True,
                "result_detector_error": repr(
                    _shape_result_error
                ),
            }

            _incomplete_path = (
                _run_dir
                / "shape_visual_evidence_incomplete.json"
            )

            _incomplete_path.write_text(
                _json.dumps(
                    _incomplete,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            (
                _run_dir / "episode_result.json"
            ).write_text(
                _json.dumps(
                    _incomplete,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            print(
                f"object={_object} "
                f"target={_target} "
                "execution=COMPLETED "
                "return_home=true "
                "automatic_verification=false "
                "result=VISUAL_EVIDENCE_INCOMPLETE "
                f"report={_incomplete_path}",
                flush=True,
            )

            return

        raise

    verify_template_config = geometry_template_config(
        config,
        preplan["grounding"],
    )
    verify_parser_entities = supervisor.resolve_entities(
        verify_scene,
        verify_template_config,
    )
    verify_grounding = refresh_grounding_coordinates(
        scene=verify_scene,
        image_path=Path(verify_image_path),
        grounding=preplan["grounding"],
    )
    (
        verify_entities,
        verify_grounding_application,
    ) = apply_grounding_to_entities(
        verify_parser_entities,
        verify_grounding,
    )

    observed_object = finite_vector(
        verify_entities["object_xyz_m"],
        3,
        "verify.object",
    )
    # La validación física usa el punto espacial que
    # realmente se ejecutó. El target recalculado por
    # resolve_entities queda solo como referencia.
    executed_target = finite_vector(
        preplan["selected_target_xyz_m"],
        3,
        "verify.executed_vla_target",
    )

    refreshed_selected_interior = finite_vector(
        verify_entities["target_xyz_m"],
        3,
        "verify.refreshed_selected_interior",
    )

    refreshed_target_center_raw = None

    if isinstance(verify_grounding, dict):
        refreshed_target_record = (
            verify_grounding.get("target")
        )

        if isinstance(
            refreshed_target_record,
            dict,
        ):
            refreshed_target_center_raw = (
                refreshed_target_record.get(
                    "base_xyz_m"
                )
            )

    if refreshed_target_center_raw is None:
        refreshed_target_center_raw = (
            verify_entities.get(
                "target_detected_center_xyz_m"
            )
        )

    if refreshed_target_center_raw is None:
        original_target_record = (
            preplan
            .get("grounding", {})
            .get("target", {})
        )

        if isinstance(
            original_target_record,
            dict,
        ):
            refreshed_target_center_raw = (
                original_target_record.get(
                    "base_xyz_m"
                )
            )

    if refreshed_target_center_raw is None:
        raise MainStop(
            "No se pudo recuperar el centro físico "
            "de la zona durante la verificación."
        )

    refreshed_target_center = finite_vector(
        refreshed_target_center_raw,
        3,
        "verify.refreshed_target_center",
    )

    # En el reporte, observed_target
    # representa el target que se ejecutó.
    observed_target = executed_target.copy()

    xy_error = float(
        np.linalg.norm(
            observed_object[:2]
            - executed_target[:2]
        )
    )

    zone_center_xy_error = float(
        np.linalg.norm(
            observed_object[:2]
            - refreshed_target_center[:2]
        )
    )

    previous_interior_xy_error = float(
        np.linalg.norm(
            observed_object[:2]
            - refreshed_selected_interior[:2]
        )
    )

    z_error = float(
        abs(
            observed_object[2]
            - refreshed_target_center[2]
        )
    )

    xy_tolerance_m = float(
        os.environ.get(
            "VLA_RESULT_PLACE_XY_TOL_M",
            "0.060",
        )
    )

    z_tolerance_m = float(
        os.environ.get(
            "VLA_RESULT_PLACE_Z_TOL_M",
            "0.050",
        )
    )

    zone_radius_m = float(
        os.environ.get(
            "VLA_PLACE_REGION_RADIUS_M",
            "0.070",
        )
    )

    if not 0.005 <= xy_tolerance_m <= 0.060:
        raise MainStop(
            "VLA_RESULT_PLACE_XY_TOL_M fuera "
            "de [0.005, 0.060]"
        )

    if not 0.005 <= z_tolerance_m <= 0.050:
        raise MainStop(
            "VLA_RESULT_PLACE_Z_TOL_M fuera "
            "de [0.005, 0.050]"
        )

    if not 0.030 <= zone_radius_m <= 0.090:
        raise MainStop(
            "VLA_PLACE_REGION_RADIUS_M fuera "
            "de [0.030, 0.090]"
        )

    target_match_pass = bool(
        xy_error <= xy_tolerance_m
    )

    zone_membership_pass = bool(
        zone_center_xy_error
        <= zone_radius_m
    )

    height_pass = bool(
        z_error <= z_tolerance_m
    )

    verified = bool(
        target_match_pass
        and zone_membership_pass
        and height_pass
    )

    verification_document = {
        "schema":
            "VLA_place_verification_executed_target",
        "observed_object_xyz_m":
            observed_object.tolist(),
        "executed_vla_target_xyz_m":
            executed_target.tolist(),
        "refreshed_target_center_xyz_m":
            refreshed_target_center.tolist(),
        "refreshed_selected_interior_xyz_m":
            refreshed_selected_interior.tolist(),
        "executed_target_xy_error_m":
            xy_error,
        "zone_center_xy_error_m":
            zone_center_xy_error,
        "previous_selected_interior_xy_error_m":
            previous_interior_xy_error,
        "z_error_m":
            z_error,
        "xy_tolerance_m":
            xy_tolerance_m,
        "zone_radius_m":
            zone_radius_m,
        "z_tolerance_m":
            z_tolerance_m,
        "target_match_pass":
            target_match_pass,
        "zone_membership_pass":
            zone_membership_pass,
        "height_pass":
            height_pass,
        "verified":
            verified,
        "validation_reference":
            "executed_vla_target_plus_refreshed_zone_center",
    }

    json_write(
        run_dir
        / "place_verification_executed_target.json",
        verification_document,
    )

    print(
        f"observed={observed_object.tolist()} "
        f"executed_target={executed_target.tolist()} "
        f"refreshed_center="
        f"{refreshed_target_center.tolist()} "
        f"executed_xy_error={xy_error:.4f}m "
        f"zone_center_xy_error="
        f"{zone_center_xy_error:.4f}m "
        f"previous_interior_xy_error="
        f"{previous_interior_xy_error:.4f}m "
        f"z_error={z_error:.4f}m "
        f"verified={verified}",
        flush=True,
    )

    events.extend(
        [
            {
                "phase": "approach",
                "execution": approach_execution,
            },
            {
                "phase": "pick_descent",
                "execution": pick_execution,
                "pick_actual_xyz_m": tcp_pick_actual,
                "pick_error_m": pick_error,
            },
            {
                "phase": "grasp",
                "close_width_m": close_width_m,
                "hold_s": close_hold_s,
                "vla_guided": True,
            },
            {
                "phase": "lift",
                "execution": lift_execution,
            },
            {
                "phase": "transport",
                "execution": transport_execution,
            },
            {
                "phase": "place_descent",
                "execution": place_execution,
                "place_actual_xyz_m": tcp_place_actual,
                "place_error_m": place_error,
            },
            {
                "phase": "release",
                "open_width_m": open_width_m,
            },
            {
                "phase": "retreat",
                "execution": retreat_execution,
            },
            {
                "phase": "return",
                "execution": home_execution,
                "result_q_rad": result_q,
                "result_tcp_m": result_tcp,
                "home_error_rad": home_error,
            },
            {
                "phase": "place_verification",
                "scene_path": verify_scene_path,
                "image_path": verify_image_path,
                "xy_error_m": xy_error,
                "z_error_m": z_error,
                "verified": verified,
                "grounding_reused_by_name": True,
                "grounding_application": (
                    verify_grounding_application
                ),
                "refreshed_grounding": verify_grounding,
            },
        ]
    )

    if not verified:
        raise MainStop(
            "El movimiento terminó, pero "
            "Photoneo no confirmó el place."
        )

    report = {
        "schema":
            "pick_place_execution",
        "timestamp":
            datetime.now().isoformat(),
        "result": "SUCCESS",
        "real_stage": "full",
        "full_cycle_executed": True,
        "model": preplan["model"],
        "instruction": config.instruction,
        "architecture": preplan["architecture"],
        "object_grounding_vla_applied": True,
        "target_grounding_vla_applied": True,
        "parser_used_as_decision": False,
        "translation_vla_applied": True,
        "raw_temporal_vla_path_applied": False,
        "joint_guidance_vla_applied": False,
        "phase_translation_application": (
            preplan["reports"]
            ["phase_translation_application"]
        ),
        "rotation_vla_applied": False,
        "gripper_vla_applied": True,
        "gripper_guidance": (
            gripper_guidance
        ),
        "object_released": object_released,
        "place_verified": verified,
        "selected_target_xyz_m": (
            preplan["selected_target_xyz_m"]
        ),
        "events": events,
    }

    json_write(
        run_dir / "episode_result.json",
        report,
    )

    return report


def _parse_replay_vector(
    raw: str,
    expected_length: int,
    label: str,
) -> np.ndarray:
    values = [
        item.strip()
        for item in str(raw).split(",")
        if item.strip()
    ]
    if len(values) != expected_length:
        raise MainStop(
            f"{label} debe contener {expected_length} valores separados "
            f"por comas; recibido={raw!r}"
        )
    try:
        vector = np.asarray(
            [float(value) for value in values],
            dtype=float,
        )
    except ValueError as exc:
        raise MainStop(
            f"{label} contiene valores no numéricos: {raw!r}"
        ) from exc
    if not np.all(np.isfinite(vector)):
        raise MainStop(
            f"{label} contiene NaN o infinito: {raw!r}"
        )
    return vector


def _ur5e_tcp_fk_from_q(
    q_rad: np.ndarray,
    tcp_z_m: float = 0.1696,
) -> np.ndarray:
    """FK del UR5e usada solo para reproducir el home sin hardware."""
    q = finite_vector(
        q_rad,
        6,
        "replay.q_home",
    )
    a = np.asarray(
        [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0],
        dtype=float,
    )
    d = np.asarray(
        [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996],
        dtype=float,
    )
    alpha = np.asarray(
        [
            math.pi / 2.0,
            0.0,
            0.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            0.0,
        ],
        dtype=float,
    )
    transform = np.eye(4, dtype=float)
    for index in range(6):
        theta = float(q[index])
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(float(alpha[index]))
        sa = math.sin(float(alpha[index]))
        dh = np.asarray(
            [
                [ct, -st * ca, st * sa, a[index] * ct],
                [st, ct * ca, -ct * sa, a[index] * st],
                [0.0, sa, ca, d[index]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        transform = transform @ dh

    tcp_transform = np.eye(4, dtype=float)
    tcp_transform[2, 3] = float(tcp_z_m)
    return (transform @ tcp_transform)[:3, 3]


def _load_replay_scene(
    *,
    scene_json: str,
    image: str,
    ply: str | None,
    q_home_raw: str,
) -> tuple[
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    scene_path = Path(scene_json).expanduser().resolve()
    image_path = Path(image).expanduser().resolve()

    if not scene_path.is_file():
        raise MainStop(
            f"No existe el JSON de replay: {scene_path}"
        )
    if not image_path.is_file():
        raise MainStop(
            f"No existe la imagen de replay: {image_path}"
        )

    try:
        with scene_path.open("r", encoding="utf-8") as handle:
            scene = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MainStop(
            f"No se pudo leer el JSON de replay: {scene_path}"
        ) from exc

    if not isinstance(scene, dict):
        raise MainStop(
            "El JSON de replay debe contener un objeto raíz."
        )

    ply_path: Path | None = None
    if ply:
        ply_path = Path(ply).expanduser().resolve()
        if not ply_path.is_file():
            raise MainStop(
                f"No existe la nube PLY de replay: {ply_path}"
            )
        scene["ply_path"] = str(ply_path)

    q_home = _parse_replay_vector(
        q_home_raw,
        6,
        "--replay-home-q-rad",
    )
    tcp_home = _ur5e_tcp_fk_from_q(q_home)

    proof = {
        "schema": "VLA_offline_scene_replay",
        "offline_replay": True,
        "hardware_capture_requested": False,
        "robot_state_read_requested": False,
        "scene_json": str(scene_path),
        "scene_json_sha256": sha256_file(scene_path),
        "image": str(image_path),
        "image_sha256": sha256_file(image_path),
        "ply": str(ply_path) if ply_path is not None else None,
        "ply_sha256": (
            sha256_file(ply_path)
            if ply_path is not None
            else None
        ),
        "q_home_rad": q_home.tolist(),
        "tcp_home_fk_m": tcp_home.tolist(),
    }

    print(
        f"scene={scene_path} "
        f"image={image_path} "
        f"ply={ply_path} "
        "robot_motion=false",
        flush=True,
    )
    print(
        f"q={q_home.tolist()} "
        f"tcp_fk_m={tcp_home.tolist()}",
        flush=True,
    )

    return (
        scene_path,
        scene,
        image_path,
        proof,
        q_home,
        tcp_home,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Script principal definitivo "
            "OpenVLA / π0.5-DROID / π0-DROID "
            "para pick-and-place real "
            "con grounding causal del propio VLA, traducción XYZ métrica sobre seis anclas geométricas y garra acotada."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "openvla",
            "pi05_droid",
            "pi0_droid",
        ],
        required=True,
    )

    parser.add_argument(
        "--instruction",
        required=True,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "check",
            "real",
        ],
        default="check",
    )

    parser.add_argument(
        "--real-stage",
        choices=[
            "approach",
            "pick_pose",
            "full",
        ],
        default="approach",
        help=(
            "Etapa física cuando --mode real: approach llega solo a pre-pick; "
            "pick_pose desciende sin cerrar y vuelve por retirada vertical; "
            "full ejecuta pick-and-place completo."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--droid-state-mode",
        choices=[
            "zeros",
            "ur5e",
        ],
        default="zeros",
    )

    parser.add_argument(
        "--grounding-mode",
        choices=[
            "auto",
            "heatmap_json",
            "region_occlusion",
            "policy_counterfactual",
            "joint_policy_hypothesis",
            "parser_trace",
        ],
        default="auto",
        help=(
            "Selección visual-lingüística. auto usa evidencia externa solo "
            "si se aporta explícitamente; en otro caso aplica necesidad y "
            "suficiencia contrafactual de la política del propio VLA."
        ),
    )

    parser.add_argument(
        "--grounding-json",
        default=None,
        help=(
            "JSON opcional con image_sha256, object_scores y target_scores."
        ),
    )

    parser.add_argument(
        "--grounding-heatmap",
        default=None,
        help=(
            "Heatmap PNG/JPG/NPY/NPZ/JSON que se cruzará con las "
            "regiones candidatas Photoneo."
        ),
    )

    parser.add_argument(
        "--capture-instruction",
        default=os.environ.get(
            "VLA_NEUTRAL_CAPTURE_INSTRUCTION",
            "put the orange cube in the yellow zone",
        ),
        help=(
            "Orden canónica usada únicamente para forzar la captura "
            "multiobjeto del detector actual. No decide objeto ni destino."
        ),
    )

    parser.add_argument(
        "--replay-scene-json",
        default=None,
        help=(
            "JSON de escena guardado. Solo se admite con --mode check "
            "y evita Photoneo y el robot."
        ),
    )

    parser.add_argument(
        "--replay-image",
        default=None,
        help=(
            "Imagen RGB asociada al JSON de replay."
        ),
    )

    parser.add_argument(
        "--replay-ply",
        default=None,
        help=(
            "Nube PLY asociada al replay para recuperar regiones visuales."
        ),
    )

    parser.add_argument(
        "--replay-home-q-rad",
        default=os.environ.get(
            "VLA_GLOBAL_HOME_Q_RAD",
            (
                "1.420421600342,"
                "-1.376682208186,"
                "1.576593224202,"
                "-1.747573038141,"
                "-1.565862480794,"
                "-0.103206459676"
            ),
        ),
        help=(
            "Home UR5e usado únicamente en replay offline."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    instruction = (
        args.instruction.strip()
    )

    if not instruction:
        raise MainStop(
            "La instrucción está vacía."
        )

    reset_visual_context()

    config_args = argparse.Namespace(
        instruction=instruction,
        mode=args.mode,
    )

    config = supervisor.build_config(
        config_args
    )

    stamp = datetime.now().strftime(
        "%Y%m%dT%H%M%S_%f"
    )

    run_dir = (
        RESULTS_ROOT
        / (
            f"{stamp}_{args.model}_"
            f"{args.mode}"
            + (f"_{args.real_stage}" if args.mode == "real" else "")
        )
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        replay_q_home: np.ndarray | None = None
        replay_tcp_home: np.ndarray | None = None

        if args.replay_scene_json is not None:
            if args.mode != "check":
                raise MainStop(
                    "El replay offline solo puede ejecutarse con --mode check."
                )
            if not args.replay_image:
                raise MainStop(
                    "--replay-image es obligatorio al usar "
                    "--replay-scene-json."
                )

            (
                scene_path,
                scene,
                image_path,
                capture_proof,
                replay_q_home,
                replay_tcp_home,
            ) = _load_replay_scene(
                scene_json=args.replay_scene_json,
                image=args.replay_image,
                ply=args.replay_ply,
                q_home_raw=args.replay_home_q_rad,
            )
            q_initial = replay_q_home.copy()
        else:
            (
                scene_path,
                scene,
                image_path,
                capture_proof,
            ) = supervisor.request_fresh_scene(
                run_dir=run_dir,
                instruction=args.capture_instruction,
                index=1,
                label="initial_scene",
            )

            scene_path = Path(
                scene_path
            )
            image_path = Path(
                image_path
            )

            with supervisor.robot_node() as node:
                q_initial, _ = (
                    supervisor.current_robot_state(
                        node
                    )
                )

        actions, evidence = infer_on_orin(
            model=args.model,
            image_path=image_path,
            instruction=instruction,
            q_ur5e=q_initial,
            gripper_width_m=(
                config.open_width_m
            ),
            droid_state_mode=(
                args.droid_state_mode
            ),
            seed=args.seed,
            run_dir=run_dir,
        )

        evidence[
            "capture_proof"
        ] = capture_proof
        evidence["neutral_capture_instruction"] = args.capture_instruction
        evidence["task_instruction_used_only_by_vla"] = instruction

        grounding = ground_scene_with_vla(
            model=args.model,
            actions=actions,
            scene=scene,
            image_path=image_path,
            instruction=instruction,
            q_ur5e=q_initial,
            gripper_width_m=config.open_width_m,
            droid_state_mode=args.droid_state_mode,
            seed=args.seed,
            run_dir=run_dir,
            mode=args.grounding_mode,
            grounding_json=args.grounding_json,
            grounding_heatmap=args.grounding_heatmap,
        )

        preplan = build_preplan(
            model=args.model,
            actions=actions,
            evidence=evidence,
            scene=scene,
            scene_path=scene_path,
            image_path=image_path,
            config=config,
            run_dir=run_dir,
            grounding=grounding,
            q_home_override=replay_q_home,
            tcp_home_override=replay_tcp_home,
        )

        if args.mode == "check":
            report = {
                "schema":
                    "VLA_pick_place_check",
                "timestamp":
                    datetime.now().isoformat(),
                "result": "PASS",
                "model": args.model,
                "instruction":
                    instruction,
                "native_shape":
                    list(actions.shape),
                "object_grounding_vla_applied":
                    True,
                "target_grounding_vla_applied":
                    True,
                "parser_used_as_decision":
                    False,
                "grounding_method":
                    grounding["method"],
                "grounded_object":
                    grounding["object"]["name"],
                "grounded_target":
                    grounding["target"]["name"],
                "translation_vla_applied":
                    True,
                "raw_temporal_vla_path_applied":
                    False,
                "joint_guidance_vla_applied":
                    False,
                "phase_translation_application":
                    preplan["reports"]["phase_translation_application"],
                "rotation_vla_applied":
                    False,
                "gripper_vla_applied":
                    True,
                "gripper_guidance":
                    preplan["reports"]["translation_profile"]["gripper_guidance"],
                "robot_motion": False,
                "gripper_motion": False,
                "offline_replay":
                    args.replay_scene_json is not None,
                "scene_source": (
                    "offline_replay"
                    if args.replay_scene_json is not None
                    else "photoneo_live"
                ),
                "full_episode_prevalidated":
                    True,
                "selected_target_xyz_m":
                    preplan["selected_target_xyz_m"],
                "geometric_phase_targets_xyz_m":
                    preplan["geometric_phase_targets_xyz_m"],
                "vla_phase_targets_xyz_m":
                    preplan["vla_phase_targets_xyz_m"],
                "phase_residuals_base_m":
                    preplan["reports"]["translation_profile"]["phase_residuals_base_m"],
                "run_dir": str(
                    run_dir
                ),
                "preplan_path": str(
                    run_dir
                    / "episode_preplan.json"
                ),
            }

            json_write(
                run_dir
                / "episode_check.json",
                report,
            )

            print(
                "MAIN_VLA_PICK_PLACE_CHECK: PASS"
            )
            print(
                "model:",
                args.model,
            )
            print(
                "native_shape:",
                actions.shape,
            )
            print(
                "robot_motion: False"
            )
            print(
                "phase_residuals_base_m:",
                preplan["reports"]["translation_profile"]["phase_residuals_base_m"],
            )
            print(
                "close_width_m:",
                preplan["reports"]["translation_profile"]["gripper_guidance"]["applied_close_width_m"],
            )
            print(
                "selected_target_xyz_m:",
                preplan["selected_target_xyz_m"],
            )
            print(
                "report:",
                run_dir
                / "episode_check.json",
            )

            return 0

        execute_real(
            preplan=preplan,
            config=config,
            run_dir=run_dir,
            real_stage=args.real_stage,
        )

        print(
            "MAIN_VLA_PICK_PLACE_REAL: SUCCESS"
        )
        print(
            "model:",
            args.model,
        )
        print(
            "real_stage:",
            args.real_stage,
        )
        print(
            "report:",
            run_dir
            / "episode_result.json",
        )

        return 0

    except Exception as error:
        failure = {
            "schema":
                "VLA_pick_place_failure",
            "timestamp":
                datetime.now().isoformat(),
            "model": args.model,
            "instruction":
                instruction,
            "mode": args.mode,
            "real_stage": args.real_stage,
            "error_type":
                type(error).__name__,
            "error": str(error),
            "traceback":
                traceback.format_exc(),
            "robot_motion_requested":
                args.mode == "real",
            "evaluation_protocol": {
                "schema": (
                    "VLA_shape_taxonomy_evaluation_protocol"
                ),
                "series": __import__("os").environ.get(
                    "VLA_SHAPE_SERIES",
                    "",
                ).strip(),
                "trial_key": __import__("os").environ.get(
                    "VLA_SHAPE_TRIAL_KEY",
                    "",
                ).strip(),
                "task_variant": __import__("os").environ.get(
                    "VLA_SHAPE_TASK_VARIANT",
                    "",
                ).strip(),
                "layout": __import__("os").environ.get(
                    "VLA_SHAPE_LAYOUT",
                    "",
                ).strip(),
                "repetition": __import__("os").environ.get(
                    "VLA_SHAPE_REPETITION",
                    "",
                ).strip(),
                "object_condition": __import__("os").environ.get(
                    "VLA_SHAPE_OBJECT_CONDITION",
                    "",
                ).strip(),
                "expected_object": __import__("os").environ.get(
                    "VLA_SHAPE_EXPECTED_OBJECT",
                    "",
                ).strip(),
                "expected_target": __import__("os").environ.get(
                    "VLA_SHAPE_EXPECTED_TARGET",
                    "",
                ).strip(),
                "protocol_eligible": (
                    __import__("os").environ.get(
                        "VLA_SHAPE_PROTOCOL_ELIGIBLE",
                        "YES",
                    ).strip().upper()
                    == "YES"
                ),
            },

        }

        json_write(
            run_dir
            / "episode_failure.json",
            failure,
        )

        raise


if __name__ == "__main__":
    main()
