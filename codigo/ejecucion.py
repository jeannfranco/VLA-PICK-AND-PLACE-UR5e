#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

import numpy as np
import mujoco
import rclpy
from std_msgs.msg import Float64MultiArray

import robot_base as base
import math
import os
import io
import copy

GRIPPER_TOPIC = "/onrobot/finger_width_controller/commands"


def get_orange_points(scene_data):
    orange = scene_data["orange_die"]

    above = np.array(orange["safe_above_base_xyz_m"], dtype=float)

    if "base_xyz_m" not in orange:
        raise RuntimeError("El JSON no tiene orange_die.base_xyz_m")

    obj = np.array(orange["base_xyz_m"], dtype=float)

    return obj, above


class PickLiftNode(base.RealNode):
    def __init__(self):
        super().__init__()
        self.grip_pub = self.create_publisher(Float64MultiArray, GRIPPER_TOPIC, 10)



    def set_gripper_width(self, width_m, wait_sec=2.5):
        """
        Envía la orden a la 3FG15 y valida el canal real de control.

                /onrobot/joint_states no se usa como condición de éxito porque
                el driver mantiene finger_width congelado aunque la garra
                se mueva físicamente.
        """
        import time

        target = float(width_m)

        if not (0.0 <= target <= 0.150):
            raise RuntimeError(
                f"Ancho de garra fuera de rango: {target}"
            )

        deadline = time.monotonic() + 6.0

        while (
            self.grip_pub.get_subscription_count() < 1
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.10)

        subscribers = self.grip_pub.get_subscription_count()

        if subscribers < 1:
            raise RuntimeError(
                "No hay suscriptores en "
                f"{GRIPPER_TOPIC}. El controlador 3FG15 "
                "no está activo."
            )

        msg = Float64MultiArray()
        msg.data = [target]

        print(
            "=== GRIPPER COMMAND === "
            f"target={target:.4f} m "
            f"subscribers={subscribers} "
            "verification=controller_subscription",
            flush=True,
        )

        # Publicación repetida para evitar perder una orden puntual.
        for _ in range(20):
            self.grip_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.10)

        print(
            "Orden enviada al controlador activo: "
            f"target={target:.4f} m",
            flush=True,
        )

        time.sleep(float(wait_sec))




def solve_point(model, data, tcp_id, qpos_ids, dof_ids, target_real, q_seed, offset):
    target_sim = np.array(target_real, dtype=float) - offset
    q0_scan = np.linspace(float(q_seed[0]) - 0.7, float(q_seed[0]) + 0.7, 9)

    sol = base.solve_topdown_ik(
        model, data, tcp_id, qpos_ids, dof_ids,
        target_sim, q_seed, q0_scan
    )

    return sol, target_sim




def _clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))



def _load_guidance_profile():
    """
    Carga un perfil de influencia VLA.

        La semántica de la tarea no se obtiene del perfil:
        objeto y destino proceden de la instrucción resuelta por flujo.py.
    """
    import json
    import os
    from pathlib import Path

    profile_path = os.environ.get(
        "VLA_GUIDANCE_PROFILE", ""
    ).strip()

    if not profile_path:
        return None

    path = Path(profile_path)

    if not path.exists():
        raise RuntimeError(
            f"VLA_GUIDANCE_PROFILE no existe: {path}"
        )

    profile = json.loads(
        path.read_text(encoding="utf-8")
    )
    profile["profile_path"] = str(path)

    guidance = profile.get("guidance", {})
    decision = profile.get("decision", {})

    if (
        isinstance(decision, dict)
        and "valid_for_execution" in decision
        and not decision["valid_for_execution"]
    ):
        raise RuntimeError(
            "Perfil marcado como no válido para ejecución."
        )

    if (
        isinstance(guidance, dict)
        and "valid_for_execution" in guidance
        and not guidance["valid_for_execution"]
    ):
        raise RuntimeError(
            "Guidance marcado como no válido."
        )

    experiment = profile.get("experiment_profile")
    critical_locked = bool(
        guidance.get(
            "critical_contact_locked",
            profile.get("critical_contact_locked", True),
        )
    )

    dry_run = os.environ.get("DRY_RUN", "1") != "0"

    if not critical_locked:
        if experiment != "with_grasp":
            raise RuntimeError(
                "Solo with_grasp puede desbloquear "
                "las fases de contacto."
            )

        if (
            not dry_run
            and os.environ.get(
                "VLA_ALLOW_GRASP_REAL", ""
            ) != "YES"
        ):
            raise RuntimeError(
                "with_grasp requiere "
                "VLA_ALLOW_GRASP_REAL=YES."
            )

    return profile





def _resolve_gripper_guidance(profile, open_width, close_width):
    report = {
        "enabled": False,
        "baseline_open_width_m": float(open_width),
        "baseline_close_width_m": float(close_width),
        "baseline_close_hold_s": 3.0,
        "applied_open_width_m": float(open_width),
        "applied_close_width_m": float(close_width),
        "applied_close_hold_s": 3.0,
        "reason": "profile_without_gripper_guidance",
    }

    if not isinstance(profile, dict):
        return report

    guidance = profile.get("gripper_guidance")
    if not isinstance(guidance, dict):
        return report

    report.update(guidance)

    enabled = bool(guidance.get("enabled", False))
    report["enabled"] = enabled

    if not enabled:
        report["applied_open_width_m"] = float(open_width)
        report["applied_close_width_m"] = float(close_width)
        report["applied_close_hold_s"] = 3.0
        report["reason"] = "gripper_fixed_baseline"
        return report

    close_bounds = guidance.get("close_width_bounds_m", [0.015, 0.024])
    hold_bounds = guidance.get("close_hold_bounds_s", [2.0, 4.0])

    close_lo, close_hi = map(float, close_bounds[:2])
    hold_lo, hold_hi = map(float, hold_bounds[:2])

    requested_close = float(
        guidance.get("requested_close_width_m", close_width)
    )
    requested_hold = float(
        guidance.get("requested_close_hold_s", 3.0)
    )

    applied_close = max(close_lo, min(close_hi, requested_close))
    applied_hold = max(hold_lo, min(hold_hi, requested_hold))

    applied_open = float(open_width)

    if not (0.0 <= applied_close < applied_open <= 0.150):
        raise RuntimeError(
            "Configuración VLA de garra inválida: "
            f"open={applied_open}, close={applied_close}"
        )

    report["applied_open_width_m"] = applied_open
    report["applied_close_width_m"] = applied_close
    report["applied_close_hold_s"] = applied_hold
    report["reason"] = "vla_gripper_guidance_applied"

    print(
        f"enabled={enabled} "
        f"raw={report.get('raw_gripper_signal')} "
        f"normalized={report.get('normalized_close_signal')} "
        f"close={applied_close:.4f}m "
        f"hold={applied_hold:.2f}s",
        flush=True,
    )

    return report


def _apply_phase_joint_guidance(
    profile,
    model,
    data,
    tcp_id,
    qpos_ids,
    q_pre,
    q_pick,
    q_lift,
    q_pre_place,
    q_place,
    q_retreat,
    q_lift_steps,
    target_sims,
    lift_step_sims,
):
    report = {
        "enabled": False,
        "reason": "profile_without_joint_guidance",
        "phase_scales": {},
        "delta_q_vla_rad": [0.0] * 6,
        "phase_metrics": {},
    }

    original = {
        "q_pre": np.asarray(q_pre, dtype=float).copy(),
        "q_pick": np.asarray(q_pick, dtype=float).copy(),
        "q_lift": np.asarray(q_lift, dtype=float).copy(),
        "q_pre_place": np.asarray(q_pre_place, dtype=float).copy(),
        "q_place": np.asarray(q_place, dtype=float).copy(),
        "q_retreat": np.asarray(q_retreat, dtype=float).copy(),
    }

    if not isinstance(profile, dict):
        return (
            q_pre, q_pick, q_lift, q_pre_place, q_place, q_retreat,
            q_lift_steps, [], report,
        )

    guidance = profile.get("joint_guidance")
    if not isinstance(guidance, dict) or not guidance.get("enabled", False):
        return (
            q_pre, q_pick, q_lift, q_pre_place, q_place, q_retreat,
            q_lift_steps, [], report,
        )

    delta = np.asarray(
        guidance.get("delta_q_vla_rad", []),
        dtype=float,
    ).reshape(-1)

    if delta.shape != (6,) or not np.all(np.isfinite(delta)):
        raise RuntimeError(
            f"delta_q_vla_rad inválido: shape={delta.shape}, values={delta}"
        )

    max_delta_limit = float(guidance.get("max_delta_rad", 0.035))
    if float(np.max(np.abs(delta))) > max_delta_limit + 1e-9:
        raise RuntimeError(
            "delta_q_vla supera su límite: "
            f"{np.max(np.abs(delta))} > {max_delta_limit}"
        )

    scales = guidance.get("phase_scales", {})
    required = (
        "q_pre", "q_pick", "q_lift",
        "q_pre_place", "q_place", "q_retreat",
    )

    clean_scales = {}
    for name in required:
        value = float(scales.get(name, 0.0))
        if not (0.0 <= value <= 1.0):
            raise RuntimeError(
                f"Escala de fase fuera de [0,1]: {name}={value}"
            )
        clean_scales[name] = value

    adapted = {
        name: original[name] + clean_scales[name] * delta
        for name in required
    }

    original_lift_steps = [
        np.asarray(q, dtype=float).copy()
        for q in q_lift_steps
    ]
    adapted_lift_steps = []

    n_steps = len(original_lift_steps)
    pick_scale = clean_scales["q_pick"]
    lift_scale = clean_scales["q_lift"]

    for index, q_step in enumerate(original_lift_steps):
        ratio = float(index + 1) / max(1, n_steps)
        step_scale = pick_scale + ratio * (lift_scale - pick_scale)
        adapted_lift_steps.append(q_step + step_scale * delta)

    def fk(q_value):
        base.set_robot_q(model, data, qpos_ids, q_value)
        pos, axis, _ = base.get_site_pos_axis(data, tcp_id)

        pos = np.asarray(pos, dtype=float).reshape(3)
        axis = np.asarray(axis, dtype=float).reshape(3)

        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-9:
            raise RuntimeError("Eje TCP inválido durante validación de perfil.")

        return pos, axis / axis_norm

    contact_limit = float(
        os.environ.get("VLA_PHASE_CONTACT_MAX_FK_ERROR_M", "0.035")
    )
    noncontact_limit = float(
        os.environ.get("VLA_PHASE_NONCONTACT_MAX_FK_ERROR_M", "0.050")
    )
    max_axis_delta_deg = float(
        os.environ.get("VLA_PHASE_MAX_AXIS_DELTA_DEG", "2.0")
    )

    metrics = {}

    for name in required:
        base_pos, base_axis = fk(original[name])
        new_pos, new_axis = fk(adapted[name])

        target = np.asarray(target_sims[name], dtype=float).reshape(3)
        target_error = float(np.linalg.norm(new_pos - target))

        dot = float(np.clip(np.dot(base_axis, new_axis), -1.0, 1.0))
        axis_delta_deg = float(np.degrees(np.arccos(dot)))

        applied_delta = adapted[name] - original[name]

        metrics[name] = {
            "phase_scale": clean_scales[name],
            "target_error_m": target_error,
            "axis_delta_from_baseline_deg": axis_delta_deg,
            "max_abs_applied_joint_delta_rad": float(
                np.max(np.abs(applied_delta))
            ),
            "applied_delta_q_rad": [
                float(x) for x in applied_delta
            ],
        }

        limit = (
            contact_limit
            if name in ("q_pick", "q_place")
            else noncontact_limit
        )

        if target_error > limit:
            raise RuntimeError(
                f"{name} pierde el target: "
                f"error={target_error:.4f} m > {limit:.4f} m"
            )

        if axis_delta_deg > max_axis_delta_deg:
            raise RuntimeError(
                f"{name} cambia demasiado el eje TCP: "
                f"{axis_delta_deg:.2f} deg > {max_axis_delta_deg:.2f} deg"
            )

    lift_step_metrics = []
    for index, (old_q, new_q, target) in enumerate(
        zip(original_lift_steps, adapted_lift_steps, lift_step_sims)
    ):
        old_pos, old_axis = fk(old_q)
        new_pos, new_axis = fk(new_q)

        target = np.asarray(target, dtype=float).reshape(3)
        error = float(np.linalg.norm(new_pos - target))
        dot = float(np.clip(np.dot(old_axis, new_axis), -1.0, 1.0))
        axis_delta = float(np.degrees(np.arccos(dot)))

        lift_step_metrics.append({
            "index": index,
            "target_error_m": error,
            "axis_delta_from_baseline_deg": axis_delta,
        })

        if error > noncontact_limit:
            raise RuntimeError(
                f"lift step {index} pierde el target: "
                f"{error:.4f} m > {noncontact_limit:.4f} m"
            )

        if axis_delta > max_axis_delta_deg:
            raise RuntimeError(
                f"lift step {index} cambia el eje TCP: "
                f"{axis_delta:.2f} deg > {max_axis_delta_deg:.2f} deg"
            )

    lift_step_max_dq = []
    previous = adapted["q_pick"]
    for q_step in adapted_lift_steps:
        lift_step_max_dq.append(
            float(np.max(np.abs(q_step - previous)))
        )
        previous = q_step

    report = {
        "enabled": True,
        "reason": "phase_joint_guidance_applied_and_fk_validated",
        "experiment_profile": profile.get("experiment_profile"),
        "historical_global_influence_percent": profile.get(
            "historical_global_influence_percent"
        ),
        "phase_scales": clean_scales,
        "delta_q_vla_rad": [float(x) for x in delta],
        "phase_metrics": metrics,
        "lift_step_metrics": lift_step_metrics,
        "contact_max_fk_error_m": contact_limit,
        "noncontact_max_fk_error_m": noncontact_limit,
        "max_axis_delta_deg": max_axis_delta_deg,
    }

    print(
        f"profile={report['experiment_profile']} "
        f"scales={clean_scales} "
        f"delta_q={report['delta_q_vla_rad']}",
        flush=True,
    )

    return (
        adapted["q_pre"],
        adapted["q_pick"],
        adapted["q_lift"],
        adapted["q_pre_place"],
        adapted["q_place"],
        adapted["q_retreat"],
        adapted_lift_steps,
        lift_step_max_dq,
        report,
    )

def _physical_execution_enabled():
    import os
    if os.environ.get("DRY_RUN", "0") == "1":
        return True
    return (
        os.environ.get("ALLOW_PHYSICAL_EXECUTION", "") == "YES"
        and os.environ.get("CONFIRM_PHYSICAL_EXECUTION", "") == "YES"
        and os.environ.get("CONFIRM_EXECUTION_RISK", "") == "YES"
    )

def _apply_guidance_to_waypoints(profile, pre_pick_xyz, pick_xyz, lift_xyz, pre_place_xyz, place_xyz, retreat_xyz):
    """
    Variante visual solo DRY_RUN:
        - permite aplicar deltas VLA también a pick/place
        - aborta si DRY_RUN != 1
        - carga el JSON completo del perfil para no perder campos filtrados por el loader
    """
    import os
    import json
    from pathlib import Path
    import numpy as np

    if not _physical_execution_enabled():
        raise RuntimeError("La ejecución física de pick-and-place requiere límites duros y confirmación explícita")

    # Recuperar perfil completo si el loader dejó solo campos básicos.
    ppath = profile.get("profile_path")
    if ppath and Path(ppath).exists():
        full = json.loads(Path(ppath).read_text(encoding="utf-8"))
        profile.update(full)
        profile["profile_path"] = ppath

    # Escala trazable de influencia VLA-guided sin tocar los JSON originales.
    import math as _math
    _gain_raw = os.environ.get("VLA_GUIDANCE_GAIN", "1.0").strip()
    try:
        _gain = float(_gain_raw)
    except Exception:
        raise RuntimeError(f"VLA_GUIDANCE_GAIN inválido: {_gain_raw!r}")

    if not _math.isfinite(_gain):
        raise RuntimeError(f"VLA_GUIDANCE_GAIN no finito: {_gain!r}")

    # Clamp duro: permite comparar 0.0, 1.0, 1.25, 1.50 sin romper seguridad.
    _gain = max(0.0, min(2.0, _gain))

    _gain_keys = (
        "pick_xy_delta_m",
        "place_xy_delta_m",
        "pre_place_xy_delta_m",
        "retreat_xy_delta_m",
        "pick_z_delta_m",
        "place_z_delta_m",
        "lift_z_delta_m",
        "retreat_z_delta_m",
        "source",
        "guidance_gain_preset",
        "phase_influence",
        "phase_influence_env",
        "phase_profile_tag",
        "phase_influence_note",
        "valid_for_execution",
        "do_not_execute",
    )

    if abs(_gain - 1.0) > 1e-12:
        profile = dict(profile or {})
        for _k in _gain_keys:
            if _k not in profile:
                continue
            _v = profile.get(_k)
            if isinstance(_v, (list, tuple)):
                profile[_k] = [float(x) * _gain for x in _v]
            elif isinstance(_v, (int, float)):
                profile[_k] = float(_v) * _gain
            else:
                raise RuntimeError(f"Campo guided no numérico para {_k}: {_v!r}")

        profile["guidance_gain"] = _gain
        profile["guidance_gain_env"] = _gain_raw
        print(f"gain={_gain:.3f} keys={list(_gain_keys)}", flush=True)
    def _xy(name, default=(0.0, 0.0), cap=0.04):
        v = profile.get(name, default)
        if v is None:
            v = default
        a = np.array(v, dtype=float).reshape(-1)
        if a.size < 2:
            a = np.array(default, dtype=float)
        a = a[:2]
        return np.clip(a, -cap, cap)

    def _scalar(name, default=0.0, lo=-0.004, hi=0.07):
        v = float(profile.get(name, default) or 0.0)
        return float(max(lo, min(hi, v)))

    allow_adjustment = bool(profile.get("allow_pick_place_adjustment", False))
    if not allow_adjustment:
        raise RuntimeError("Este ejecutor espera un perfil que permita el ajuste de pick-and-place")

    # Deltas no críticos.
    lift_z_delta_m = _scalar("lift_z_delta_m", 0.0, lo=0.0, hi=0.06)
    pre_place_xy_delta_m = _xy("pre_place_xy_delta_m", cap=0.04)
    retreat_xy_delta_m = _xy("retreat_xy_delta_m", cap=0.04)
    retreat_z_delta_m = _scalar("retreat_z_delta_m", 0.0, lo=0.0, hi=0.07)

    # Deltas críticos, acotados más fuerte.
    pick_xy_delta_m = _xy("pick_xy_delta_m", cap=0.008)
    place_xy_delta_m = _xy("place_xy_delta_m", cap=0.015)
    pick_z_delta_m = _scalar("pick_z_delta_m", 0.0, lo=-0.003, hi=0.004)
    place_z_delta_m = _scalar("place_z_delta_m", 0.0, lo=-0.003, hi=0.004)

    # Aplicar pick de forma coherente: pre_pick, pick y lift se desplazan juntos en XY.
    pre_pick_xyz[0] += pick_xy_delta_m[0]
    pre_pick_xyz[1] += pick_xy_delta_m[1]

    pick_xyz[0] += pick_xy_delta_m[0]
    pick_xyz[1] += pick_xy_delta_m[1]
    pick_xyz[2] += pick_z_delta_m

    lift_xyz[0] += pick_xy_delta_m[0]
    lift_xyz[1] += pick_xy_delta_m[1]
    lift_xyz[2] += lift_z_delta_m

    # Aplicar place de forma coherente: pre_place, place y retreat.
    pre_place_xyz[0] += pre_place_xy_delta_m[0] + place_xy_delta_m[0]
    pre_place_xyz[1] += pre_place_xy_delta_m[1] + place_xy_delta_m[1]

    place_xyz[0] += place_xy_delta_m[0]
    place_xyz[1] += place_xy_delta_m[1]
    place_xyz[2] += place_z_delta_m

    retreat_xyz[0] += pre_place_xy_delta_m[0] + retreat_xy_delta_m[0] + place_xy_delta_m[0]
    retreat_xyz[1] += pre_place_xy_delta_m[1] + retreat_xy_delta_m[1] + place_xy_delta_m[1]
    retreat_xyz[2] += retreat_z_delta_m

    return {
        "allow_pick_place_adjustment": True,
        "pick_adjustment_applied": True,
        "place_adjustment_applied": True,
        "pick_xy_delta_m": [float(x) for x in pick_xy_delta_m],
        "place_xy_delta_m": [float(x) for x in place_xy_delta_m],
        "pick_z_delta_m": float(pick_z_delta_m),
        "place_z_delta_m": float(place_z_delta_m),
        "lift_z_delta_m": float(lift_z_delta_m),
        "pre_place_xy_delta_m": [float(x) for x in pre_place_xy_delta_m],
        "retreat_xy_delta_m": [float(x) for x in retreat_xy_delta_m],
        "retreat_z_delta_m": float(retreat_z_delta_m),
    }



def _angle_wrap(a):
    import math
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def _angle_dist(a, b):
    return abs(_angle_wrap(float(a) - float(b)))


def _xyz_from_obj(obj):
    if not isinstance(obj, dict):
        return None
    for k in ("base_xyz_m", "center_base_xyz_m", "xyz_m", "safe_above_base_xyz_m"):
        v = obj.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            try:
                return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)
            except Exception:
                pass
    return None


def _radius_from_obj(obj, default=0.025):
    if not isinstance(obj, dict):
        return float(default)
    for k in ("radius_m", "radius_xy_m", "radius_p90_xy_m", "object_radius_m", "grasp_radius_m"):
        if k in obj:
            try:
                v = float(obj[k])
                if 0.005 <= v <= 0.080:
                    return v
            except Exception:
                pass
    return float(default)


def _collect_grasp_obstacles(scene_data, selected_xyz):
    obstacles = []
    selected_key = str(scene_data.get("selected_object", "") or "").lower()

    containers = []

    if isinstance(scene_data.get("objects"), dict):
        for k, v in scene_data["objects"].items():
            containers.append((f"objects.{k}", v))

    for k, v in scene_data.items():
        if isinstance(v, dict):
            containers.append((str(k), v))

    seen = set()

    for label, obj in containers:
        llabel = str(label).lower()

        # No usar zonas/targets como obstáculos de agarre.
        if any(x in llabel for x in ("target", "zone", "yellow", "blue", "source")):
            continue

        txt = json.dumps(obj, ensure_ascii=False).lower()
        if "target_color" in txt or "selected_as" in txt:
            continue

        # Evitar duplicados del propio objeto seleccionado.
        if selected_key and selected_key in llabel:
            continue
        if selected_key and selected_key in str(obj.get("semantic_override", "")).lower():
            continue

        xyz = _xyz_from_obj(obj)
        if xyz is None:
            continue

        dxy = float(np.linalg.norm(xyz[:2] - selected_xyz[:2]))

        # El mismo objeto suele aparecer duplicado como orange_die y white_cylinder.
        if dxy < 0.012:
            continue

        key = (round(float(xyz[0]), 4), round(float(xyz[1]), 4), round(float(xyz[2]), 4))
        if key in seen:
            continue
        seen.add(key)

        obstacles.append({
            "label": label,
            "xyz_m": [float(x) for x in xyz],
            "dist_xy_m": dxy,
            "radius_m": _radius_from_obj(obj),
            "object_color": obj.get("object_color"),
            "object_shape": obj.get("object_shape"),
            "semantic_override": obj.get("semantic_override"),
        })

    return sorted(obstacles, key=lambda o: o["dist_xy_m"])


def _dynamic_grasp_plan(scene_data, selected_xyz, q_reference):
    import math
    import os

    enabled_env = os.environ.get("VLA_DYNAMIC_GRASP", "1").strip() != "0"
    force_env = os.environ.get("VLA_FORCE_DYNAMIC_GRASP", "0").strip() == "1"

    threshold = float(os.environ.get("VLA_DYNAMIC_GRASP_NEAR_M", "0.090"))
    max_yaw_deg = float(os.environ.get("VLA_DYNAMIC_GRASP_MAX_YAW_DEG", "35"))
    selected_radius = float(os.environ.get("VLA_DYNAMIC_GRASP_OBJECT_RADIUS_M", "0.025"))

    q_reference = np.asarray(q_reference, dtype=float).reshape(-1)
    wrist3_ref = float(q_reference[-1]) if q_reference.size else 0.0

    obstacles = _collect_grasp_obstacles(scene_data, np.asarray(selected_xyz, dtype=float))

    plan = {
        "enabled": False,
        "reason": "disabled",
        "env_enabled": bool(enabled_env),
        "force": bool(force_env),
        "threshold_m": float(threshold),
        "max_yaw_deg": float(max_yaw_deg),
        "selected_xyz_m": [float(x) for x in np.asarray(selected_xyz, dtype=float)],
        "selected_radius_m": float(selected_radius),
        "obstacles": obstacles[:8],
        "nearest_obstacle": obstacles[0] if obstacles else None,
        "wrist3_reference_rad": float(wrist3_ref),
        "wrist3_delta_rad": 0.0,
        "wrist3_delta_deg": 0.0,
        "score": None,
    }

    if not enabled_env:
        plan["reason"] = "VLA_DYNAMIC_GRASP=0"
        return plan

    if not obstacles:
        plan["reason"] = "no_obstacles_detected"
        return plan

    nearest = obstacles[0]
    nearest_clearance = float(nearest["dist_xy_m"] - selected_radius - float(nearest.get("radius_m", 0.025)))
    plan["nearest_clearance_m"] = nearest_clearance

    if (nearest["dist_xy_m"] > threshold) and not force_env:
        plan["reason"] = "nearest_obstacle_far_enough"
        return plan

    selected_xy = np.asarray(selected_xyz, dtype=float)[:2]

    obs_angles = []
    for obs in obstacles:
        if obs["dist_xy_m"] > max(threshold * 1.6, threshold + 0.04) and not force_env:
            continue

        oxyz = np.asarray(obs["xyz_m"], dtype=float)
        v = oxyz[:2] - selected_xy
        ang = math.atan2(float(v[1]), float(v[0]))
        weight = 1.0 / max(float(obs["dist_xy_m"]), 0.015)
        obs_angles.append((ang, weight, obs["label"], obs["dist_xy_m"]))

    if not obs_angles:
        plan["reason"] = "no_relevant_close_obstacles"
        return plan

    # Candidatos conservadores: reorientación moderada de wrist_3.
    max_yaw = math.radians(max(5.0, min(90.0, max_yaw_deg)))
    cand_degs = list(range(-int(round(math.degrees(max_yaw))), int(round(math.degrees(max_yaw))) + 1, 5))
    if 0 not in cand_degs:
        cand_degs.append(0)

    best = None

    for deg in cand_degs:
        delta = math.radians(float(deg))
        yaw = wrist3_ref + delta

        # Modelo simple de 3 dedos separados 120 grados.
        finger_dirs = [yaw, yaw + 2.0 * math.pi / 3.0, yaw + 4.0 * math.pi / 3.0]

        # La orientación evita que un dedo apunte directamente al obstáculo cercano.
        min_margin = 999.0
        weighted_margin = 0.0

        for ang, weight, label, dist in obs_angles:
            margin = min(_angle_dist(ang, fd) for fd in finger_dirs)
            min_margin = min(min_margin, margin)
            weighted_margin += weight * margin

        # Penalizar cambios innecesarios.
        score = weighted_margin + 0.5 * min_margin - 0.25 * abs(delta)

        item = {
            "deg": float(deg),
            "delta": float(delta),
            "score": float(score),
            "min_margin_deg": float(math.degrees(min_margin)),
        }

        if best is None or item["score"] > best["score"]:
            best = item

    if best is None:
        plan["reason"] = "no_candidate"
        return plan

    plan["enabled"] = True
    plan["reason"] = "close_obstacle_reorientation"
    plan["wrist3_delta_rad"] = float(best["delta"])
    plan["wrist3_delta_deg"] = float(best["deg"])
    plan["score"] = float(best["score"])
    plan["min_finger_obstacle_margin_deg"] = float(best["min_margin_deg"])

    print(
        "enabled=True "
        f"nearest={nearest['label']} dist={nearest['dist_xy_m']:.3f}m "
        f"clearance={nearest_clearance:.3f}m "
        f"wrist3_delta={best['deg']:.1f}deg",
        flush=True,
    )

    return plan



def _unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def _get_away_vector_from_obstacle(selected_xyz, dynamic_grasp_plan):
    """
    Vector horizontal desde obstáculo hacia objeto.
        Sirve para inclinar la garra alejándose del obstáculo cercano.
    """
    selected = np.asarray(selected_xyz, dtype=float).reshape(3)

    nearest = None
    if isinstance(dynamic_grasp_plan, dict):
        nearest = dynamic_grasp_plan.get("nearest_obstacle")

    if isinstance(nearest, dict) and nearest.get("xyz_m") is not None:
        obs = np.asarray(nearest["xyz_m"], dtype=float).reshape(3)
        away = selected[:2] - obs[:2]
    else:
        away = np.array([1.0, 0.0], dtype=float)

    n = float(np.linalg.norm(away))
    if n < 1e-9:
        away = np.array([1.0, 0.0], dtype=float)
        n = 1.0

    away = away / n
    return np.array([float(away[0]), float(away[1]), 0.0], dtype=float)


def _build_grasp_orientation_candidates(selected_xyz, dynamic_grasp_plan):
    """
    Define las orientaciones candidatas para el agarre.

    - sin un obstáculo cercano y fiable se mantiene top-down;
    - con obstáculo se añaden candidatos inclinados o laterales;
    - la aproximación lateral sin obstáculo solo se habilita de forma explícita.

    La orientación lateral se reserva para escenas donde realmente aporta
    separación respecto a un obstáculo.
    """

    requested_mode = os.environ.get("VLA_GRASP_ORIENTATION_MODE", "auto").strip().lower()
    requested_force_non_topdown = (
        os.environ.get("VLA_GRASP_FORCE_NON_TOPDOWN", "0").strip() == "1"
    )
    allow_lateral_without_obstacle = (
        os.environ.get("VLA_GRASP_ALLOW_LATERAL_WITHOUT_OBSTACLE", "0").strip() == "1"
    )

    down = np.array([0.0, 0.0, -1.0], dtype=float)
    selected = np.asarray(selected_xyz, dtype=float).reshape(3)

    directions = []

    nearest = None
    nearest_dist = None

    if isinstance(dynamic_grasp_plan, dict):
        nearest = dynamic_grasp_plan.get("nearest_obstacle")

    if isinstance(nearest, dict) and nearest.get("xyz_m") is not None:
        try:
            nearest_dist = float(nearest.get("dist_xy_m", 999.0))
        except Exception:
            nearest_dist = 999.0

        threshold = float(os.environ.get("VLA_DYNAMIC_GRASP_NEAR_M", "0.090"))

        # Solo consideramos obstáculo fiable si realmente está dentro del umbral
        # y no es una región amplia ya descartada antes.
        if nearest_dist <= threshold:
            obs = np.asarray(nearest["xyz_m"], dtype=float).reshape(3)
            away2 = selected[:2] - obs[:2]
            n = float(np.linalg.norm(away2))

            if n > 1e-9:
                away2 = away2 / n
                perp_l = np.array([-away2[1], away2[0]], dtype=float)
                perp_r = np.array([away2[1], -away2[0]], dtype=float)

                directions.extend([
                    ("away_obstacle", np.array([away2[0], away2[1], 0.0], dtype=float)),
                    ("perp_left_obstacle", np.array([perp_l[0], perp_l[1], 0.0], dtype=float)),
                    ("perp_right_obstacle", np.array([perp_r[0], perp_r[1], 0.0], dtype=float)),
                ])

    reliable_obstacle = bool(directions)

    # Si no hay obstáculo fiable, anulamos lateral forzado salvo diag explícito.
    effective_mode = requested_mode
    force_non_topdown = bool(requested_force_non_topdown)

    if not reliable_obstacle and not allow_lateral_without_obstacle:
        effective_mode = "topdown"
        force_non_topdown = False

    # Solo en modo diag se prueban direcciones laterales sin obstáculo.
    if not directions and allow_lateral_without_obstacle:
        base_dirs = [
            ("px", [1.0, 0.0, 0.0]),
            ("nx", [-1.0, 0.0, 0.0]),
            ("py", [0.0, 1.0, 0.0]),
            ("ny", [0.0, -1.0, 0.0]),
            ("px_py", [1.0, 1.0, 0.0]),
            ("px_ny", [1.0, -1.0, 0.0]),
            ("nx_py", [-1.0, 1.0, 0.0]),
            ("nx_ny", [-1.0, -1.0, 0.0]),
        ]
        for name, v in base_dirs:
            directions.append((name, _unit(np.array(v, dtype=float))))

    candidates = []

    def add(name, tilt_deg, lateral_dir=None):
        if lateral_dir is None:
            axis = down.copy()
        else:
            th = math.radians(float(tilt_deg))
            axis = math.cos(th) * down + math.sin(th) * lateral_dir
            axis = _unit(axis)

        candidates.append({
            "name": str(name),
            "tilt_deg": float(tilt_deg),
            "target_axis": [float(x) for x in axis],
        })

    if not force_non_topdown:
        add("topdown", 0.0, None)

    if effective_mode in ("auto", "tilt", "side", "lateral", "any") and directions:
        raw = os.environ.get("VLA_GRASP_TILT_DEGS", "10,20,30,40,55")
        for dname, dvec in directions:
            dvec = _unit(dvec)
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                deg = float(part)
                if deg > 0:
                    add(f"tilt_{deg:g}_{dname}", deg, dvec)

    # Como medida de seguridad, la lista nunca se deja vacía.
    if not candidates:
        add("topdown", 0.0, None)

    return {
        "enabled": bool(effective_mode in ("auto", "tilt", "side", "lateral", "any")),
        "requested_mode": requested_mode,
        "mode": effective_mode,
        "requested_force_non_topdown": bool(requested_force_non_topdown),
        "force_non_topdown": bool(force_non_topdown),
        "reliable_obstacle_for_lateral": bool(reliable_obstacle),
        "allow_lateral_without_obstacle": bool(allow_lateral_without_obstacle),
        "nearest_obstacle_dist_xy_m": nearest_dist,
        "directions_tested": [
            {"name": name, "xy": [float(vec[0]), float(vec[1])]}
            for name, vec in directions
        ],
        "candidates": candidates,
        "tested": [],
        "selected": None,
        "selected_axis_name": None,
    }


def solve_point_axis(model, data, tcp_id, qpos_ids, dof_ids, target_real, q_seed, offset, target_axis):
    """
    Resuelve la IK con un eje de TCP elegido.

    - con VLA_STRICT_VERTICAL_TOPDOWN=1 y target_axis=[0,0,-1] se exige una pose
      vertical estricta;
    - en los demás casos se usa el solver de eje/top-down.

    Las soluciones que inclinan la garra fuera de la tolerancia se descartan.
    """

    target_real = np.asarray(target_real, dtype=float).reshape(3)
    target_sim = target_real - np.asarray(offset, dtype=float).reshape(3)
    q_seed = np.asarray(q_seed, dtype=float).reshape(-1)
    axis = np.asarray(target_axis, dtype=float).reshape(3)
    axis_norm = axis / max(1e-9, float(np.linalg.norm(axis)))

    strict_vertical = (
        os.environ.get("VLA_STRICT_VERTICAL_TOPDOWN", "1").strip() == "1"
        and float(np.dot(axis_norm, np.array([0.0, 0.0, -1.0]))) > 0.995
    )

    def _rot_delta_deg(Ra, Rb):
        R = np.asarray(Ra, dtype=float).reshape(3, 3).T @ np.asarray(Rb, dtype=float).reshape(3, 3)
        c = (float(np.trace(R)) - 1.0) / 2.0
        c = max(-1.0, min(1.0, c))
        return float(math.degrees(math.acos(c)))

    def _make_vertical_target_rot(q_ref):
        base.set_robot_q(model, data, qpos_ids, np.asarray(q_ref, dtype=float))
        mujoco.mj_forward(model, data)

        Rref = np.asarray(data.site_xmat[tcp_id], dtype=float).reshape(3, 3).copy()

        z_axis = np.array([0.0, 0.0, -1.0], dtype=float)

        # Se conserva el yaw aproximado del seed y se elimina la inclinación.
        x_axis = np.asarray(Rref[:, 0], dtype=float)
        x_axis = x_axis - float(np.dot(x_axis, z_axis)) * z_axis

        if float(np.linalg.norm(x_axis)) < 1e-8:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=float)

        x_axis = x_axis / float(np.linalg.norm(x_axis))
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / float(np.linalg.norm(y_axis))

        # Columnas = ejes del TCP. x × y = z.
        return np.column_stack([x_axis, y_axis, z_axis])

    def _strict_vertical_pose_ik():
        """
        Resuelve una IK top-down estricta.

        - fija la posición del TCP;
        - mantiene el eje Z del TCP vertical hacia abajo;
        - no fija el yaw de la garra;
        - no usa la orientación del objeto;
        - prueba varias semillas de muñeca para evitar mínimos locales.

        La posición y la verticalidad deben cumplirse al mismo tiempo.
        """
        max_iter = int(os.environ.get("VLA_VERTICAL_IK_MAX_ITER", "500"))
        damping = float(os.environ.get("VLA_VERTICAL_IK_DAMPING", "0.025"))
        max_step = float(os.environ.get("VLA_VERTICAL_IK_MAX_STEP_RAD", "0.055"))

        pos_w = float(os.environ.get("VLA_VERTICAL_IK_POS_WEIGHT", "1.0"))
        axis_w = float(os.environ.get("VLA_VERTICAL_IK_AXIS_WEIGHT", "1.20"))

        pos_tol = float(os.environ.get("VLA_VERTICAL_IK_POS_TOL_M", "0.030"))
        tilt_tol_deg = float(os.environ.get("VLA_MAX_TOPDOWN_TILT_DEG", "2.0"))

        target_z = np.array([0.0, 0.0, -1.0], dtype=float)
        dof_idx = np.asarray(dof_ids, dtype=int).reshape(-1)

        # El yaw es libre, pero la solución articular no debe saltar de rama.
        # Se penaliza la distancia respecto a q_seed y se añade una regularización
        # articular secundaria para favorecer la continuidad entre soluciones.
        cont_null_w = float(os.environ.get("VLA_VERTICAL_IK_CONTINUITY_NULL_WEIGHT", "0.08"))
        cont_score_w = float(os.environ.get("VLA_VERTICAL_IK_CONTINUITY_SCORE_WEIGHT", "0.08"))
        wrist_score_w = float(os.environ.get("VLA_VERTICAL_IK_WRIST_SCORE_WEIGHT", "0.10"))
        yaw_score_w = float(os.environ.get("VLA_VERTICAL_IK_YAW_SCORE_WEIGHT", "0.002"))

        def _angle_delta_to_seed(q_now):
            q_now = np.asarray(q_now, dtype=float).reshape(-1)
            return np.arctan2(np.sin(q_now - q_seed), np.cos(q_now - q_seed))

        def _wrap_near_seed(q_now):
            return q_seed + _angle_delta_to_seed(q_now)

        def _continuity_metrics(q_now, yaw_delta_rad):
            dq_seed = _angle_delta_to_seed(q_now)
            max_abs_dq = float(np.max(np.abs(dq_seed))) if dq_seed.size else 0.0
            if dq_seed.size >= 6:
                max_abs_wrist = float(np.max(np.abs(dq_seed[3:6])))
                wrist3_abs = float(abs(dq_seed[5]))
            else:
                max_abs_wrist = max_abs_dq
                wrist3_abs = max_abs_dq

            return {
                "max_abs_dq_from_seed": max_abs_dq,
                "max_abs_wrist_dq_from_seed": max_abs_wrist,
                "wrist3_abs_dq_from_seed": wrist3_abs,
                "yaw_abs_deg": float(abs(math.degrees(yaw_delta_rad))),
            }

        yaw_scan_deg = os.environ.get(
            "VLA_VERTICAL_YAW_SCAN_DEG",
            "0,30,-30,60,-60,90,-90,120,-120,180"
        )

        yaw_deltas = []
        for s in yaw_scan_deg.split(","):
            s = s.strip()
            if not s:
                continue
            try:
                yaw_deltas.append(math.radians(float(s)))
            except Exception:
                pass

        if not yaw_deltas:
            yaw_deltas = [0.0]

        def run(q_start, sign, yaw_delta_rad):
            q = np.asarray(q_start, dtype=float).copy()
            best = None

            for it in range(max_iter):
                base.set_robot_q(model, data, qpos_ids, q)
                mujoco.mj_forward(model, data)

                pos = np.asarray(data.site_xpos[tcp_id], dtype=float).copy()
                rot = np.asarray(data.site_xmat[tcp_id], dtype=float).reshape(3, 3).copy()

                pos_err = target_sim - pos

                tcp_z = np.asarray(rot[:, 2], dtype=float)
                tcp_z = tcp_z / max(1e-9, float(np.linalg.norm(tcp_z)))

                topdown_dot = float(np.dot(tcp_z, target_z))
                topdown_dot = max(-1.0, min(1.0, topdown_dot))
                tilt_deg = float(math.degrees(math.acos(topdown_dot)))

                # Error mínimo para alinear solo el eje Z del TCP.
                # El yaw alrededor del eje queda libre.
                axis_err = np.cross(tcp_z, target_z)
                axis_err = float(sign) * axis_err

                pos_norm = float(np.linalg.norm(pos_err))

                # Prioridad real:
                # 1) llegar al punto;
                # 2) mantener verticalidad;
                # 3) mantener continuidad articular respecto al seed anterior.
                cont = _continuity_metrics(q, yaw_delta_rad)
                score = (
                    pos_norm
                    + 0.010 * tilt_deg
                    + cont_score_w * cont["max_abs_dq_from_seed"]
                    + wrist_score_w * cont["max_abs_wrist_dq_from_seed"]
                    + yaw_score_w * abs(float(math.degrees(yaw_delta_rad)))
                )

                if best is None or score < best["score"]:
                    q_cmd = _wrap_near_seed(q)
                    best = {
                        "q": q_cmd.copy(),
                        "q_raw": q.copy(),
                        "pos": pos.copy(),
                        "pos_err_m": float(pos_norm),
                        "topdown_dot": float(topdown_dot),
                        "tilt_deg": float(tilt_deg),
                        "score": float(score),
                        "iters": int(it),
                        "sign": float(sign),
                        "yaw_delta_deg": float(math.degrees(yaw_delta_rad)),
                        "max_abs_dq_from_seed": float(cont["max_abs_dq_from_seed"]),
                        "max_abs_wrist_dq_from_seed": float(cont["max_abs_wrist_dq_from_seed"]),
                        "wrist3_abs_dq_from_seed": float(cont["wrist3_abs_dq_from_seed"]),
                    }

                if pos_norm <= pos_tol and tilt_deg <= tilt_tol_deg:
                    # La solución válida es la iteración que acaba de
                    # cumplir posición y verticalidad, no el registro
                    # histórico "best" elegido por un score que también
                    # penaliza continuidad articular.
                    q_cmd = _wrap_near_seed(q)

                    converged = {
                        "q": q_cmd.copy(),
                        "q_raw": q.copy(),
                        "pos": pos.copy(),
                        "pos_err_m": float(pos_norm),
                        "topdown_dot": float(topdown_dot),
                        "tilt_deg": float(tilt_deg),
                        "score": float(score),
                        "iters": int(it),
                        "sign": float(sign),
                        "yaw_delta_deg": float(
                            math.degrees(
                                yaw_delta_rad
                            )
                        ),
                        "max_abs_dq_from_seed": float(
                            cont[
                                "max_abs_dq_from_seed"
                            ]
                        ),
                        "max_abs_wrist_dq_from_seed": float(
                            cont[
                                "max_abs_wrist_dq_from_seed"
                            ]
                        ),
                        "wrist3_abs_dq_from_seed": float(
                            cont[
                                "wrist3_abs_dq_from_seed"
                            ]
                        ),
                        "ok": True,
                        "reason": (
                            "strict_vertical_axis_converged"
                        ),
                    }

                    return converged

                jacp = np.zeros((3, model.nv), dtype=float)
                jacr = np.zeros((3, model.nv), dtype=float)
                mujoco.mj_jacSite(model, data, jacp, jacr, tcp_id)

                J = np.vstack([
                    pos_w * jacp[:, dof_idx],
                    axis_w * jacr[:, dof_idx],
                ])

                e = np.concatenate([
                    pos_w * pos_err,
                    axis_w * axis_err,
                ])

                A = J @ J.T + (damping ** 2) * np.eye(6)

                try:
                    Ainv_e = np.linalg.solve(A, e)
                    Ainv_J = np.linalg.solve(A, J)
                    dq_task = J.T @ Ainv_e
                    J_pinv = J.T @ Ainv_J
                except np.linalg.LinAlgError:
                    A_pinv = np.linalg.pinv(A)
                    dq_task = J.T @ (A_pinv @ e)
                    J_pinv = J.T @ (A_pinv @ J)

                # Regularización articular secundaria: como el yaw es libre,
                # se favorecen soluciones cercanas a q_seed. La posición y la
                # verticalidad del TCP siguen siendo los criterios principales.
                q_err_seed = _angle_delta_to_seed(q)
                N = np.eye(J.shape[1], dtype=float) - J_pinv @ J
                dq_null = -cont_null_w * q_err_seed
                dq = dq_task + N @ dq_null

                dq = np.asarray(dq, dtype=float).reshape(-1)
                max_abs = float(np.max(np.abs(dq))) if dq.size else 0.0

                if max_abs > max_step:
                    dq *= max_step / max_abs

                q = q + dq

            if best is None:
                best = {
                    "q": q_start.copy(),
                    "pos": target_sim.copy(),
                    "pos_err_m": 999.0,
                    "topdown_dot": -1.0,
                    "tilt_deg": 999.0,
                    "score": 999.0,
                    "iters": 0,
                    "sign": float(sign),
                    "yaw_delta_deg": float(math.degrees(yaw_delta_rad)),
                }

            best["ok"] = bool(best["pos_err_m"] <= pos_tol and best["tilt_deg"] <= tilt_tol_deg)
            best["reason"] = "strict_vertical_axis_best_after_max_iter"
            return best

        all_results = []

        for yaw_delta in yaw_deltas:
            q0 = q_seed.copy()

            # Con yaw libre se prueban distintas semillas de wrist_3, sin imponer la orientación del objeto.
            if q0.size >= 6:
                q0[5] = q0[5] + yaw_delta

            for sign in (+1.0, -1.0):
                all_results.append(run(q0, sign, yaw_delta))

        ok_results = [r for r in all_results if bool(r.get("ok"))]
        best_pool = ok_results if ok_results else all_results
        best = min(best_pool, key=lambda x: float(x.get("score", 999.0)))

        q = np.asarray(best["q"], dtype=float).copy()
        base.set_robot_q(model, data, qpos_ids, q)
        mujoco.mj_forward(model, data)

        sim_tcp = np.asarray(data.site_xpos[tcp_id], dtype=float).copy()

        sol = {
            "q": q,
            "ok": bool(best.get("ok")),
            "reason": str(best.get("reason")),
            "pos_err_m": float(best.get("pos_err_m", 999.0)),
            "axis_dot": float(best.get("topdown_dot", -1.0)),
            "topdown_dot": float(best.get("topdown_dot", -1.0)),
            "tilt_deg": float(best.get("tilt_deg", 999.0)),
            "iters": int(best.get("iters", -1)),
            "sign": float(best.get("sign", 0.0)),
            "yaw_delta_deg": float(best.get("yaw_delta_deg", 0.0)),
            "strict_vertical_topdown": True,
            "strict_vertical_mode": "position_plus_vertical_axis_yaw_free_continuous",
            "max_abs_dq_from_seed": float(best.get("max_abs_dq_from_seed", 999.0)),
            "max_abs_wrist_dq_from_seed": float(best.get("max_abs_wrist_dq_from_seed", 999.0)),
            "wrist3_abs_dq_from_seed": float(best.get("wrist3_abs_dq_from_seed", 999.0)),
        }

        print(
            f"target={target_real.tolist()} "
            f"ok={sol['ok']} "
            f"pos_err={sol['pos_err_m']:.4f}m "
            f"tilt={sol['tilt_deg']:.2f}deg "
            f"dot={sol['topdown_dot']:.6f} "
            f"yaw_delta={sol['yaw_delta_deg']:.1f}deg "
            f"dq_seed={sol['max_abs_dq_from_seed']:.3f}rad "
            f"wrist_seed={sol['max_abs_wrist_dq_from_seed']:.3f}rad",
            flush=True,
        )

        return sol, sim_tcp

    if strict_vertical:
        return _strict_vertical_pose_ik()

    # Alternativa usada en modos no estrictos.
    q0_scan = np.linspace(float(q_seed[0]) - 0.7, float(q_seed[0]) + 0.7, 9)

    if hasattr(base, "solve_axis_ik"):
        sol = base.solve_axis_ik(
            model, data, tcp_id, qpos_ids, dof_ids,
            target_sim, q_seed, q0_scan, target_axis
        )
    else:
        sol = base.solve_topdown_ik(
            model, data, tcp_id, qpos_ids, dof_ids,
            target_sim, q_seed, q0_scan
        )

    q = np.asarray(sol["q"], dtype=float)
    base.set_robot_q(model, data, qpos_ids, q)
    mujoco.mj_forward(model, data)

    sim_tcp = np.asarray(data.site_xpos[tcp_id], dtype=float).copy()

    return sol, sim_tcp


def _solve_pick_lift_orientation_auto(
    model,
    data,
    tcp_id,
    qpos_ids,
    dof_ids,
    pre_pick_xyz,
    pick_xyz,
    lift_xyz,
    q_home,
    q_home_for_ik,
    offset,
    dynamic_grasp_plan=None,
):
    """
    Usa una ruta top-down vertical para pre_pick, pick y lift.

    - no se buscan orientaciones laterales o inclinadas;
    - no se usa el yaw del objeto;
    - el eje permitido es [0, 0, -1];
    - si la IK no cumple posición y verticalidad, el plan se rechaza.

    La orientación queda fijada de forma determinista antes de resolver la IK.
    """

    axis = np.array([0.0, 0.0, -1.0], dtype=float)

    lift_steps_n = int(os.environ.get("LIFT_STEPS", "4"))
    lift_steps_n = max(1, lift_steps_n)

    max_tilt_deg = float(os.environ.get("VLA_MAX_TOPDOWN_TILT_DEG", "2.0"))
    min_topdown_dot = math.cos(math.radians(max_tilt_deg))

    search = {
        "enabled": False,
        "requested_mode": "strict_topdown_only",
        "mode": "strict_topdown_only",
        "requested_force_non_topdown": False,
        "force_non_topdown": False,
        "reliable_obstacle_for_lateral": False,
        "allow_lateral_without_obstacle": False,
        "nearest_obstacle_dist_xy_m": (
            dynamic_grasp_plan.get("nearest_dist_xy_m")
            if isinstance(dynamic_grasp_plan, dict)
            else None
        ),
        "directions_tested": [],
        "candidates": [
            {
                "name": "topdown",
                "tilt_deg": 0.0,
                "target_axis": [0.0, 0.0, -1.0],
            }
        ],
        "tested": [],
        "selected_axis_name": "topdown",
        "reason": "clean_strict_topdown_only_no_orientation_search",
    }

    def _tilt_from_dot(dot):
        dot = max(-1.0, min(1.0, float(dot)))
        return float(math.degrees(math.acos(dot)))

    def _require_vertical(label, sol):
        dot = float(sol.get("topdown_dot", sol.get("axis_dot", -1.0)))
        tilt = _tilt_from_dot(dot)
        sol["topdown_dot"] = dot
        sol["tilt_deg"] = tilt

        if dot < min_topdown_dot:
            raise RuntimeError(
                f"{label} no es vertical: "
                f"topdown_dot={dot:.6f}, tilt={tilt:.2f} deg > {max_tilt_deg:.2f} deg"
            )

    print("ruta única: pre_pick/pick/lift con eje [0,0,-1]", flush=True)

    sol_pre, pre_sim = solve_point_axis(
        model, data, tcp_id, qpos_ids, dof_ids,
        pre_pick_xyz, q_home_for_ik, offset, axis
    )
    q_pre = np.asarray(sol_pre["q"], dtype=float)
    _require_vertical("pre_pick", sol_pre)

    sol_pick, pick_sim = solve_point_axis(
        model, data, tcp_id, qpos_ids, dof_ids,
        pick_xyz, q_pre, offset, axis
    )
    q_pick = np.asarray(sol_pick["q"], dtype=float)
    _require_vertical("pick", sol_pick)

    lift_step_xyzs = []
    lift_step_solutions = []
    lift_step_sims = []
    q_lift_steps = []
    lift_step_max_dq = []

    q_prev = q_pick.copy()

    for i in range(1, lift_steps_n + 1):
        alpha = float(i) / float(lift_steps_n)
        step_xyz = np.asarray(pick_xyz, dtype=float) + alpha * (
            np.asarray(lift_xyz, dtype=float) - np.asarray(pick_xyz, dtype=float)
        )

        sol_step, step_sim = solve_point_axis(
            model, data, tcp_id, qpos_ids, dof_ids,
            step_xyz, q_prev, offset, axis
        )
        q_step = np.asarray(sol_step["q"], dtype=float)
        _require_vertical(f"lift_step_{i}", sol_step)

        lift_step_xyzs.append(step_xyz.copy())
        lift_step_solutions.append(sol_step)
        lift_step_sims.append(step_sim)
        q_lift_steps.append(q_step.copy())
        lift_step_max_dq.append(float(np.max(np.abs(q_step - q_prev))))

        q_prev = q_step.copy()

    sol_lift = lift_step_solutions[-1]
    lift_sim = lift_step_sims[-1]
    q_lift = q_lift_steps[-1]

    max_home_pre = float(np.max(np.abs(q_pre - np.asarray(q_home, dtype=float))))
    max_pre_pick = float(np.max(np.abs(q_pick - q_pre)))
    max_pick_lift = float(max(lift_step_max_dq)) if lift_step_max_dq else 0.0

    ok_chain = bool(
        sol_pre.get("ok")
        and sol_pick.get("ok")
        and sol_lift.get("ok")
        and float(sol_pre.get("topdown_dot", -1.0)) >= min_topdown_dot
        and float(sol_pick.get("topdown_dot", -1.0)) >= min_topdown_dot
        and float(sol_lift.get("topdown_dot", -1.0)) >= min_topdown_dot
    )

    tested = {
        "name": "topdown",
        "tilt_deg": 0.0,
        "target_axis": [0.0, 0.0, -1.0],
        "ok_chain": bool(ok_chain),
        "score": float(
            float(sol_pre.get("pos_err_m", 9.0))
            + float(sol_pick.get("pos_err_m", 9.0))
            + float(sol_lift.get("pos_err_m", 9.0))
            + 0.010 * max_home_pre
            + 0.025 * max_pre_pick
            + 0.025 * max_pick_lift
        ),
        "pre_ok": bool(sol_pre.get("ok")),
        "pick_ok": bool(sol_pick.get("ok")),
        "lift_ok": bool(sol_lift.get("ok")),
        "pre_err_m": float(sol_pre.get("pos_err_m", 999.0)),
        "pick_err_m": float(sol_pick.get("pos_err_m", 999.0)),
        "lift_err_m": float(sol_lift.get("pos_err_m", 999.0)),
        "pre_axis_dot": float(sol_pre.get("axis_dot", sol_pre.get("topdown_dot", 0.0))),
        "pick_axis_dot": float(sol_pick.get("axis_dot", sol_pick.get("topdown_dot", 0.0))),
        "lift_axis_dot": float(sol_lift.get("axis_dot", sol_lift.get("topdown_dot", 0.0))),
        "pre_topdown_dot": float(sol_pre.get("topdown_dot", 0.0)),
        "pick_topdown_dot": float(sol_pick.get("topdown_dot", 0.0)),
        "lift_topdown_dot": float(sol_lift.get("topdown_dot", 0.0)),
        "pre_tilt_deg": float(sol_pre.get("tilt_deg", 999.0)),
        "pick_tilt_deg": float(sol_pick.get("tilt_deg", 999.0)),
        "lift_tilt_deg": float(sol_lift.get("tilt_deg", 999.0)),
        "max_abs_dq_home_to_pre": float(max_home_pre),
        "max_abs_dq_pre_to_pick": float(max_pre_pick),
        "max_abs_dq_pick_to_lift": float(max_pick_lift),
    }

    search["tested"].append(tested)
    search["selected"] = tested

    print(
        "selected=topdown "
        f"ok_chain={ok_chain} "
        f"pre_tilt={tested['pre_tilt_deg']:.2f}deg "
        f"pick_tilt={tested['pick_tilt_deg']:.2f}deg "
        f"lift_tilt={tested['lift_tilt_deg']:.2f}deg "
        f"pick_err={tested['pick_err_m']:.4f}m",
        flush=True,
    )

    if not ok_chain:
        raise RuntimeError(
            "La cadena IK top-down vertical no es válida: "
            f"pre_ok={tested['pre_ok']} pick_ok={tested['pick_ok']} lift_ok={tested['lift_ok']} "
            f"pre_err={tested['pre_err_m']:.4f} pick_err={tested['pick_err_m']:.4f} lift_err={tested['lift_err_m']:.4f} "
            f"pre_tilt={tested['pre_tilt_deg']:.2f} pick_tilt={tested['pick_tilt_deg']:.2f} lift_tilt={tested['lift_tilt_deg']:.2f}"
        )

    return (
        search,
        sol_pre, pre_sim, q_pre,
        sol_pick, pick_sim, q_pick,
        sol_lift, lift_sim, q_lift,
        lift_steps_n,
        lift_step_xyzs, lift_step_solutions, lift_step_sims,
        q_lift_steps, lift_step_max_dq,
    )




def _wrap_to_near(angle, reference):
    import math
    a = float(angle)
    r = float(reference)
    while a - r > math.pi:
        a -= 2.0 * math.pi
    while a - r < -math.pi:
        a += 2.0 * math.pi
    return a


def _read_photoneo_ply_xyz_camera_mm(ply_path, max_points=250000):
    """
    Lee PLY de Photoneo en formato ASCII o binary_little/big_endian.
        Devuelve puntos x,y,z en sistema cámara y unidades originales del PLY, normalmente mm.
    """

    ply_path = str(ply_path)

    type_map = {
        "char": "i1", "int8": "i1",
        "uchar": "u1", "uint8": "u1",
        "short": "i2", "int16": "i2",
        "ushort": "u2", "uint16": "u2",
        "int": "i4", "int32": "i4",
        "uint": "u4", "uint32": "u4",
        "float": "f4", "float32": "f4",
        "double": "f8", "float64": "f8",
    }

    try:
        with open(ply_path, "rb") as f:
            fmt = None
            vertex_count = None
            vertex_props = []
            in_vertex = False

            while True:
                raw = f.readline()
                if not raw:
                    return None

                line = raw.decode("ascii", errors="ignore").strip()

                if line.startswith("format "):
                    fmt = line.split()[1]

                elif line.startswith("element "):
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] == "vertex":
                        vertex_count = int(parts[2])
                        in_vertex = True
                    else:
                        in_vertex = False

                elif in_vertex and line.startswith("property "):
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] != "list":
                        typ = parts[1].lower()
                        name = parts[2]
                        vertex_props.append((name, typ))

                elif line == "end_header":
                    break

            if fmt is None or vertex_count is None or vertex_count <= 0:
                print("PLY sin formato/vertices válidos.", flush=True)
                return None

            names_lower = {name.lower(): i for i, (name, _) in enumerate(vertex_props)}
            if not all(k in names_lower for k in ("x", "y", "z")):
                print(f"PLY sin campos x/y/z. props={vertex_props[:10]}", flush=True)
                return None

            ix = names_lower["x"]
            iy = names_lower["y"]
            iz = names_lower["z"]

            if fmt == "ascii":
                txt = f.read().decode("ascii", errors="ignore")
                arr = np.loadtxt(
                    io.StringIO(txt),
                    dtype=np.float32,
                    max_rows=vertex_count,
                    usecols=(ix, iy, iz),
                )

            elif fmt in ("binary_little_endian", "binary_big_endian"):
                endian = "<" if fmt == "binary_little_endian" else ">"

                dtype_fields = []
                for name, typ in vertex_props:
                    if typ not in type_map:
                        print(f"Tipo PLY no soportado: {typ}", flush=True)
                        return None
                    dtype_fields.append((name, endian + type_map[typ]))

                dtype = np.dtype(dtype_fields)
                data = np.fromfile(f, dtype=dtype, count=vertex_count)

                if data.shape[0] <= 0:
                    return None

                xname = vertex_props[ix][0]
                yname = vertex_props[iy][0]
                zname = vertex_props[iz][0]

                arr = np.column_stack([
                    data[xname].astype(np.float32, copy=False),
                    data[yname].astype(np.float32, copy=False),
                    data[zname].astype(np.float32, copy=False),
                ])

            else:
                print(f"Formato PLY no soportado: {fmt}", flush=True)
                return None

        if arr.ndim != 2 or arr.shape[1] != 3:
            return None

        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.shape[0] <= 10:
            return None

        if arr.shape[0] > max_points:
            step = max(1, arr.shape[0] // max_points)
            arr = arr[::step].copy()

        print(
            f"PLY leído ok: fmt={fmt} points={arr.shape[0]} path={ply_path}",
            flush=True,
        )

        return arr

    except Exception as e:
        print(f"No pude leer PLY para yaw: {e}", flush=True)
        return None


def _camera_mm_to_base_m_points(xyz_camera_mm):

    T = np.array([
        [ 4.10000e-02, -9.98000e-01, -5.60000e-02, -2.19488e+02],
        [-9.99000e-01, -4.10000e-02,  0.00000e+00, -9.52756e+02],
        [-2.00000e-03,  5.60000e-02, -9.98000e-01,  8.66163e+02],
        [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00],
    ], dtype=float)

    pts = np.asarray(xyz_camera_mm, dtype=float)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    hom = np.hstack([pts, ones])
    base_mm = (T @ hom.T).T[:, :3]
    return base_mm / 1000.0


def _estimate_object_yaw_from_ply(scene_data, selected_xyz):
    """
    Estima yaw del objeto en XY usando PCA local sobre la nube Photoneo.

        No fuerza nada si la nube no es fiable.
        Esto sirve para rotar la garra alrededor de Z en agarres top-down.
    """

    report = {
        "enabled": False,
        "ok": False,
        "reason": "not_started",
        "ply": None,
        "yaw_rad": None,
        "yaw_deg": None,
        "quality": None,
        "num_points": 0,
    }

    if os.environ.get("VLA_OBJECT_YAW_GRASP", "1").strip() != "1":
        report["reason"] = "disabled_by_env"
        return report

    if not isinstance(scene_data, dict):
        report["reason"] = "scene_data_not_dict"
        return report

    ply = (
        scene_data.get("ply")
        or scene_data.get("ply_path")
        or os.environ.get("VLA_LIVE_SCENE_PLY")
        or "data/photoneo/live_scene.ply"
    )

    report["ply"] = str(ply)

    xyz_cam = _read_photoneo_ply_xyz_camera_mm(
        ply,
        max_points=int(os.environ.get("VLA_OBJECT_YAW_MAX_POINTS", "250000")),
    )

    if xyz_cam is None:
        report["reason"] = "ply_unreadable"
        return report

    pts = _camera_mm_to_base_m_points(xyz_cam)
    selected = np.asarray(selected_xyz, dtype=float).reshape(3)

    crop_r = float(os.environ.get("VLA_OBJECT_YAW_CROP_R_M", "0.075"))
    z_band = float(os.environ.get("VLA_OBJECT_YAW_Z_BAND_M", "0.055"))
    top_band = float(os.environ.get("VLA_OBJECT_YAW_TOP_BAND_M", "0.035"))
    min_pts = int(os.environ.get("VLA_OBJECT_YAW_MIN_POINTS", "80"))
    min_quality = float(os.environ.get("VLA_OBJECT_YAW_MIN_QUALITY", "0.10"))

    dxy = np.linalg.norm(pts[:, :2] - selected[:2], axis=1)
    dz = np.abs(pts[:, 2] - selected[2])

    crop = pts[(dxy <= crop_r) & (dz <= z_band)]

    if crop.shape[0] < min_pts:
        report["reason"] = "not_enough_crop_points"
        report["num_points"] = int(crop.shape[0])
        return report

    # Quedarse con la parte superior/lateral cercana del objeto, no con la mesa.
    zmax = float(np.nanmax(crop[:, 2]))
    top = crop[crop[:, 2] >= (zmax - top_band)]

    if top.shape[0] < min_pts:
        top = crop

    xy = top[:, :2]
    xy = xy - np.mean(xy, axis=0, keepdims=True)

    if xy.shape[0] < min_pts:
        report["reason"] = "not_enough_top_points"
        report["num_points"] = int(xy.shape[0])
        return report

    cov = np.cov(xy.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    denom = float(vals[0] + vals[1] + 1e-12)
    quality = float((vals[0] - vals[1]) / denom)

    # Si el objeto es casi cuadrado perfecto, la PCA puede ser ambigua.
    # Aun así puede usarse, pero con quality baja no se fuerza.
    if quality < min_quality:
        report.update({
            "reason": "low_pca_quality_square_or_noisy_object",
            "quality": quality,
            "num_points": int(top.shape[0]),
        })
        return report

    v = vecs[:, 0]
    yaw = float(math.atan2(v[1], v[0]))

    report.update({
        "enabled": True,
        "ok": True,
        "reason": "pca_object_yaw_ok",
        "yaw_rad": yaw,
        "yaw_deg": float(math.degrees(yaw)),
        "quality": quality,
        "num_points": int(top.shape[0]),
        "crop_r_m": float(crop_r),
        "z_band_m": float(z_band),
        "top_band_m": float(top_band),
    })

    print(
        f"ok=True yaw_deg={report['yaw_deg']:.1f} "
        f"quality={quality:.3f} points={top.shape[0]}",
        flush=True,
    )

    return report



def _build_topdown_yaw_seed_candidates(q_seed, object_yaw_report=None, dynamic_grasp_plan=None):
    """
    Busca varias orientaciones wrist_3 manteniendo agarre top-down.

        - Si Photoneo estima yaw del objeto, lo usa como referencia.
        - Si el objeto es ambiguo, prueba alternativa -90,-45,0,45,90.
        - No cambia X/Y/Z ni fuerza lateral.
    """
    q0 = np.asarray(q_seed, dtype=float).copy()
    current = float(q0[5])

    enabled = os.environ.get("VLA_TOPDOWN_YAW_SEARCH", "1").strip() == "1"
    max_delta_deg = float(os.environ.get("VLA_TOPDOWN_YAW_MAX_DELTA_DEG", "120"))
    max_delta = math.radians(max_delta_deg)

    report = {
        "enabled": bool(enabled),
        "current_wrist3_rad": float(current),
        "current_wrist3_deg": float(math.degrees(current)),
        "object_yaw_ok": bool(isinstance(object_yaw_report, dict) and object_yaw_report.get("ok")),
        "object_yaw_deg": object_yaw_report.get("yaw_deg") if isinstance(object_yaw_report, dict) else None,
        "max_delta_deg": float(max_delta_deg),
        "candidates": [],
        "trials": [],
        "selected_trial_idx": None,
        "selected_wrist3_rad": None,
        "selected_wrist3_deg": None,
        "selected_score": None,
    }

    if not enabled:
        report["reason"] = "disabled"
        return [q0], report

    angles = []

    # Prioridad: yaw del objeto si es fiable.
    if isinstance(object_yaw_report, dict) and object_yaw_report.get("ok") and object_yaw_report.get("yaw_rad") is not None:
        yaw = float(object_yaw_report["yaw_rad"])

        # En objetos tipo dado/cubo hay equivalencias cada 90 grados.
        for a in [
            yaw,
            yaw + math.pi / 2.0,
            yaw - math.pi / 2.0,
            yaw + math.pi,
            yaw - math.pi,
        ]:
            angles.append(a)

    # Como alternativa se prueban posturas alrededor de la actual.
    raw = os.environ.get("VLA_TOPDOWN_YAW_FALLBACK_DEGS", "-90,-45,0,45,90")
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        angles.append(current + math.radians(float(part)))

    # La postura actual también se incluye.
    angles.append(current)

    seeds = []
    seen = set()

    for a in angles:
        aa = _wrap_to_near(float(a), current)

        if abs(aa - current) > max_delta:
            continue

        key = round(aa, 3)
        if key in seen:
            continue
        seen.add(key)

        q = q0.copy()
        q[5] = aa
        seeds.append(q)

        report["candidates"].append({
            "wrist3_rad": float(aa),
            "wrist3_deg": float(math.degrees(aa)),
            "delta_deg": float(math.degrees(aa - current)),
        })

    if not seeds:
        seeds = [q0]
        report["reason"] = "current_only"
    else:
        report["reason"] = "ok"

    print(
        f"enabled={enabled} candidates={len(seeds)} current={math.degrees(current):.1f}deg "
        f"object_yaw_ok={report['object_yaw_ok']}",
        flush=True,
    )

    return seeds, report


def _apply_object_yaw_seed(q_seed, object_yaw_report):
    """
    Ajusta wrist_3 como semilla de IK para que el solver conserve una orientación
        de garra más coherente con la orientación geométrica del objeto.
    """
    q = np.asarray(q_seed, dtype=float).copy()

    if os.environ.get("VLA_OBJECT_YAW_GRASP", "1").strip() != "1":
        return q

    if not isinstance(object_yaw_report, dict) or not object_yaw_report.get("ok"):
        return q

    yaw = object_yaw_report.get("yaw_rad")
    if yaw is None:
        return q

    gain = float(os.environ.get("VLA_OBJECT_YAW_WRIST_GAIN", "1.0"))
    offset = float(os.environ.get("VLA_OBJECT_YAW_WRIST_OFFSET_RAD", "0.0"))

    current = float(q[5])

    # Para cubos/dados, 90 grados puede ser equivalente.
    # Se prueban variantes y se elige la que produce el menor salto desde la postura actual.
    variants = [
        float(yaw) + offset,
        float(yaw) + offset + math.pi / 2.0,
        float(yaw) + offset - math.pi / 2.0,
        float(yaw) + offset + math.pi,
        float(yaw) + offset - math.pi,
    ]

    variants = [_wrap_to_near(v, current) for v in variants]
    target = min(variants, key=lambda v: abs(v - current))

    q[5] = current + gain * (target - current)

    object_yaw_report["wrist3_before_rad"] = current
    object_yaw_report["wrist3_after_rad"] = float(q[5])
    object_yaw_report["wrist3_delta_rad"] = float(q[5] - current)
    object_yaw_report["wrist3_after_deg"] = float(math.degrees(q[5]))

    print(
        f"wrist3_before={math.degrees(current):.1f}deg "
        f"wrist3_after={math.degrees(float(q[5])):.1f}deg "
        f"delta={math.degrees(float(q[5]-current)):.1f}deg",
        flush=True,
    )

    return q



def _apply_object_yaw_after_ik(
    q_pre,
    q_pick,
    q_lift,
    q_lift_steps,
    object_orientation_grasp,
    topdown_yaw_search,
    grasp_orientation_search,
):
    """
    Aplica orientación de objeto al wrist_3 después de obtener una IK válida.

        - El IK top-down puede resolver bien la posición pero dejar una yaw de garra accidental.
        - Si Photoneo estima la orientación del objeto, wrist_3 se alinea con una orientación equivalente.
        - Para dados/cubos se aceptan equivalencias cada 90 grados.
        - No cambia X/Y/Z ni fuerza lateral.
    """


    q_pre = np.asarray(q_pre, dtype=float).copy()
    q_pick = np.asarray(q_pick, dtype=float).copy()
    q_lift = np.asarray(q_lift, dtype=float).copy()
    q_lift_steps = [np.asarray(q, dtype=float).copy() for q in (q_lift_steps or [])]

    report = dict(topdown_yaw_search or {})
    report["after_ik_apply_enabled"] = False
    report["after_ik_apply_reason"] = "not_started"

    if os.environ.get("VLA_APPLY_OBJECT_YAW_AFTER_IK", "1").strip() != "1":
        report["after_ik_apply_reason"] = "disabled_by_env"
        return q_pre, q_pick, q_lift, q_lift_steps, report

    axis_name = None
    if isinstance(grasp_orientation_search, dict):
        axis_name = grasp_orientation_search.get("selected_axis_name")

    # Esta corrección se aplica solo al top-down.
    # Si un obstáculo exige una aproximación lateral o inclinada, la búsqueda de eje tiene prioridad.
    if axis_name not in (None, "topdown"):
        report["after_ik_apply_reason"] = f"non_topdown_axis_{axis_name}"
        return q_pre, q_pick, q_lift, q_lift_steps, report

    if not isinstance(object_orientation_grasp, dict) or not object_orientation_grasp.get("ok"):
        report["after_ik_apply_reason"] = "object_yaw_not_reliable"
        return q_pre, q_pick, q_lift, q_lift_steps, report

    yaw = object_orientation_grasp.get("yaw_rad")
    if yaw is None:
        report["after_ik_apply_reason"] = "missing_yaw_rad"
        return q_pre, q_pick, q_lift, q_lift_steps, report

    def wrap_near(a, ref):
        a = float(a)
        ref = float(ref)
        while a - ref > math.pi:
            a -= 2.0 * math.pi
        while a - ref < -math.pi:
            a += 2.0 * math.pi
        return a

    current = float(q_pick[5])
    max_delta_deg = float(os.environ.get("VLA_OBJECT_YAW_AFTER_IK_MAX_DELTA_DEG", "120"))
    max_delta = math.radians(max_delta_deg)

    # Para dado/cubo: orientaciones equivalentes cada 90 grados.
    candidates = [
        float(yaw),
        float(yaw) + math.pi / 2.0,
        float(yaw) - math.pi / 2.0,
        float(yaw) + math.pi,
        float(yaw) - math.pi,
    ]

    # La postura actual también se mantiene como alternativa.
    candidates.append(current)

    trials = []
    best = None

    for c in candidates:
        cc = wrap_near(c, current)
        delta = cc - current

        if abs(delta) > max_delta:
            continue

        # Distancia a cualquier orientación equivalente del objeto.
        equivs = [
            wrap_near(float(yaw) + k * math.pi / 2.0, cc)
            for k in [-2, -1, 0, 1, 2]
        ]
        object_cost = min(abs(cc - e) for e in equivs)
        smooth_cost = abs(delta)

        score = object_cost + 0.20 * smooth_cost

        tr = {
            "candidate_wrist3_rad": float(cc),
            "candidate_wrist3_deg": float(math.degrees(cc)),
            "delta_deg": float(math.degrees(delta)),
            "object_cost_rad": float(object_cost),
            "smooth_cost_rad": float(smooth_cost),
            "score": float(score),
        }
        trials.append(tr)

        if best is None or score < best[0]:
            best = (score, cc, tr)

    if best is None:
        report["after_ik_apply_reason"] = "no_candidate_within_delta"
        report["after_ik_trials"] = trials
        return q_pre, q_pick, q_lift, q_lift_steps, report

    _, selected, selected_trial = best

    before = float(q_pick[5])

    q_pre[5] = selected
    q_pick[5] = selected
    q_lift[5] = selected

    for q in q_lift_steps:
        q[5] = selected

    report.update({
        "after_ik_apply_enabled": True,
        "after_ik_apply_reason": "object_yaw_applied_to_wrist3",
        "enabled": True,
        "selected_trial_idx": 0,
        "selected_wrist3_rad": float(selected),
        "selected_wrist3_deg": float(math.degrees(selected)),
        "selected_score": float(selected_trial["score"]),
        "wrist3_before_apply_rad": float(before),
        "wrist3_before_apply_deg": float(math.degrees(before)),
        "wrist3_delta_apply_deg": float(math.degrees(selected - before)),
        "after_ik_trials": trials,
        "trials": trials,
    })

    print(
        f"applied=True before={math.degrees(before):.1f}deg "
        f"after={math.degrees(selected):.1f}deg "
        f"delta={math.degrees(selected-before):.1f}deg",
        flush=True,
    )

    return q_pre, q_pick, q_lift, q_lift_steps, report



def _refine_prepick_orientation(
    model,
    data,
    site_id,
    qpos_ids,
    dof_ids,
    pre_pick_xyz,
    q_pre,
    pre_info,
    q_pick,
    q0_scan_values,
    grasp_orientation_search,
):
    """
    Recalcula el pre-pick usando q_pick como semilla para que la muñeca llegue
        orientada antes de bajar al objeto, no durante la bajada.
    """

    report = {
        "enabled": False,
        "accepted": False,
        "reason": "disabled",
    }

    if os.environ.get("VLA_EARLY_PREPICK_ORIENTATION", "1").strip() != "1":
        return q_pre, pre_info, report

    report["enabled"] = True

    if "solve_point_axis" not in globals():
        report["reason"] = "solve_point_axis_not_available"
        return q_pre, pre_info, report

    try:
        selected = None
        if isinstance(grasp_orientation_search, dict):
            selected = grasp_orientation_search.get("selected")

        target_axis = None
        if isinstance(selected, dict):
            target_axis = selected.get("target_axis")

        if target_axis is None:
            target_axis = [0.0, 0.0, -1.0]

        q_old = np.asarray(q_pre, dtype=float).copy()
        q_pick_np = np.asarray(q_pick, dtype=float).copy()

        q_new, info_new = solve_point_axis(
            model,
            data,
            site_id,
            qpos_ids,
            dof_ids,
            pre_pick_xyz,
            q_pick_np,
            q0_scan_values,
            target_axis,
        )

        old_wrist_dq = float(np.max(np.abs(q_old[3:6] - q_pick_np[3:6])))
        new_wrist_dq = float(np.max(np.abs(q_new[3:6] - q_pick_np[3:6])))

        old_total_dq = float(np.max(np.abs(q_old - q_pick_np)))
        new_total_dq = float(np.max(np.abs(q_new - q_pick_np)))

        report.update({
            "reason": "candidate_computed",
            "target_axis": [float(x) for x in target_axis],
            "old_wrist_dq_to_pick": old_wrist_dq,
            "new_wrist_dq_to_pick": new_wrist_dq,
            "old_total_dq_to_pick": old_total_dq,
            "new_total_dq_to_pick": new_total_dq,
            "old_pre_err_m": float((pre_info or {}).get("pos_err", 999.0)),
            "new_pre_err_m": float(info_new.get("pos_err", 999.0)),
            "new_pre_ok": bool(info_new.get("ok")),
        })

        # Aceptar si la IK es válida y la muñeca ya queda al menos igual o mejor preparada.
        if bool(info_new.get("ok")) and new_wrist_dq <= old_wrist_dq + 1e-6:
            report["accepted"] = True
            report["reason"] = "accepted_prepick_reoriented_before_descent"
            print(
                "accepted=True "
                f"wrist_dq_old={old_wrist_dq:.4f} "
                f"wrist_dq_new={new_wrist_dq:.4f} "
                f"pre_err={float(info_new.get('pos_err', 999.0)):.4f}",
                flush=True,
            )
            return q_new, info_new, report

        report["accepted"] = False
        report["reason"] = "rejected_no_improvement_or_invalid"
        print(
            "accepted=False "
            f"wrist_dq_old={old_wrist_dq:.4f} "
            f"wrist_dq_new={new_wrist_dq:.4f} "
            f"new_ok={bool(info_new.get('ok'))}",
            flush=True,
        )

        return q_pre, pre_info, report

    except Exception as e:
        report["accepted"] = False
        report["reason"] = f"exception: {e}"
        print(f"error={e}", flush=True)
        return q_pre, pre_info, report


def _apply_dynamic_waypoints_for_grasp(
    pre_pick_xyz,
    pick_xyz,
    lift_xyz,
    selected_xyz,
    dynamic_grasp_plan,
):
    """
    Waypoints generales:

        - Sin obstáculo fiable: no cambia la trayectoria. Top-down normal.
        - Con obstáculo fiable cerca: acerca desde el lado contrario al obstáculo
          y desplaza ligeramente el lift para salir de la zona de contacto.

        Esto evita hacer lateral si no hace falta, pero permite trayectoria dinámica
        cuando hay objetos muy juntos.
    """

    pre = np.asarray(pre_pick_xyz, dtype=float).copy()
    pick = np.asarray(pick_xyz, dtype=float).copy()
    lift = np.asarray(lift_xyz, dtype=float).copy()
    selected = np.asarray(selected_xyz, dtype=float).copy()

    report = {
        "enabled": False,
        "reason": "no_close_reliable_obstacle",
        "mode": "topdown_default",
        "pre_pick_before": [float(x) for x in pre],
        "pick_before": [float(x) for x in pick],
        "lift_before": [float(x) for x in lift],
        "pre_pick_after": None,
        "pick_after": None,
        "lift_after": None,
        "away_vector_xy": None,
    }

    if not isinstance(dynamic_grasp_plan, dict):
        report["reason"] = "no_dynamic_grasp_plan"
        report["pre_pick_after"] = [float(x) for x in pre]
        report["pick_after"] = [float(x) for x in pick]
        report["lift_after"] = [float(x) for x in lift]
        return pre, pick, lift, report

    nearest = dynamic_grasp_plan.get("nearest_obstacle")
    if not isinstance(nearest, dict) or nearest.get("xyz_m") is None:
        report["reason"] = "no_nearest_obstacle"
        report["pre_pick_after"] = [float(x) for x in pre]
        report["pick_after"] = [float(x) for x in pick]
        report["lift_after"] = [float(x) for x in lift]
        return pre, pick, lift, report

    try:
        dist = float(nearest.get("dist_xy_m", 999.0))
    except Exception:
        dist = 999.0

    threshold = float(os.environ.get("VLA_DYNAMIC_GRASP_NEAR_M", "0.090"))

    if dist > threshold:
        report["reason"] = "nearest_obstacle_far_enough"
        report["nearest_dist_xy_m"] = float(dist)
        report["threshold_m"] = float(threshold)
        report["pre_pick_after"] = [float(x) for x in pre]
        report["pick_after"] = [float(x) for x in pick]
        report["lift_after"] = [float(x) for x in lift]
        return pre, pick, lift, report

    obs = np.asarray(nearest["xyz_m"], dtype=float).reshape(3)
    away2 = selected[:2] - obs[:2]
    n = float(np.linalg.norm(away2))

    if n < 1e-9:
        report["reason"] = "invalid_away_vector"
        report["pre_pick_after"] = [float(x) for x in pre]
        report["pick_after"] = [float(x) for x in pick]
        report["lift_after"] = [float(x) for x in lift]
        return pre, pick, lift, report

    away2 = away2 / n

    approach_offset = float(os.environ.get("VLA_DYNAMIC_APPROACH_OFFSET_M", "0.055"))
    lift_away = float(os.environ.get("VLA_DYNAMIC_LIFT_AWAY_M", "0.020"))
    pre_pick_max_z = float(os.environ.get("VLA_PRE_PICK_MAX_Z_M", "0.200"))

    # Pre-pick desde el lado seguro, no encima del obstáculo.
    pre[:2] = pick[:2] + away2 * approach_offset
    pre[2] = min(float(pre[2]), pre_pick_max_z)
    pre[2] = max(float(pre[2]), float(pick[2]) + 0.100)

    # Pick se mantiene centrado sobre el objeto, porque el TCP representa el centro de agarre.
    # Lift sale un poco alejándose del obstáculo.
    lift[:2] = pick[:2] + away2 * lift_away

    report.update({
        "enabled": True,
        "reason": "close_obstacle_dynamic_approach",
        "mode": "dynamic_approach_away_from_obstacle",
        "nearest_label": nearest.get("label"),
        "nearest_dist_xy_m": float(dist),
        "threshold_m": float(threshold),
        "approach_offset_m": float(approach_offset),
        "lift_away_m": float(lift_away),
        "away_vector_xy": [float(away2[0]), float(away2[1])],
        "pre_pick_after": [float(x) for x in pre],
        "pick_after": [float(x) for x in pick],
        "lift_after": [float(x) for x in lift],
    })

    print(
        "enabled=True "
        f"reason={report['reason']} "
        f"nearest={report.get('nearest_label')} "
        f"dist={dist:.3f} "
        f"away={report['away_vector_xy']}",
        flush=True,
    )

    return pre, pick, lift, report


def _apply_dynamic_grasp_seed(q_seed, dynamic_grasp_plan):
    q = np.asarray(q_seed, dtype=float).copy()
    if dynamic_grasp_plan and dynamic_grasp_plan.get("enabled"):
        q[-1] += float(dynamic_grasp_plan.get("wrist3_delta_rad", 0.0))
    return q



def _merge_live_objects_into_scene_for_grasp(scene_data, scene_json_path):
    """
    Fusiona en la escena de planificación los objetos detectados en la captura live.

    flujo.py puede escribir un *_flujo/resultado_zonas.json y los
    obstáculos locales del PLY pueden quedar en *_cilindro_blanco/*. La fusión
    mantiene local_ply_obstacle_0 disponible para dynamic_grasp.
    """



    scene_json_path = Path(scene_json_path)

    live_path = (
        os.environ.get("VLA_LIVE_OBJECTS_SCENE_JSON")
        or os.environ.get("VLA_FORCE_LIVE_SCENE_JSON")
        or os.environ.get("DETECTED_SCENE_JSON")
    )

    report = {
        "enabled": True,
        "live_path": live_path,
        "merged_objects": [],
        "reason": None,
    }

    if not live_path:
        report["reason"] = "no_live_scene_env"
        scene_data["grasp_live_objects_merge"] = report
        return scene_data

    live_path = Path(live_path)

    if not live_path.exists():
        report["reason"] = "live_scene_path_not_found"
        scene_data["grasp_live_objects_merge"] = report
        return scene_data

    try:
        if live_path.resolve() == scene_json_path.resolve():
            report["reason"] = "same_scene_json"
            scene_data["grasp_live_objects_merge"] = report
            return scene_data
    except Exception:
        pass

    try:
        live_data = json.loads(live_path.read_text(encoding="utf-8"))
    except Exception as e:
        report["reason"] = f"live_scene_read_failed: {e}"
        scene_data["grasp_live_objects_merge"] = report
        return scene_data

    scene_objects = scene_data.setdefault("objects", {})
    live_objects = live_data.get("objects") or {}

    selected_key = (
        scene_data.get("selected_object")
        or live_data.get("selected_object")
        or scene_data.get("object_key")
        or scene_data.get("selected_object_key")
    )

    for key, obj in live_objects.items():
        if not isinstance(obj, dict):
            continue

        # No pisar el objeto seleccionado si ya existe en el JSON de salida.
        if key == selected_key and key in scene_objects:
            continue

        # Sí fusionar obstáculos locales y otros objetos candidatos.
        scene_objects[key] = obj
        report["merged_objects"].append(key)

    for meta_key in (
        "local_ply_grasp_obstacles",
        "live_object_detection",
        "live_obstacle_detection",
    ):
        if meta_key in live_data:
            scene_data[meta_key] = live_data[meta_key]

    report["reason"] = "ok"
    report["objects_after_merge"] = list(scene_objects.keys())
    scene_data["grasp_live_objects_merge"] = report

    try:
        scene_json_path.write_text(json.dumps(scene_data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        report["write_back_error"] = repr(e)

    print(
        "live objects merge: "
        f"live={live_path} merged={report['merged_objects']} objects={list(scene_objects.keys())}",
        flush=True,
    )

    return scene_data



def _obj_xyz(obj):
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


def _dist_xy_list(a, b):
    import math
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _enforce_live_object_consistency_for_grasp(scene_data):
    """
    Para agarre, la geometría local PLY manda sobre una detección semántica incoherente.

        Si un objeto semántico aparece lejos, pero existe local_ply_obstacle_0 cerca del objeto
        seleccionado, se corrige su posición para la planificación de agarre y se conserva la
        detección original en raw_semantic_detection.
    """

    objects = scene_data.setdefault("objects", {})
    selected_key = scene_data.get("selected_object") or scene_data.get("object_key") or scene_data.get("selected_object_key")

    report = {
        "enabled": True,
        "selected_object": selected_key,
        "corrections": [],
        "warnings": [],
    }

    if not selected_key or selected_key not in objects:
        report["warnings"].append("selected_object_missing_in_objects")
        scene_data["live_object_consistency"] = report
        return scene_data

    selected_xyz = _obj_xyz(objects.get(selected_key))
    if selected_xyz is None:
        report["warnings"].append("selected_xyz_missing")
        scene_data["live_object_consistency"] = report
        return scene_data

    local_key = None
    local_obj = None
    local_xyz = None
    best_d = None

    for k, obj in objects.items():
        if k == selected_key or not isinstance(obj, dict):
            continue

        if obj.get("semantic_override") != "local_ply_obstacle" and obj.get("source") != "local_ply_geometric_annulus":
            continue

        xyz = _obj_xyz(obj)
        if xyz is None:
            continue

        d = _dist_xy_list(selected_xyz, xyz)
        if best_d is None or d < best_d:
            best_d = d
            local_key = k
            local_obj = obj
            local_xyz = xyz

    if local_obj is None:
        report["warnings"].append("no_local_ply_obstacle")
        scene_data["live_object_consistency"] = report
        return scene_data

    near_m = float(os.environ.get("VLA_DYNAMIC_GRASP_NEAR_M", "0.090"))
    semantic_bad_m = float(os.environ.get("VLA_SEMANTIC_OBJECT_BAD_IF_FAR_M", "0.180"))

    for k, obj in list(objects.items()):
        if k == selected_key or k == local_key or not isinstance(obj, dict):
            continue

        xyz = _obj_xyz(obj)
        if xyz is None:
            continue

        d_sem = _dist_xy_list(selected_xyz, xyz)

        is_semantic_object = (
            k in ("orange_cube", "orange_die", "white_cylinder")
            or obj.get("source") == "optional_live_orange_obstacle_detector"
            or obj.get("object_color") in ("orange", "white")
        )

        if not is_semantic_object:
            continue

        if best_d <= near_m and d_sem >= semantic_bad_m:
            corrected = dict(local_obj)
            corrected["raw_semantic_detection"] = obj
            corrected["base_xyz_m"] = list(local_xyz)
            corrected["center_base_xyz_m"] = list(local_xyz)
            corrected["object_color"] = obj.get("object_color") or corrected.get("object_color") or "unknown"
            corrected["object_shape"] = obj.get("object_shape") or corrected.get("object_shape") or "local_ply_obstacle"
            corrected["semantic_override"] = obj.get("semantic_override") or k
            corrected["source"] = "live_ply_geometry_corrected_semantic_position"
            corrected["corrected_from_key"] = local_key
            corrected["semantic_raw_dist_xy_to_selected_m"] = float(d_sem)
            corrected["corrected_dist_xy_to_selected_m"] = float(best_d)

            objects[k] = corrected

            report["corrections"].append({
                "object_key": k,
                "raw_semantic_xyz_m": xyz,
                "corrected_xyz_m": list(local_xyz),
                "raw_dist_xy_to_selected_m": float(d_sem),
                "corrected_dist_xy_to_selected_m": float(best_d),
                "local_source_key": local_key,
            })

    report["objects_after_consistency"] = list(objects.keys())
    scene_data["objects"] = objects
    scene_data["live_object_consistency"] = report

    print(
        "live consistency corrections: "
        + json.dumps(report["corrections"], ensure_ascii=False),
        flush=True,
    )

    return scene_data


def _mark_dynamic_grasp_safety(dynamic_grasp_plan):


    min_clearance = float(os.environ.get("VLA_DYNAMIC_GRASP_MIN_CLEARANCE_M", "0.015"))
    dynamic_grasp_plan["min_required_clearance_m"] = float(min_clearance)

    clearance = dynamic_grasp_plan.get("nearest_clearance_m")

    if dynamic_grasp_plan.get("enabled") and clearance is not None and float(clearance) < min_clearance:
        dynamic_grasp_plan["safe_to_pick"] = False
        dynamic_grasp_plan["safety_reason"] = "nearest_clearance_too_small"
    else:
        dynamic_grasp_plan["safe_to_pick"] = True
        dynamic_grasp_plan["safety_reason"] = "ok"

    return dynamic_grasp_plan



def _align_prepick_wrist_to_pick(q_pre, q_pick, pre_info):
    """
    Evita que la muñeca termine de girar durante la bajada.

        En vez de recalcular IK, que depende de nombres internos como site_id,
        deja el pre-pick con la misma orientación de muñeca que el pick.
        Así el robot llega arriba del objeto ya orientado.

        No cambia hombro/codo de q_pre, solo wrist_1/wrist_2/wrist_3.
    """


    q_pre_new = np.asarray(q_pre, dtype=float).copy()
    q_pick_np = np.asarray(q_pick, dtype=float).copy()

    report = {
        "enabled": False,
        "accepted": False,
        "reason": "disabled",
    }

    if os.environ.get("VLA_EARLY_PREPICK_ORIENTATION", "1").strip() != "1":
        return q_pre, pre_info, report

    report["enabled"] = True

    old_wrist_dq = float(np.max(np.abs(q_pre_new[3:6] - q_pick_np[3:6])))
    old_total_dq = float(np.max(np.abs(q_pre_new - q_pick_np)))

    # La muñeca llega al pre-pick con la orientación que tendrá durante el pick.
    q_pre_new[3:6] = q_pick_np[3:6]

    new_wrist_dq = float(np.max(np.abs(q_pre_new[3:6] - q_pick_np[3:6])))
    new_total_dq = float(np.max(np.abs(q_pre_new - q_pick_np)))

    report.update({
        "accepted": True,
        "reason": "accepted_wrist_aligned_to_pick_before_descent",
        "old_wrist_dq_to_pick": old_wrist_dq,
        "new_wrist_dq_to_pick": new_wrist_dq,
        "old_total_dq_to_pick": old_total_dq,
        "new_total_dq_to_pick": new_total_dq,
        "q_pre_wrist_before": [float(x) for x in np.asarray(q_pre, dtype=float)[3:6]],
        "q_pre_wrist_after": [float(x) for x in q_pre_new[3:6]],
        "q_pick_wrist": [float(x) for x in q_pick_np[3:6]],
        "note": "pre-pick wrist copied from pick so orientation is ready before descent",
    })

    if isinstance(pre_info, dict):
        pre_info = dict(pre_info)
        pre_info["early_wrist_aligned_to_pick"] = True

    print(
        "accepted=True "
        f"old_wrist_dq={old_wrist_dq:.4f} "
        f"new_wrist_dq={new_wrist_dq:.4f}",
        flush=True,
    )

    return q_pre_new, pre_info, report



def _env_float_any(names, default):

    for name in names:
        v = os.environ.get(name)
        if v is None:
            continue
        v = str(v).strip()
        if not v:
            continue
        try:
            return float(v)
        except Exception:
            print(f"{name}={v!r} no es float; ignoro.", flush=True)
    return float(default)


def _apply_runtime_pick_gripper_config():
    """
    Unifica la configuración de pick Z y garra.

    VLA_PICK_Z_RELATIVE_M, min_pick_z, open_width y pick_z_override se normalizan
    en un único punto antes de planificar.
    """

    open_w = _env_float_any(
        ["VLA_OPEN_WIDTH_M", "OPEN_WIDTH_M", "GRIPPER_OPEN_WIDTH_M"],
        0.090,
    )
    close_w = _env_float_any(
        ["VLA_CLOSE_WIDTH_M", "CLOSE_WIDTH_M", "GRIPPER_CLOSE_WIDTH_M"],
        0.018,
    )

    pick_z_min = _env_float_any(
        ["VLA_PICK_Z_MIN_M", "PICK_Z_MIN_M", "MIN_PICK_Z_M"],
        0.032,
    )
    pick_z_guard_min = _env_float_any(
        ["VLA_PICK_Z_GUARD_MIN_M", "PICK_Z_GUARD_MIN_M"],
        max(0.0, pick_z_min - 0.002),
    )
    pick_z_relative = _env_float_any(
        ["VLA_PICK_Z_RELATIVE_M", "PICK_Z_RELATIVE_M"],
        -0.060,
    )

    # Si no hay un override explícito, se usa el cálculo adaptativo en lugar de fijar 0.06.
    # El modo adaptativo usa un mínimo bajo.
    os.environ["OPEN_WIDTH_M"] = f"{open_w:.6f}"
    os.environ["VLA_OPEN_WIDTH_M"] = f"{open_w:.6f}"
    os.environ["GRIPPER_OPEN_WIDTH_M"] = f"{open_w:.6f}"

    os.environ["CLOSE_WIDTH_M"] = f"{close_w:.6f}"
    os.environ["VLA_CLOSE_WIDTH_M"] = f"{close_w:.6f}"
    os.environ["GRIPPER_CLOSE_WIDTH_M"] = f"{close_w:.6f}"

    os.environ["VLA_PICK_Z_MODE"] = os.environ.get("VLA_PICK_Z_MODE", "adaptive")

    os.environ["VLA_PICK_Z_MIN_M"] = f"{pick_z_min:.6f}"
    os.environ["PICK_Z_MIN_M"] = f"{pick_z_min:.6f}"
    os.environ["MIN_PICK_Z_M"] = f"{pick_z_min:.6f}"

    os.environ["VLA_PICK_Z_GUARD_MIN_M"] = f"{pick_z_guard_min:.6f}"
    os.environ["PICK_Z_GUARD_MIN_M"] = f"{pick_z_guard_min:.6f}"

    os.environ["VLA_PICK_Z_RELATIVE_M"] = f"{pick_z_relative:.6f}"
    os.environ["PICK_Z_RELATIVE_M"] = f"{pick_z_relative:.6f}"

    # Contrato único para la altura de pick:
    # - adaptive: se calcula desde Photoneo y se eliminan valores residuales.
    # - override: el valor solicitado se conserva, valida y usa realmente.
    pick_z_mode = os.environ.get(
        "VLA_PICK_Z_MODE",
        "adaptive",
    ).strip().lower()

    override_text = (
        os.environ.get(
            "PICK_Z_OVERRIDE_M",
            "",
        ).strip()
        or os.environ.get(
            "VLA_PICK_Z_OVERRIDE_M",
            "",
        ).strip()
    )

    if pick_z_mode == "override":
        if not override_text:
            raise RuntimeError(
                "VLA_PICK_Z_MODE=override requiere "
                "PICK_Z_OVERRIDE_M."
            )

        pick_z_override = float(override_text)

        if not (0.020 <= pick_z_override <= 0.200):
            raise RuntimeError(
                "PICK_Z_OVERRIDE_M fuera del rango "
                f"validado: {pick_z_override:.4f} m"
            )

        os.environ["VLA_PICK_Z_MODE"] = "override"
        os.environ["PICK_Z_OVERRIDE_M"] = (
            f"{pick_z_override:.6f}"
        )
        os.environ["VLA_PICK_Z_OVERRIDE_M"] = (
            f"{pick_z_override:.6f}"
        )

        override_report = (
            f"{pick_z_override:.3f}"
        )

    else:
        os.environ.pop(
            "PICK_Z_OVERRIDE_M",
            None,
        )
        os.environ.pop(
            "VLA_PICK_Z_OVERRIDE_M",
            None,
        )

        override_report = "disabled"

    print(
        f"open_width={open_w:.3f} "
        f"close_width={close_w:.3f} "
        f"pick_z_min={pick_z_min:.3f} "
        f"guard_min={pick_z_guard_min:.3f} "
        f"pick_z_relative={pick_z_relative:.3f} "
        f"pick_z_override={override_report}",
        flush=True,
    )

    return {
        "open_width_m": float(open_w),
        "close_width_m": float(close_w),
        "pick_z_min_m": float(pick_z_min),
        "pick_z_guard_min_m": float(pick_z_guard_min),
        "pick_z_relative_m": float(pick_z_relative),
    }



def _xyz_from_obj(obj):

    if not isinstance(obj, dict):
        return None
    for k in ("base_xyz_m", "center_base_xyz_m", "xyz_m"):
        v = obj.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            try:
                return np.asarray([float(v[0]), float(v[1]), float(v[2])], dtype=float)
            except Exception:
                pass
    v = obj.get("safe_above_base_xyz_m")
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        try:
            return np.asarray([float(v[0]), float(v[1]), float(v[2])], dtype=float)
        except Exception:
            pass
    return None


def _obj_radius_m(obj, default=0.030):
    if not isinstance(obj, dict):
        return float(default)
    for k in ("radius_m", "radius_p90_m", "radius_median_m"):
        try:
            v = float(obj.get(k))
            if 0.005 <= v <= 0.120:
                return float(v)
        except Exception:
            pass
    shape = str(obj.get("object_shape") or obj.get("semantic_override") or "").lower()
    if "cylinder" in shape:
        return 0.035
    return float(default)


def _is_physical_place_object_key(key, obj):
    """
    Decide si una entrada de escena es un objeto físico que puede ocupar el destino.

        No deben contar como ocupantes las zonas/superficies/targets:
        yellow_target, blue_source, blue_zone, targets.*, etc.
    """
    k = str(key or "").lower()

    # Rechazo explícito de targets/superficies/zonas.
    reject_tokens = (
        "target",
        "zone",
        "source",
        "surface",
        "safe_probe",
        "planned",
        "diag",
        "stats",
        "bbox",
    )
    if any(t in k for t in reject_tokens):
        return False

    if not isinstance(obj, dict):
        return False

    sem = str(
        obj.get("semantic_override")
        or obj.get("object_shape")
        or obj.get("class")
        or obj.get("label")
        or ""
    ).lower()

    if any(t in sem for t in ("target", "zone", "source", "surface")):
        return False

    # Solo se aceptan objetos físicos conocidos u obstáculos geométricos.
    accept_tokens = (
        "orange",
        "cube",
        "die",
        "white",
        "cylinder",
        "obstacle",
        "local_ply",
    )

    return any(t in k for t in accept_tokens) or any(t in sem for t in accept_tokens)


def _collect_place_occupants(scene_data, selected_xyz, target_xyz):


    selected_xyz = np.asarray(selected_xyz, dtype=float).reshape(3)
    target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)

    occupants = []
    seen = set()

    # No se recorre scene_data completo porque también contiene targets y superficies.
    # Primero objects; luego aliases top-level de objetos físicos.
    candidate_items = []

    if isinstance(scene_data, dict):
        objects = scene_data.get("objects") or {}
        if isinstance(objects, dict):
            for key, obj in objects.items():
                candidate_items.append((f"objects.{key}", obj))

        for key in (
            "orange_cube",
            "orange_die",
            "white_cylinder",
            "local_ply_obstacle_0",
            "obstacle_0",
            "obstacle_1",
            "obstacle_2",
        ):
            if isinstance(scene_data.get(key), dict):
                candidate_items.append((key, scene_data[key]))

    for key, obj in candidate_items:
        if not _is_physical_place_object_key(key, obj):
            continue

        xyz = _xyz_from_obj(obj)
        if xyz is None:
            continue

        # Deduplicar objetos físicos detectados dos veces, por ejemplo:
        # local_ply_obstacle_0 y orange_cube pueden ser el mismo objeto.
        duplicate_of = None
        dedup_xy_m = float(os.environ.get("VLA_PLACE_OCCUPANT_DEDUP_XY_M", "0.025"))

        for prev in occupants:
            prev_xy = np.asarray(prev["xyz_m"][:2], dtype=float)
            if float(np.linalg.norm(xyz[:2] - prev_xy)) < dedup_xy_m:
                duplicate_of = prev
                break

        if duplicate_of is not None:
            # Si coinciden, se prefiere el nombre semántico frente a local_ply_obstacle.
            old_key = str(duplicate_of.get("key", ""))
            new_key = str(key)

            old_is_local = "local_ply" in old_key or "obstacle" in old_key
            new_is_semantic = any(t in new_key for t in ("orange", "white", "cube", "cylinder", "die"))

            if old_is_local and new_is_semantic:
                duplicate_of["key"] = new_key
                duplicate_of["xyz_m"] = [float(x) for x in xyz[:3]]
                duplicate_of["radius_m"] = float(_obj_radius_m(obj))
                duplicate_of["dedup_replaced_by"] = new_key
            else:
                duplicate_of.setdefault("dedup_ignored", []).append(new_key)
            continue

        sig = tuple(round(float(x), 3) for x in xyz[:2])
        if sig in seen:
            continue
        seen.add(sig)

        dist_to_selected = float(np.linalg.norm(xyz[:2] - selected_xyz[:2]))

        # El objeto seleccionado para el agarre se excluye. Si ya estaba cerca del target,
        # no debe bloquearse a sí mismo.
        if dist_to_selected < float(os.environ.get("VLA_PLACE_SELECTED_EXCLUDE_M", "0.070")):
            continue

        dist_to_target = float(np.linalg.norm(xyz[:2] - target_xyz[:2]))

        # Solo interesan objetos físicos cerca de la zona de destino.
        if dist_to_target > float(os.environ.get("VLA_PLACE_OCCUPANCY_NEAR_TARGET_M", "0.180")):
            continue

        occupants.append({
            "key": str(key),
            "xyz_m": [float(x) for x in xyz[:3]],
            "radius_m": float(_obj_radius_m(obj)),
            "dist_to_selected_m": dist_to_selected,
            "dist_to_target_m": dist_to_target,
        })

    return sorted(occupants, key=lambda o: o["dist_to_target_m"])


def _apply_global_place_occupancy(scene_data, selected_xyz, target_xyz, pre_place_xyz, place_xyz, retreat_xyz):
    """
    Elige un hueco libre dentro de la zona destino antes de ejecutar el place.

        No mueve el target de forma fija: genera candidatos alrededor del centro y
        elige el primero con separación suficiente frente a objetos ya presentes.
    """

    enabled = os.environ.get("VLA_GLOBAL_PLACE_OCCUPANCY", "1").strip() != "0"

    report = {
        "enabled": bool(enabled),
        "reason": "disabled",
        "selected_xyz_m": [float(x) for x in np.asarray(selected_xyz, dtype=float).reshape(3)],
        "target_xyz_m": [float(x) for x in np.asarray(target_xyz, dtype=float).reshape(3)],
        "original_place_xyz_m": [float(x) for x in np.asarray(place_xyz, dtype=float).reshape(3)],
        "occupants": [],
        "candidates": [],
        "selected_candidate": None,
    }

    if not enabled:
        return report

    selected_xyz = np.asarray(selected_xyz, dtype=float).reshape(3)
    target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)

    occupants = _collect_place_occupants(scene_data, selected_xyz, target_xyz)
    report["occupants"] = occupants

    # Separación física razonable:
    # distancia centro-centro >= radio_objeto_colocado + radio_objeto_a_colocar + margen.
    selected_radius = float(os.environ.get("VLA_PLACE_SELECTED_RADIUS_M", "0.030"))
    margin = float(os.environ.get("VLA_PLACE_OCCUPANCY_MARGIN_M", "0.018"))
    zone_radius = float(os.environ.get("VLA_PLACE_ZONE_RADIUS_M", "0.110"))

    base_xy = np.asarray(place_xyz[:2], dtype=float).copy()

    # Candidatos: centro y anillos hasta casi el borde útil de la zona.
    # Antes se quedaba corto con 0.090 y podía declarar imposible algo que no lo era.
    offsets = [(0.0, 0.0)]
    for r in (0.035, 0.055, 0.075, 0.095, 0.105):
        for deg in (0, 45, 90, 135, 180, 225, 270, 315):
            a = math.radians(deg)
            offsets.append((r * math.cos(a), r * math.sin(a)))

    best = None

    for idx, off in enumerate(offsets):
        cand_xy = base_xy + np.asarray(off, dtype=float)

        dist_from_target = float(np.linalg.norm(cand_xy - target_xyz[:2]))
        if dist_from_target > zone_radius:
            ok_zone = False
        else:
            ok_zone = True

        min_clear = 999.0
        nearest = None
        required_clearance = 0.0

        for occ in occupants:
            occ_xy = np.asarray(occ["xyz_m"][:2], dtype=float)
            occ_radius = float(occ.get("radius_m", 0.030))
            required = occ_radius + selected_radius + margin
            center_dist = float(np.linalg.norm(cand_xy - occ_xy))
            d = center_dist - required
            if d < min_clear:
                min_clear = d
                nearest = occ["key"]
                required_clearance = required

        # min_clear ya descuenta radio ocupante + radio objeto + margen.
        # Para real no basta con >0: exigimos margen extra medible.
        min_extra_clearance = float(os.environ.get("VLA_PLACE_MIN_EXTRA_CLEARANCE_M", "0.010"))
        ok_clear = min_clear >= min_extra_clearance if occupants else True

        item = {
            "idx": int(idx),
            "xy_m": [float(cand_xy[0]), float(cand_xy[1])],
            "offset_m": [float(off[0]), float(off[1])],
            "dist_from_target_m": float(dist_from_target),
            "min_clearance_m": float(min_clear),
            "required_center_distance_m": float(required_clearance),
            "min_extra_clearance_m": float(min_extra_clearance if occupants else 0.0),
            "nearest": nearest,
            "ok_zone": bool(ok_zone),
            "ok_clear": bool(ok_clear),
            "ok": bool(ok_zone and ok_clear),
        }

        # No se toma el primer hueco válido; se elige el candidato con más margen
        # frente al ocupante, penalizando alejarse demasiado del centro de zona.
        report["candidates"].append(item)

        if item["ok"]:
            # Cuando el candidato ya es seguro, se prioriza la cercanía al centro
            # de la zona. El margen solo desempata. Esto evita irse al borde.
            score = -float(item["dist_from_target_m"]) + 0.15 * float(item["min_clearance_m"])
            item["selection_score"] = float(score)
            if best is None or score > float(best.get("selection_score", -999.0)):
                best = item

    if best is None:
        report["reason"] = "no_free_place_candidate"
        print(
            "No hay hueco libre en la zona destino. "
            f"occupants={len(occupants)} margin={margin:.3f} zone_radius={zone_radius:.3f} "
            f"keys={[o.get('key') for o in occupants]}",
            flush=True,
        )
        raise RuntimeError("No hay hueco libre para place dentro de la zona destino")

    dx = float(best["offset_m"][0])
    dy = float(best["offset_m"][1])

    place_xyz[0] += dx
    place_xyz[1] += dy

    pre_place_xyz[0] += dx
    pre_place_xyz[1] += dy

    retreat_xyz[0] += dx
    retreat_xyz[1] += dy

    report["reason"] = "selected_free_candidate"
    report["selected_candidate"] = best
    report["place_xyz_m"] = [float(x) for x in place_xyz]

    print(
        f"occupants={len(occupants)} keys={[o.get('key') for o in occupants]} "
        f"offset=({dx:.3f},{dy:.3f}) "
        f"place_xy=({place_xyz[0]:.3f},{place_xyz[1]:.3f}) "
        f"min_clear={best['min_clearance_m']:.3f}",
        flush=True,
    )

    return report



def _preload_live_objects_for_place_occupancy(scene_data):
    """
    Carga la escena multiobjeto antes de calcular el place.

    Se toma el JSON live más reciente y se fusiona objects para que
    place_occupancy evalúe los ocupantes antes de generar candidatos.
    """
    report = {
        "enabled": True,
        "reason": "not_run",
        "live_json": None,
        "merged_objects": [],
        "objects_after_preload": [],
    }

    if not isinstance(scene_data, dict):
        report["reason"] = "scene_data_not_dict"
        return report

    roots = [
        Path("results/validacion_robot"),
        Path("results/validacion_robot"),
    ]

    candidates = []
    now = time.time()
    max_age_s = float(__import__("os").environ.get("VLA_PLACE_OCCUPANCY_LIVE_JSON_MAX_AGE_S", "300"))

    for root in roots:
        if not root.exists():
            continue
        for j in root.glob("escena_*/resultado_zonas.json"):
            try:
                age = now - j.stat().st_mtime
                if age > max_age_s:
                    continue
                data = json.loads(j.read_text(encoding="utf-8"))
                objects = data.get("objects") or {}
                if not isinstance(objects, dict) or not objects:
                    continue
                candidates.append((j.stat().st_mtime_ns, j, data))
            except Exception:
                continue

    if not candidates:
        report["reason"] = "no_recent_all_objects_json"
        scene_data["place_occupancy_preload"] = report
        print("no recent all_objects json", flush=True)
        return report

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, live_json, live_data = candidates[0]

    dst_objects = scene_data.setdefault("objects", {})
    if not isinstance(dst_objects, dict):
        dst_objects = {}
        scene_data["objects"] = dst_objects

    for key, obj in (live_data.get("objects") or {}).items():
        if not isinstance(obj, dict):
            continue
        before = dst_objects.get(key)
        dst_objects[key] = obj
        if before is None:
            report["merged_objects"].append(str(key))

    # También se mantienen los aliases top-level útiles.
    for key in ("orange_cube", "white_cylinder"):
        if key in dst_objects and isinstance(dst_objects[key], dict):
            scene_data[key] = dst_objects[key]

    report["reason"] = "ok"
    report["live_json"] = str(live_json)
    report["objects_after_preload"] = sorted(dst_objects.keys())

    scene_data["place_occupancy_preload"] = report

    print(
        f"live={live_json} merged={report['merged_objects']} "
        f"objects={report['objects_after_preload']}",
        flush=True,
    )

    return report



def _force_latest_camera_scene_json(current_json_path):
    """
    Usa el JSON canónico de la última captura multiobjeto.

    El ejecutor no toma como referencia un json_path interno tipo
    *_cilindro_blanco cuando existe la escena multiobjeto canónica. Si esa escena
    no existe o no contiene orange_cube y white_cylinder, se aborta.
    """

    current = Path(str(current_json_path)).expanduser()
    latest_file = Path("results/validacion_robot/ultima_escena.txt")

    if not latest_file.exists():
        raise RuntimeError(
            "No existe ultima_escena.txt. "
            "No ejecuto con JSON genérico ni con escena interna."
        )

    canon = Path(latest_file.read_text(encoding="utf-8").strip()).expanduser()

    if not canon.exists():
        raise RuntimeError(
            f"LATEST_CAMERA_SCENE_JSON apunta a archivo inexistente: {canon}"
        )

    data = json.loads(canon.read_text(encoding="utf-8"))
    objects = data.get("objects") or {}

    if not isinstance(objects, dict) or "orange_cube" not in objects or "white_cylinder" not in objects:
        raise RuntimeError(
            f"JSON canónico incompleto: "
            f"{canon} objects={list(objects.keys()) if isinstance(objects, dict) else objects}"
        )

    if data.get("selected_object") not in ("orange_cube", "white_cylinder"):
        raise RuntimeError(
            f"selected_object inválido en JSON canónico: "
            f"{canon} selected_object={data.get('selected_object')}"
        )

    if current.resolve() != canon.resolve():
        print(
            f"override json_path: requested={current} -> canonical={canon}",
            flush=True,
        )
    else:
        print(
            f"using canonical json_path={canon}",
            flush=True,
        )

    return canon



def _reconcile_selected_object_pick(scene_data):
    """
    Ajusta scene_data["orange_die"] al objeto seleccionado por el VLA/adaptador.

    En escenas multiobjeto, "orange_die" funciona como alias de compatibilidad y
    no tiene por qué representar el dado naranja físico.

    Si selected_object == white_cylinder:
        scene_data["orange_die"] = scene_data["objects"]["white_cylinder"]

    Si selected_object == orange_cube:
        scene_data["orange_die"] = scene_data["objects"]["orange_cube"]

    Si la escena está incompleta, se aborta en lugar de elegir una alternativa
    silenciosa.
    """

    if not isinstance(scene_data, dict):
        raise RuntimeError("scene_data no es dict")

    selected = scene_data.get("selected_object")
    objects = scene_data.get("objects") or {}

    if selected not in ("orange_cube", "white_cylinder"):
        raise RuntimeError(
            f"selected_object inválido o ausente: {selected!r}"
        )

    if not isinstance(objects, dict) or selected not in objects or not isinstance(objects[selected], dict):
        raise RuntimeError(
            f"objects[{selected!r}] no existe. "
            f"objects={list(objects.keys()) if isinstance(objects, dict) else objects}"
        )

    selected_obj = copy.deepcopy(objects[selected])

    xyz = (
        selected_obj.get("base_xyz_m")
        or selected_obj.get("center_base_xyz_m")
        or selected_obj.get("xyz_m")
    )

    if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
        raise RuntimeError(
            f"{selected} no tiene base_xyz_m válido: {xyz}"
        )

    xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2])]

    if any((not math.isfinite(v)) for v in xyz):
        raise RuntimeError(
            f"{selected} tiene xyz no finito: {xyz}"
        )

    selected_obj["base_xyz_m"] = xyz
    selected_obj["center_base_xyz_m"] = list(xyz)

    if "safe_above_base_xyz_m" not in selected_obj:
        selected_obj["safe_above_base_xyz_m"] = [xyz[0], xyz[1], max(0.18, xyz[2] + 0.12)]

    selected_obj["pick_alias"] = "orange_die"
    selected_obj["selected_object_key"] = selected

    old_orange_die = scene_data.get("orange_die")
    old_xyz = None
    if isinstance(old_orange_die, dict):
        old_xyz = (
            old_orange_die.get("base_xyz_m")
            or old_orange_die.get("center_base_xyz_m")
            or old_orange_die.get("xyz_m")
        )

    scene_data["orange_die"] = selected_obj
    scene_data[selected] = selected_obj
    objects[selected] = selected_obj
    scene_data["objects"] = objects

    report = {
        "enabled": True,
        "selected_object": selected,
        "selected_xyz_m": xyz,
        "previous_orange_die_xyz_m": old_xyz,
        "orange_die_now_aliases_selected_object": True,
    }
    scene_data["selected_object_pick_reconcile"] = report

    print(
        f"selected_object={selected} orange_die/base_xyz_m={xyz}",
        flush=True,
    )

    return scene_data


def _model_specific_transport(profile, lift_xyz, pre_place_xyz):
    """
    Genera transporte diferenciado a partir de la salida real del modelo.

        OpenVLA:
          vector único -> 2 puntos de transporte.

        PI0:
          secuencia temporal -> 5 puntos de transporte.

        Pick y place no se modifican.
    """

    report = {
        "enabled": False,
        "model": None,
        "decoder": None,
        "waypoints_xyz_m": [],
    }

    if not isinstance(profile, dict):
        report["reason"] = "profile_missing"
        return [], report

    experiment = str(
        profile.get("experiment_profile", "")
    )
    model_name = str(
        profile.get("model", "")
    ).strip().lower()

    report["model"] = model_name
    report["experiment_profile"] = experiment

    if experiment != "with_grasp":
        report["reason"] = "grasp_profile_required"
        return [], report

    evidence = profile.get("raw_model_evidence", {})
    raw = evidence.get("raw")

    start = np.asarray(lift_xyz, dtype=float).reshape(3)
    end = np.asarray(pre_place_xyz, dtype=float).reshape(3)

    def base_point(t):
        return (1.0 - t) * start + t * end

    def safe_point(point):
        point = np.asarray(point, dtype=float).reshape(3)

        point[0] = float(np.clip(point[0], -0.82, 0.82))
        point[1] = float(np.clip(point[1], -1.02, 0.12))

        # El transporte siempre queda por encima de las zonas
        # de contacto.
        point[2] = float(
            np.clip(max(point[2], 0.200), 0.200, 0.340)
        )

        return point

    points = []

    if model_name == "openvla":
        if not isinstance(raw, list) or len(raw) < 6:
            raise RuntimeError(
                "Raw OpenVLA inválido."
            )

        values = [float(v) for v in raw[:6]]

        signal = [
            math.tanh(80.0 * values[0]),
            math.tanh(80.0 * values[1]),
            math.tanh(80.0 * values[2]),
            math.tanh(30.0 * values[3]),
            math.tanh(30.0 * values[4]),
            math.tanh(30.0 * values[5]),
        ]

        sx, sy, sz, sr, sp, syaw = signal

        p1 = base_point(0.33)
        p1 += np.array([
            0.060 * sx,
            0.060 * sy,
            0.045 + 0.020 * abs(sz),
        ])

        p2 = base_point(0.68)
        p2 += np.array([
            0.045 * sr - 0.025 * syaw,
            0.045 * sp + 0.025 * syaw,
            0.030 + 0.020 * abs(sz),
        ])

        points = [
            safe_point(p1),
            safe_point(p2),
        ]

        report["decoder"] = (
            "openvla_vector_to_2_waypoints"
        )
        report["scaled_signal"] = signal
        report["raw_shape"] = [len(raw)]

    elif model_name == "pi0":
        if (
            not isinstance(raw, list)
            or not raw
            or not isinstance(raw[0], list)
        ):
            raise RuntimeError(
                "Raw PI0 inválido."
            )

        matrix = np.asarray(raw, dtype=float)

        if (
            matrix.ndim != 2
            or matrix.shape[0] < 5
            or matrix.shape[1] < 6
        ):
            raise RuntimeError(
                f"Shape PI0 inválida: "
                f"{matrix.shape}"
            )

        groups = np.array_split(matrix, 5)
        temporal_signals = []

        for index, group in enumerate(groups, start=1):
            mean = np.mean(group, axis=0)

            sx = math.tanh(4.0 * float(mean[0]))
            sy = math.tanh(4.0 * float(mean[1]))
            sz = math.tanh(4.0 * float(mean[2]))
            sr = math.tanh(3.0 * float(mean[3]))
            sp = math.tanh(3.0 * float(mean[4]))
            syaw = math.tanh(3.0 * float(mean[5]))

            t = float(index) / 6.0
            point = base_point(t)

            point += np.array([
                0.065 * sx + 0.020 * sr,
                0.065 * sy + 0.020 * sp,
                0.030
                + 0.025 * abs(sz)
                + 0.010 * abs(syaw),
            ])

            points.append(safe_point(point))

            temporal_signals.append({
                "segment": index,
                "signal": [
                    sx, sy, sz, sr, sp, syaw
                ],
            })

        report["decoder"] = (
            "pi0_temporal_sequence_to_5_waypoints"
        )
        report["temporal_signals"] = temporal_signals
        report["raw_shape"] = [
            int(matrix.shape[0]),
            int(matrix.shape[1]),
        ]

    else:
        raise RuntimeError(
            f"Modelo desconocido: "
            f"{model_name}"
        )

    report["enabled"] = True
    report["reason"] = "model_specific_transport_generated"
    report["waypoints_xyz_m"] = [
        [float(value) for value in point]
        for point in points
    ]

    print(
        f"model={model_name} "
        f"decoder={report['decoder']} "
        f"waypoints={len(points)}",
        flush=True,
    )

    for index, point in enumerate(points, start=1):
        print(
            f"waypoint_{index}="
            f"{[round(float(v), 5) for v in point]}",
            flush=True,
        )

    return points, report

def main():
    _apply_runtime_pick_gripper_config()

    if not _physical_execution_enabled():
        raise RuntimeError('La ejecución física requiere ALLOW_PHYSICAL_EXECUTION=YES, CONFIRM_PHYSICAL_EXECUTION=YES y CONFIRM_EXECUTION_RISK=YES')
    dry_run = os.environ.get("DRY_RUN", "1") != "0"
    confirm = os.environ.get("CONFIRM_PHYSICAL_EXECUTION") == "YES"
    execute_place = os.environ.get(
        "EXECUTE_PLACE",
        os.environ.get("PLACE_YELLOW", "0"),
    ).strip() == "1"

    # Parámetros conservadores.
    open_width = float(os.environ.get("OPEN_WIDTH_M", "0.120"))
    close_width = float(os.environ.get("CLOSE_WIDTH_M", "0.030"))

    # La bajada parte del punto alto ya validado.
    # PICK_ABOVE_DELTA_M se aplica sin quedar limitado por un max(...).
    pick_drop_from_above = float(os.environ.get("PICK_DROP_FROM_ABOVE_M", "0.095"))
    min_pick_z = float(os.environ.get("MIN_PICK_Z_M", "0.105"))
    lift_delta = float(os.environ.get("LIFT_DELTA_M", "0.060"))
    min_place_z = float(os.environ.get("MIN_PLACE_Z_M", "0.020"))
    if not (0.020 <= min_place_z <= 0.200):
        raise RuntimeError(f"MIN_PLACE_Z_M fuera de rango razonable: {min_place_z}")


    if not (0.030 <= pick_drop_from_above <= 0.230):
        raise RuntimeError(f"PICK_DROP_FROM_ABOVE_M fuera de rango seguro: {pick_drop_from_above}")

    if not (0.020 <= min_pick_z <= 0.200):
        raise RuntimeError(f"MIN_PICK_Z_M fuera de rango razonable: {min_pick_z}")

    json_path = base.latest_scene_json()
    json_path = _force_latest_camera_scene_json(json_path)
    scene_data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    scene_data = _reconcile_selected_object_pick(scene_data)
    yellow_obj = np.array(scene_data["yellow_target"]["base_xyz_m"], dtype=float)
    yellow_above = np.array(scene_data["yellow_target"]["safe_above_base_xyz_m"], dtype=float)


    orange_obj, orange_above = get_orange_points(scene_data)

    # Punto de agarre: mismo XY del dado, Z un poco por encima del centro/superficie detectada.
    # Se usa max para evitar una bajada excesiva.
    pick_z_auto = max(min_pick_z, orange_above[2] - pick_drop_from_above)

    pick_z_mode = os.environ.get("VLA_PICK_Z_MODE", "adaptive").strip().lower()
    pick_z_override = os.environ.get("PICK_Z_OVERRIDE_M", "").strip()

    # En modo normal, el pick_z se calcula desde Photoneo.
    # PICK_Z_OVERRIDE_M solo se respeta en modo experto o modo override explícito.
    if pick_z_override and (
        os.environ.get("VLA_EXPERT_ENV", "0") == "1" or pick_z_mode == "override"
    ):
        pick_z = float(pick_z_override)
        print(f"PICK_Z_OVERRIDE_M={pick_z:.4f} m")

    elif pick_z_mode == "adaptive":
        rel = float(os.environ.get("VLA_PICK_Z_RELATIVE_M", "-0.030"))
        zmin = max(float(min_pick_z), float(os.environ.get("VLA_PICK_Z_MIN_M", "0.045")))
        zmax = float(os.environ.get("VLA_PICK_Z_MAX_M", "0.075"))

        pick_z_raw = float(orange_obj[2]) + rel
        pick_z = float(np.clip(pick_z_raw, zmin, zmax))

        print(
            "PICK_Z adaptativo: "
            f"obj_z={float(orange_obj[2]):.4f} "
            f"rel={rel:.4f} raw={pick_z_raw:.4f} "
            f"clamp=[{zmin:.4f},{zmax:.4f}] -> pick_z={pick_z:.4f} m"
        )

    else:
        pick_z = pick_z_auto
        print(f"PICK_Z automático clásico={pick_z:.4f} m")

    if pick_z < min_pick_z:
        raise RuntimeError(
            f"PICK_Z_OVERRIDE_M={pick_z:.4f} está por debajo de MIN_PICK_Z_M={min_pick_z:.4f}"
        )

    pick_xyz = np.array([
        orange_above[0],
        orange_above[1],
        pick_z,
    ], dtype=float)

    lift_xyz = np.array([
        orange_above[0],
        orange_above[1],
        pick_xyz[2] + lift_delta,
    ], dtype=float)

    pre_pick_xyz = orange_above.copy()

    # En objetos alejados/laterales, el safe_above automático puede quedar muy alto
    # y producir IK mala aunque el pick sea alcanzable. Permitimos cap/override.
    pre_pick_z_override = os.environ.get("VLA_PRE_PICK_Z_M", "").strip()
    pre_pick_max_z = float(os.environ.get("VLA_PRE_PICK_MAX_Z_M", "0.220"))

    if pre_pick_z_override:
        pre_pick_xyz[2] = float(pre_pick_z_override)
        print(f"VLA_PRE_PICK_Z_M={pre_pick_xyz[2]:.4f} m")
    elif pre_pick_xyz[2] > pre_pick_max_z:
        print(
            f"Cap pre_pick_z: {pre_pick_xyz[2]:.4f} -> {pre_pick_max_z:.4f} m"
        )
        pre_pick_xyz[2] = pre_pick_max_z

    place_z_env = os.environ.get("PLACE_Z_OVERRIDE_M", "").strip()
    if place_z_env:
        place_z = float(place_z_env)
        print(f"PLACE_Z_OVERRIDE_M={place_z:.4f} m")
    else:
        place_z = float(pick_xyz[2])

    if place_z < min_place_z:
        # El límite cartesiano estático era demasiado conservador para la escena real.
        # Por defecto no se aborta aquí; la decisión queda en la IK vertical, los límites articulares,
        # controlador real, error de llegada y STOP_BEFORE_CLOSE.
        if os.environ.get("VLA_ENFORCE_STATIC_WORKSPACE_GUARD", "0").strip() == "1":
            raise RuntimeError(
                f"PLACE_Z_OVERRIDE_M={place_z:.4f} está por debajo de MIN_PLACE_Z_M={min_place_z:.4f}"
            )
        else:
            print(
                "Punto fuera del workspace estático conservador, "
                "pero VLA_ENFORCE_STATIC_WORKSPACE_GUARD=0; continúo y dejo decidir al IK/verticalidad/controlador.",
                flush=True,
            )

    def _check_pick_workspace(name, p):
        x, y, z = map(float, p)

        x_min = float(os.environ.get("VLA_WORKSPACE_X_MIN_M", "-0.90"))
        x_max = float(os.environ.get("VLA_WORKSPACE_X_MAX_M", "0.90"))
        y_min = float(os.environ.get("VLA_WORKSPACE_Y_MIN_M", "-1.10"))
        y_max = float(os.environ.get("VLA_WORKSPACE_Y_MAX_M", "0.20"))
        z_min = float(os.environ.get("VLA_WORKSPACE_Z_MIN_M", "0.045"))
        z_max = float(os.environ.get("VLA_WORKSPACE_Z_MAX_M", "0.45"))

        ok = (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max)

        if not ok:
            # Guard cartesiano estático conservador convertido en opcional.
            # Por defecto no aborta aquí: decide el IK vertical, el controlador real,
            # el error de llegada y STOP_BEFORE_CLOSE.
            if __import__("os").environ.get("VLA_ENFORCE_STATIC_WORKSPACE_GUARD", "0").strip() == "1":
                raise SystemExit(
                    ""
                    f"{name} fuera del workspace seguro: "
                    f"p={[x, y, z]} "
                    f"bounds=x[{x_min},{x_max}] y[{y_min},{y_max}] z[{z_min},{z_max}]. "
                    "No es un problema de top-down/lateral: el objeto está fuera del volumen "
                    "permitido o la detección/calibración ha dado una posición no ejecutable."
                )
            else:
                print(
                    "Punto fuera del workspace estático conservador; "
                    "VLA_ENFORCE_STATIC_WORKSPACE_GUARD=0, continúo y dejo decidir al IK/verticalidad/controlador.",
                    flush=True,
                )

        return np.array([x, y, z], dtype=float)

    pre_place_xyz = yellow_above.copy()
    pre_place_xyz[2] = max(float(pre_place_xyz[2]), 0.230)

    place_xyz = np.array([
        yellow_above[0],
        yellow_above[1],
        place_z,
    ], dtype=float)

    retreat_xyz = pre_place_xyz.copy()

    guidance_profile = _load_guidance_profile()
    guidance_applied = _apply_guidance_to_waypoints(guidance_profile, pre_pick_xyz, pick_xyz, lift_xyz, pre_place_xyz, place_xyz, retreat_xyz)

    gripper_guidance_report = _resolve_gripper_guidance(
        guidance_profile,
        open_width,
        close_width,
    )
    open_width = float(
        gripper_guidance_report["applied_open_width_m"]
    )
    close_width = float(
        gripper_guidance_report["applied_close_width_m"]
    )
    close_hold_sec = float(
        gripper_guidance_report["applied_close_hold_s"]
    )

    place_occupancy_preload_report = _preload_live_objects_for_place_occupancy(scene_data)

    place_occupancy_report = _apply_global_place_occupancy(
        scene_data=scene_data,
        selected_xyz=pick_xyz,
        target_xyz=yellow_obj,
        pre_place_xyz=pre_place_xyz,
        place_xyz=place_xyz,
        retreat_xyz=retreat_xyz,
    )


    # Guards de espacio seguro.
    # Primero check específico de pick con mensaje claro.
    _check_pick_workspace("pre_pick", pre_pick_xyz)
    _check_pick_workspace("pick", pick_xyz)
    _check_pick_workspace("lift", lift_xyz)

    # base.check_target() tiene límites cartesianos estáticos conservadores.
    # En esta configuración no se bloquea por un límite artificial en X/Y; deciden la IK,
    # verticalidad, controlador real, error de llegada y STOP_BEFORE_CLOSE.
    if os.environ.get("VLA_ENFORCE_STATIC_WORKSPACE_GUARD", "0").strip() == "1":
        base.check_target("pre_pick", pre_pick_xyz)
        base.check_target("lift", lift_xyz)
        base.check_target("pre_place", pre_place_xyz)
        base.check_target("retreat", retreat_xyz)
    else:
        print("base.check_target desactivado por VLA_ENFORCE_STATIC_WORKSPACE_GUARD=0", flush=True)

    if os.environ.get("VLA_ENFORCE_STATIC_WORKSPACE_GUARD", "0").strip() == "1":
        if not (-0.90 <= place_xyz[0] <= 0.90):
            raise RuntimeError(f"place x fuera de rango: {place_xyz[0]}")
        if not (-1.10 <= place_xyz[1] <= 0.20):
            raise RuntimeError(f"place y fuera de rango: {place_xyz[1]}")
    else:
        print("check estático X/Y de place desactivado; mantengo check Z.", flush=True)

    if not (min_place_z <= place_xyz[2] <= 0.45):
        raise RuntimeError(f"place z fuera de rango: {place_xyz[2]}")


    # El punto de pick puede estar más bajo que el guard general,
    # porque ya estamos haciendo agarre real. Aun así se limita.
    if os.environ.get("VLA_ENFORCE_STATIC_WORKSPACE_GUARD", "0").strip() == "1":
        if not (-0.90 <= pick_xyz[0] <= 0.90):
            raise RuntimeError(f"pick x fuera de rango: {pick_xyz[0]}")
        if not (-1.10 <= pick_xyz[1] <= 0.20):
            raise RuntimeError(f"pick y fuera de rango: {pick_xyz[1]}")
    else:
        print("check estático X/Y de pick desactivado; mantengo check Z.", flush=True)

    # El pick_z adaptativo puede quedar unas décimas por debajo del min_pick_z
    # después de aplicar offsets/perfil. El guard físico debe tener un margen
    # independiente, sin desactivar la seguridad.
    pick_z_guard_min = float(os.environ.get(
        "VLA_PICK_Z_GUARD_MIN_M",
        str(max(0.035, float(min_pick_z) - 0.007))
    ))

    if not (pick_z_guard_min <= pick_xyz[2] <= 0.45):
        raise RuntimeError(
            f"pick z fuera de rango: {pick_xyz[2]} "
            f"(guard_min={pick_z_guard_min}, min_pick_z={min_pick_z})"
        )

    # VLA_pick_place_noop_guard DISABLED
    print("guard pick/place cerca desactivado: permito tarea aunque el objeto ya esté cerca del destino.", flush=True)
    scene, model, data, tcp_site, tcp_id, qpos_ids, dof_ids = base.load_scene()

    rclpy.init()
    node = PickLiftNode()
    node.wait_state()

    q_home = node.current_q()
    real_tcp_current = node.current_tcp_xyz()

    base.set_robot_q(model, data, qpos_ids, q_home)
    sim_tcp_current, _, _ = base.get_site_pos_axis(data, tcp_id)

    offset = real_tcp_current - sim_tcp_current

    scene_data = _merge_live_objects_into_scene_for_grasp(scene_data, json_path)
    scene_data = _enforce_live_object_consistency_for_grasp(scene_data)

    dynamic_grasp_plan = _dynamic_grasp_plan(scene_data, orange_obj, q_home)
    dynamic_grasp_plan = _mark_dynamic_grasp_safety(dynamic_grasp_plan)
    q_home_for_ik = _apply_dynamic_grasp_seed(q_home, dynamic_grasp_plan)

    object_orientation_grasp = _estimate_object_yaw_from_ply(scene_data, orange_obj)
    q_home_for_ik = _apply_object_yaw_seed(q_home_for_ik, object_orientation_grasp)


    topdown_yaw_seeds, topdown_yaw_search = _build_topdown_yaw_seed_candidates(
        q_home_for_ik,
        object_orientation_grasp,
        dynamic_grasp_plan,
    )

    pre_pick_xyz, pick_xyz, lift_xyz, dynamic_waypoints = _apply_dynamic_waypoints_for_grasp(
        pre_pick_xyz,
        pick_xyz,
        lift_xyz,
        orange_obj,
        dynamic_grasp_plan,
    )

    # La orientación de la garra se decide dentro de la IK:
    # se prueban semillas de yaw equivalentes del objeto y solo se acepta una cadena
    # pre_pick -> pick -> lift donde la muñeca no tenga que corregir tarde durante la bajada.
    def _angle_dist_rad(a, ref):
        return abs(float(_wrap_to_near(float(a), float(ref))) - float(ref))

    def _object_yaw_cost_for_q(q_candidate, object_yaw_report):
        if not isinstance(object_yaw_report, dict):
            return None, None
        if not object_yaw_report.get("ok") or object_yaw_report.get("yaw_rad") is None:
            return None, None

        q_np = np.asarray(q_candidate, dtype=float).reshape(-1)
        if q_np.size < 6:
            return None, None

        wrist3 = float(q_np[5])
        yaw = float(object_yaw_report["yaw_rad"])

        # Cubo/dado: equivalencias cada 90 grados.
        costs = [
            _angle_dist_rad(yaw + k * np.pi / 2.0, wrist3)
            for k in range(-2, 3)
        ]
        best_rad = float(min(costs))
        return best_rad, float(np.degrees(best_rad))

    def _pre_pick_wrist_metrics(q_pre_c, q_pick_c, object_yaw_report):
        pre_np = np.asarray(q_pre_c, dtype=float).reshape(-1)
        pick_np = np.asarray(q_pick_c, dtype=float).reshape(-1)

        if pre_np.size < 6 or pick_np.size < 6:
            raise RuntimeError("q_pre/q_pick no tienen 6 articulaciones")

        wrist_delta_rad = float(np.max(np.abs(pick_np[3:6] - pre_np[3:6])))
        wrist3_delta_rad = float(_angle_dist_rad(pick_np[5], pre_np[5]))
        object_cost_rad, object_cost_deg = _object_yaw_cost_for_q(pick_np, object_yaw_report)

        return {
            "wrist_delta_pre_to_pick_rad": float(wrist_delta_rad),
            "wrist_delta_pre_to_pick_deg": float(np.degrees(wrist_delta_rad)),
            "wrist3_delta_pre_to_pick_rad": float(wrist3_delta_rad),
            "wrist3_delta_pre_to_pick_deg": float(np.degrees(wrist3_delta_rad)),
            "selected_grasp_yaw_rad": float(pick_np[5]),
            "selected_grasp_yaw_deg": float(np.degrees(pick_np[5])),
            "object_yaw_cost_rad": None if object_cost_rad is None else float(object_cost_rad),
            "object_yaw_cost_deg": None if object_cost_deg is None else float(object_cost_deg),
        }

    yaw_seed_list = list(topdown_yaw_seeds or [q_home_for_ik])
    wrist_limit_deg = float(os.environ.get("VLA_MAX_WRIST_DELTA_PRE_TO_PICK_DEG", "25"))

    yaw_ik_trials = []
    best_yaw_item = None

    for yaw_trial_idx, q_home_for_ik_candidate in enumerate(yaw_seed_list):
        try:
            (
                gos_i,
                sol_pre_i, pre_sim_i, q_pre_i,
                sol_pick_i, pick_sim_i, q_pick_i,
                sol_lift_i, lift_sim_i, q_lift_i,
                lift_steps_n_i,
                lift_step_xyzs_i, lift_step_solutions_i, lift_step_sims_i,
                q_lift_steps_i, lift_step_max_dq_i,
            ) = _solve_pick_lift_orientation_auto(
                model, data, tcp_id, qpos_ids, dof_ids,
                pre_pick_xyz, pick_xyz, lift_xyz,
                q_home, q_home_for_ik_candidate, offset,
                dynamic_grasp_plan,
            )

            metrics_i = _pre_pick_wrist_metrics(q_pre_i, q_pick_i, object_orientation_grasp)

            selected_i = gos_i.get("selected", {}) if isinstance(gos_i, dict) else {}
            base_score_i = float(selected_i.get("score", 999.0)) if isinstance(selected_i, dict) else 999.0

            object_penalty_i = 0.0
            if metrics_i["object_yaw_cost_deg"] is not None:
                object_penalty_i = 0.015 * float(metrics_i["object_yaw_cost_deg"])

            # Penalizamos explícitamente que la muñeca tenga que corregir durante la bajada.
            score_i = (
                base_score_i
                + 0.030 * float(metrics_i["wrist_delta_pre_to_pick_deg"])
                + 0.020 * float(metrics_i["wrist3_delta_pre_to_pick_deg"])
                + object_penalty_i
            )

            ok_chain_i = bool(
                sol_pre_i.get("ok")
                and sol_pick_i.get("ok")
                and sol_lift_i.get("ok")
            )

            accepted_by_wrist_i = bool(
                float(metrics_i["wrist_delta_pre_to_pick_deg"]) <= wrist_limit_deg
            )

            # Si el lock de pose completa está disponible, la orientación visual real
            # de la garra se decide por tcp_rot_delta_pre_to_pick_deg, no por el salto
            # articular bruto de wrist_1/wrist_2/wrist_3.
            tcp_rot_limit_deg_i = float(os.environ.get("VLA_MAX_TCP_ROT_DELTA_PRE_TO_PICK_DEG", "4"))

            tcp_rot_delta_i = None
            if isinstance(selected_i, dict) and selected_i.get("tcp_rot_delta_pre_to_pick_deg") is not None:
                tcp_rot_delta_i = float(selected_i.get("tcp_rot_delta_pre_to_pick_deg"))

            # En top-down estable no se usa la rotación completa del TCP como control
            # obligatorio. Esa métrica solo debe abortar cuando estemos probando
            # explícitamente el lock/orientación de garra por objeto.
            tcp_rot_guard_enabled_i = (
                os.environ.get("VLA_ENFORCE_TCP_ROT_GUARD", "0").strip() == "1"
                and os.environ.get("VLA_LOCK_PREPICK_ORIENTATION_FROM_PICK", "0").strip() == "1"
            )

            if tcp_rot_guard_enabled_i and tcp_rot_delta_i is not None:
                accepted_by_orientation_i = bool(tcp_rot_delta_i <= tcp_rot_limit_deg_i)
                orientation_accept_reason_i = "tcp_rot_delta_guard_enabled"
            else:
                accepted_by_orientation_i = True
                orientation_accept_reason_i = "topdown_only_tcp_rot_guard_disabled"

            trial_i = {
                "trial_idx": int(yaw_trial_idx),
                "seed_wrist3_rad": float(np.asarray(q_home_for_ik_candidate, dtype=float).reshape(-1)[5]),
                "seed_wrist3_deg": float(np.degrees(np.asarray(q_home_for_ik_candidate, dtype=float).reshape(-1)[5])),
                "ok_chain": bool(ok_chain_i),
                "accepted_by_wrist": bool(accepted_by_wrist_i),
                "accepted_by_orientation": bool(accepted_by_orientation_i),
                "orientation_accept_reason": orientation_accept_reason_i,
                "tcp_rot_delta_pre_to_pick_deg": tcp_rot_delta_i,
                "tcp_rot_limit_deg": float(tcp_rot_limit_deg_i),
                "score": float(score_i),
                "base_score": float(base_score_i),
                "wrist_limit_deg": float(wrist_limit_deg),
                **metrics_i,
                "selected_axis_name": gos_i.get("selected_axis_name") if isinstance(gos_i, dict) else None,
            }

            yaw_ik_trials.append(trial_i)

            item_i = {
                "trial": trial_i,
                "ok_chain": bool(ok_chain_i),
                "accepted_by_wrist": bool(accepted_by_wrist_i),
                "accepted_by_orientation": bool(accepted_by_orientation_i),
                "score": float(score_i),
                "bundle": (
                    gos_i,
                    sol_pre_i, pre_sim_i, q_pre_i,
                    sol_pick_i, pick_sim_i, q_pick_i,
                    sol_lift_i, lift_sim_i, q_lift_i,
                    lift_steps_n_i,
                    lift_step_xyzs_i, lift_step_solutions_i, lift_step_sims_i,
                    q_lift_steps_i, lift_step_max_dq_i,
                ),
            }

            if best_yaw_item is None:
                best_yaw_item = item_i
            elif item_i["ok_chain"] and not best_yaw_item["ok_chain"]:
                best_yaw_item = item_i
            elif item_i["ok_chain"] == best_yaw_item["ok_chain"]:
                if item_i["accepted_by_orientation"] and not best_yaw_item["accepted_by_orientation"]:
                    best_yaw_item = item_i
                elif item_i["accepted_by_orientation"] == best_yaw_item["accepted_by_orientation"]:
                    if item_i["score"] < best_yaw_item["score"]:
                        best_yaw_item = item_i

        except Exception as e:
            yaw_ik_trials.append({
                "trial_idx": int(yaw_trial_idx),
                "ok_chain": False,
                "accepted_by_wrist": False,
                "error": repr(e),
                "score": 999.0,
                "wrist_limit_deg": float(wrist_limit_deg),
            })

    topdown_yaw_search["ik_selection_enabled"] = True
    topdown_yaw_search["ik_selection_reason"] = "object_yaw_seeds_solved_inside_ik"
    topdown_yaw_search["ik_trials"] = yaw_ik_trials
    topdown_yaw_search["wrist_limit_deg"] = float(wrist_limit_deg)

    if best_yaw_item is None or not best_yaw_item["ok_chain"]:
        raise RuntimeError(f"Ningún yaw candidato produjo una cadena IK válida. trials={yaw_ik_trials}")

    topdown_yaw_search["selected_ik_trial"] = best_yaw_item["trial"]
    topdown_yaw_search["selected_trial_idx"] = int(best_yaw_item["trial"]["trial_idx"])
    topdown_yaw_search["selected_wrist3_rad"] = float(best_yaw_item["trial"]["selected_grasp_yaw_rad"])
    topdown_yaw_search["selected_wrist3_deg"] = float(best_yaw_item["trial"]["selected_grasp_yaw_deg"])
    topdown_yaw_search["selected_grasp_yaw_deg"] = float(best_yaw_item["trial"]["selected_grasp_yaw_deg"])
    topdown_yaw_search["wrist_delta_pre_to_pick_deg"] = float(best_yaw_item["trial"]["wrist_delta_pre_to_pick_deg"])
    topdown_yaw_search["wrist3_delta_pre_to_pick_deg"] = float(best_yaw_item["trial"]["wrist3_delta_pre_to_pick_deg"])

    if not best_yaw_item["accepted_by_orientation"]:
        tr_bad = best_yaw_item["trial"]
        if tr_bad.get("tcp_rot_delta_pre_to_pick_deg") is not None:
            raise RuntimeError(
                "La mejor solución IK todavía cambia demasiado la orientación completa de la garra durante la bajada: "
                f"{tr_bad.get('tcp_rot_delta_pre_to_pick_deg'):.1f} deg > {tr_bad.get('tcp_rot_limit_deg'):.1f} deg"
            )
        raise RuntimeError(
            "La mejor solución IK todavía reorienta demasiado la muñeca durante la bajada y no hay métrica TCP disponible: "
            f"{tr_bad.get('wrist_delta_pre_to_pick_deg'):.1f} deg > {wrist_limit_deg:.1f} deg"
        )

    (
        grasp_orientation_search,
        sol_pre, pre_sim, q_pre,
        sol_pick, pick_sim, q_pick,
        sol_lift, lift_sim, q_lift,
        lift_steps_n,
        lift_step_xyzs, lift_step_solutions, lift_step_sims,
        q_lift_steps, lift_step_max_dq,
    ) = best_yaw_item["bundle"]

    if isinstance(grasp_orientation_search, dict):
        grasp_orientation_search["object_yaw_ik_selection"] = best_yaw_item["trial"]

    print(
        f"selected_trial={best_yaw_item['trial']['trial_idx']} "
        f"yaw={best_yaw_item['trial']['selected_grasp_yaw_deg']:.1f}deg "
        f"wrist_delta={best_yaw_item['trial']['wrist_delta_pre_to_pick_deg']:.1f}deg "
        f"score={best_yaw_item['score']:.4f}",
        flush=True,
    )

    model_transport_xyzs, model_transport_report = (
        _model_specific_transport(
            guidance_profile,
            lift_xyz,
            pre_place_xyz,
        )
    )

    q_model_transport_steps = []
    model_transport_ik = []

    transport_seed = np.asarray(q_lift, dtype=float)
    transport_previous_q = np.asarray(q_lift, dtype=float)

    max_transport_dq = float(
        os.environ.get(
            "VLA_MODEL_TRANSPORT_MAX_DQ_RAD",
            "0.70",
        )
    )

    for transport_index, transport_xyz in enumerate(
        model_transport_xyzs,
        start=1,
    ):
        transport_sol, transport_sim = solve_point(
            model,
            data,
            tcp_id,
            qpos_ids,
            dof_ids,
            transport_xyz,
            transport_seed,
            offset,
        )

        if not transport_sol.get("ok", False):
            raise RuntimeError(
                "IK inválida en "
                f"waypoint {transport_index}: "
                f"{transport_sol}"
            )

        transport_q = np.asarray(
            transport_sol["q"],
            dtype=float,
        )

        step_dq = float(
            np.max(
                np.abs(
                    transport_q - transport_previous_q
                )
            )
        )

        if step_dq > max_transport_dq:
            raise RuntimeError(
                "Salto articular "
                f"en waypoint {transport_index}: "
                f"{step_dq:.4f} > "
                f"{max_transport_dq:.4f} rad"
            )

        q_model_transport_steps.append(transport_q)

        model_transport_ik.append({
            "index": int(transport_index),
            "ok": True,
            "pos_err_m": float(
                transport_sol.get("pos_err_m", 0.0)
            ),
            "topdown_dot": (
                float(transport_sol["topdown_dot"])
                if transport_sol.get("topdown_dot") is not None
                else None
            ),
            "max_abs_dq_from_previous_rad": step_dq,
            "sim_xyz_m": [
                float(value)
                for value in transport_sim
            ],
        })

        transport_seed = transport_q
        transport_previous_q = transport_q

    sol_pre_place, pre_place_sim = solve_point(
        model,
        data,
        tcp_id,
        qpos_ids,
        dof_ids,
        pre_place_xyz,
        transport_seed,
        offset,
    )
    q_pre_place = sol_pre_place["q"]

    if q_model_transport_steps:
        result_dq = float(
            np.max(
                np.abs(
                    np.asarray(q_pre_place, dtype=float)
                    - q_model_transport_steps[-1]
                )
            )
        )

        if result_dq > max_transport_dq:
            raise RuntimeError(
                "Salto result hacia "
                f"pre_place: {result_dq:.4f} > "
                f"{max_transport_dq:.4f} rad"
            )

        model_transport_report[
            "max_abs_dq_to_pre_place_rad"
        ] = result_dq

    model_transport_report["ik"] = model_transport_ik

    sol_place, place_sim = solve_point(model, data, tcp_id, qpos_ids, dof_ids, place_xyz, q_pre_place, offset)
    q_place = sol_place["q"]

    sol_retreat, retreat_sim = solve_point(model, data, tcp_id, qpos_ids, dof_ids, retreat_xyz, q_place, offset)
    q_retreat = sol_retreat["q"]

    (
        q_pre,
        q_pick,
        q_lift,
        q_pre_place,
        q_place,
        q_retreat,
        q_lift_steps,
        lift_step_max_dq,
        phase_joint_guidance_report,
    ) = _apply_phase_joint_guidance(
        profile=guidance_profile,
        model=model,
        data=data,
        tcp_id=tcp_id,
        qpos_ids=qpos_ids,
        q_pre=q_pre,
        q_pick=q_pick,
        q_lift=q_lift,
        q_pre_place=q_pre_place,
        q_place=q_place,
        q_retreat=q_retreat,
        q_lift_steps=q_lift_steps,
        target_sims={
            "q_pre": pre_sim,
            "q_pick": pick_sim,
            "q_lift": lift_sim,
            "q_pre_place": pre_place_sim,
            "q_place": place_sim,
            "q_retreat": retreat_sim,
        },
        lift_step_sims=lift_step_sims,
    )

    def _q_to_vec(q):
        if isinstance(q, dict):
            return np.array([float(v) for v in q.values()], dtype=float)
        return np.asarray(q, dtype=float).reshape(-1)

    def _q_to_json(q):
        if isinstance(q, dict):
            return {str(k): float(v) for k, v in q.items()}
        return [float(x) for x in np.asarray(q, dtype=float).reshape(-1)]

    def _maxdq_place(qa, qb):
        a = _q_to_vec(qa)
        b = _q_to_vec(qb)
        if a.shape != b.shape:
            raise RuntimeError(f"Comparación articular incompatible: {a.shape} vs {b.shape}")
        return float(np.max(np.abs(a - b)))

    if _maxdq_place(q_lift, q_pre_place) > 2.00:
        raise RuntimeError(f"Salto lift->pre_place demasiado grande: {_maxdq_place(q_lift, q_pre_place)}")
    if _maxdq_place(q_pre_place, q_place) > 1.00:
        raise RuntimeError(f"Salto pre_place->place demasiado grande: {_maxdq_place(q_pre_place, q_place)}")
    if _maxdq_place(q_place, q_retreat) > 1.00:
        raise RuntimeError(f"Salto place->retreat demasiado grande: {_maxdq_place(q_place, q_retreat)}")


    # Reorientación temprana: llegar al pre-pick con la muñeca ya preparada.
    # Reorientación temprana: el pre-pick usa ya la muñeca del pick.
    result = {
        "json_path": str(json_path),
        "scene_xml": scene,
        "tcp_site": tcp_site,
        "dry_run": dry_run,
        "guidance_profile": guidance_applied,
        "experiment_profile": (
            guidance_profile.get("experiment_profile")
            if isinstance(guidance_profile, dict)
            else None
        ),
        "phase_joint_guidance": phase_joint_guidance_report,
        "model_specific_transport": model_transport_report,
        "model_transport_xyz_m": [
            [float(value) for value in point]
            for point in model_transport_xyzs
        ],
        "q_model_transport_steps": [
            _q_to_json(q_value)
            for q_value in q_model_transport_steps
        ],
        "gripper_guidance": gripper_guidance_report,
        "close_hold_sec": close_hold_sec,
        "dynamic_grasp": dynamic_grasp_plan,
        "object_orientation_grasp": object_orientation_grasp,
        "topdown_yaw_search": topdown_yaw_search,
        "dynamic_waypoints": dynamic_waypoints,
        "grasp_orientation_search": grasp_orientation_search,
"live_object_consistency": scene_data.get("live_object_consistency"),
        "grasp_live_objects_merge": scene_data.get("grasp_live_objects_merge"),
        "local_ply_grasp_obstacles": scene_data.get("local_ply_grasp_obstacles"),
        "place_occupancy_preload": place_occupancy_preload_report,
        "place_occupancy": place_occupancy_report,
        "current_real_tcp_xyz_m": [float(x) for x in real_tcp_current],
        "orange_obj_base_xyz_m": [float(x) for x in orange_obj],
        "pre_pick_xyz_m": [float(x) for x in pre_pick_xyz],
        "pick_xyz_m": [float(x) for x in pick_xyz],
        "lift_xyz_m": [float(x) for x in lift_xyz],
        "open_width_m": open_width,
        "close_width_m": close_width,
        "pick_drop_from_above_m": pick_drop_from_above,
        "min_pick_z_m": min_pick_z,
        "pick_z_override_m": float(os.environ.get("PICK_Z_OVERRIDE_M", "nan")) if os.environ.get("PICK_Z_OVERRIDE_M", "").strip() else None,
        "object_orientation_grasp": object_orientation_grasp,
        "topdown_yaw_search": topdown_yaw_search,
        "grasp_orientation_search": grasp_orientation_search,
        "ik": {
            "pre_err_m": float(sol_pre["pos_err_m"]),
            "pre_topdown_dot": float(sol_pre["topdown_dot"]),
            "pre_ok": bool(sol_pre["ok"]),
            "pick_err_m": float(sol_pick["pos_err_m"]),
            "pick_topdown_dot": float(sol_pick["topdown_dot"]),
            "pick_ok": bool(sol_pick["ok"]),
            "lift_err_m": float(sol_lift["pos_err_m"]),
            "lift_topdown_dot": float(sol_lift["topdown_dot"]),
            "lift_ok": bool(sol_lift["ok"]),
            "lift_steps_n": int(lift_steps_n),
            "lift_step_max_dq": [float(x) for x in lift_step_max_dq],
            "max_abs_dq_pick_to_lift": float(max(lift_step_max_dq)),
            "max_abs_dq_pick_to_lift_total": float(np.max(np.abs(q_lift - q_pick))),
            "max_abs_dq_home_to_pre": float(np.max(np.abs(q_pre - q_home))),
            "max_abs_dq_pre_to_pick": float(np.max(np.abs(q_pick - q_pre))),
            "max_abs_wrist_dq_pre_to_pick": float(np.max(np.abs(q_pick[3:6] - q_pre[3:6]))),
            "wrist_delta_pre_to_pick_deg": float(np.degrees(np.max(np.abs(q_pick[3:6] - q_pre[3:6])))),
            "wrist3_delta_pre_to_pick_deg": float(np.degrees(abs(q_pick[5] - q_pre[5]))),
            "selected_grasp_yaw_deg": float(np.degrees(q_pick[5])),
            "tcp_rot_delta_pre_to_pick_deg": (
                float(grasp_orientation_search.get("selected", {}).get("tcp_rot_delta_pre_to_pick_deg"))
                if isinstance(grasp_orientation_search, dict)
                and isinstance(grasp_orientation_search.get("selected"), dict)
                and grasp_orientation_search.get("selected", {}).get("tcp_rot_delta_pre_to_pick_deg") is not None
                else None
            ),
            "object_yaw_deg": (
                float(object_orientation_grasp.get("yaw_deg"))
                if isinstance(object_orientation_grasp, dict) and object_orientation_grasp.get("yaw_deg") is not None
                else None
            ),
            "object_yaw_cost_deg": (
                float(topdown_yaw_search.get("selected_ik_trial", {}).get("object_yaw_cost_deg"))
                if isinstance(topdown_yaw_search, dict)
                and isinstance(topdown_yaw_search.get("selected_ik_trial"), dict)
                and topdown_yaw_search.get("selected_ik_trial", {}).get("object_yaw_cost_deg") is not None
                else None
            ),
        },
        "q_home": dict(zip(base.JOINTS, [float(x) for x in q_home])),
        "q_pre_pick": dict(zip(base.JOINTS, [float(x) for x in q_pre])),
        "q_pick": dict(zip(base.JOINTS, [float(x) for x in q_pick])),
        "q_lift": dict(zip(base.JOINTS, [float(x) for x in q_lift])),
        "lift_steps_xyz_m": [[float(v) for v in xyz] for xyz in lift_step_xyzs],
        "q_lift_steps": [
            dict(zip(base.JOINTS, [float(x) for x in q_step]))
            for q_step in q_lift_steps
        ],
        "safety": {
            "uses_photoneo_result": True,
            "uses_mujoco_ik": True,
            "uses_topdown_verticalization": bool(grasp_orientation_search.get("selected_axis_name") == "topdown"),
            "uses_axis_orientation_search": bool(grasp_orientation_search.get("enabled")),
            "uses_follow_joint_trajectory": True,
            "uses_real_3fg15_topic": True,
            "does_not_place_yet": True,
            "returns_home": True,
        },
    }

    out = Path(json_path).parent / "plan_ejecucion.json"

    print("Plan de ejecución con 3FG15")
    result.update({
        "yellow_obj_base_xyz_m": [float(x) for x in yellow_obj],
        "pre_place_xyz_m": [float(x) for x in pre_place_xyz],
        "place_xyz_m": [float(x) for x in place_xyz],
        "retreat_xyz_m": [float(x) for x in retreat_xyz],
        "place_z_override_m": float(os.environ.get("PLACE_Z_OVERRIDE_M", "nan")) if os.environ.get("PLACE_Z_OVERRIDE_M", "").strip() else None,
        "min_place_z_m": min_place_z,
        "q_pre_place": _q_to_json(q_pre_place),
        "q_place": _q_to_json(q_place),
        "q_retreat": _q_to_json(q_retreat),
    })

    def _sol_float(sol, *keys):
        if isinstance(sol, dict):
            for k in keys:
                if k in sol and sol[k] is not None:
                    try:
                        return float(sol[k])
                    except Exception:
                        pass
        return None

    def _sim_err(target, actual):
        try:
            t = np.asarray(target, dtype=float).reshape(-1)
            a = np.asarray(actual, dtype=float).reshape(-1)
            if t.shape == a.shape:
                return float(np.linalg.norm(a - t))
        except Exception:
            pass
        return None

    def _sol_err(sol, target, actual):
        v = _sol_float(sol, "err_m", "error_m", "pos_err_m", "position_error_m", "err")
        if v is not None:
            return v
        return _sim_err(target, actual)

    def _sol_dot(sol):
        return _sol_float(sol, "topdown_dot", "dot", "vertical_dot")

    def _sol_ok(sol, target, actual):
        if isinstance(sol, dict) and "ok" in sol:
            return bool(sol["ok"])
        e = _sol_err(sol, target, actual)
        if e is not None:
            return bool(e < 0.04)
        return bool(isinstance(sol, dict) and "q" in sol)

    result["ik"].update({
        "pre_place_err_m": _sol_err(sol_pre_place, pre_place_xyz, pre_place_sim),
        "pre_place_topdown_dot": _sol_dot(sol_pre_place),
        "pre_place_ok": _sol_ok(sol_pre_place, pre_place_xyz, pre_place_sim),
        "place_err_m": _sol_err(sol_place, place_xyz, place_sim),
        "place_topdown_dot": _sol_dot(sol_place),
        "place_ok": _sol_ok(sol_place, place_xyz, place_sim),
        "retreat_err_m": _sol_err(sol_retreat, retreat_xyz, retreat_sim),
        "retreat_topdown_dot": _sol_dot(sol_retreat),
        "retreat_ok": _sol_ok(sol_retreat, retreat_xyz, retreat_sim),
        "max_abs_dq_lift_to_pre_place": _maxdq_place(q_lift, q_pre_place),
        "max_abs_dq_pre_place_to_place": _maxdq_place(q_pre_place, q_place),
        "max_abs_dq_place_to_retreat": _maxdq_place(q_place, q_retreat),
    })

    if execute_place:
        result["safety"]["does_not_place_yet"] = False

    print(json.dumps(result, indent=2, ensure_ascii=False))
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Plan completo guardado en: {out}")
    print()
    print("Plan guardado en:", out)

    if not (sol_pre["ok"] and sol_pick["ok"] and sol_lift["ok"]):
        raise SystemExit("IK no válida. No se mueve.")

    max_dq_home_to_pre_allowed = float(os.environ.get("VLA_MAX_DQ_HOME_TO_PRE", "1.4"))
    result["ik"]["max_abs_dq_home_to_pre_allowed"] = float(max_dq_home_to_pre_allowed)

    if result["ik"]["max_abs_dq_home_to_pre"] > max_dq_home_to_pre_allowed:
        raise SystemExit("Salto home->pre_pick demasiado grande.")

    max_tcp_rot_delta_pre_to_pick_deg = float(os.environ.get("VLA_MAX_TCP_ROT_DELTA_PRE_TO_PICK_DEG", "4"))
    tcp_rot_delta_pre_to_pick_deg = result["ik"].get("tcp_rot_delta_pre_to_pick_deg", None)

    enforce_tcp_rot_guard = (
        os.environ.get("VLA_ENFORCE_TCP_ROT_GUARD", "0").strip() == "1"
        and os.environ.get("VLA_LOCK_PREPICK_ORIENTATION_FROM_PICK", "0").strip() == "1"
    )

    if (
        enforce_tcp_rot_guard
        and tcp_rot_delta_pre_to_pick_deg is not None
        and tcp_rot_delta_pre_to_pick_deg > max_tcp_rot_delta_pre_to_pick_deg
    ):
        raise RuntimeError(
            "Plan rechazado: la orientación completa de la garra cambia demasiado durante la bajada "
            f"({tcp_rot_delta_pre_to_pick_deg:.1f} deg > {max_tcp_rot_delta_pre_to_pick_deg:.1f} deg)"
        )

    max_wrist_delta_pre_to_pick_deg = float(os.environ.get("VLA_MAX_WRIST_DELTA_PRE_TO_PICK_DEG", "25"))
    enforce_wrist_delta_guard = os.environ.get("VLA_ENFORCE_WRIST_DELTA_GUARD", "0").strip() == "1"

    if (
        enforce_wrist_delta_guard
        and result["ik"].get("wrist_delta_pre_to_pick_deg", 999.0) > max_wrist_delta_pre_to_pick_deg
    ):
        raise RuntimeError(
            "Plan rechazado: la muñeca se reorienta demasiado durante la bajada "
            f"({result['ik'].get('wrist_delta_pre_to_pick_deg'):.1f} deg > {max_wrist_delta_pre_to_pick_deg:.1f} deg)"
        )

    # Se usa top-down vertical puro.
    # No orientación por yaw de objeto, no lateral, no lock experimental.
    # Si el TCP no está suficientemente vertical en pre/pick/lift, no se ejecuta.
    if os.environ.get("VLA_STRICT_VERTICAL_TOPDOWN", "1").strip() == "1":
        import math

        max_tilt_deg = float(os.environ.get("VLA_MAX_TOPDOWN_TILT_DEG", "2.0"))
        min_dot = math.cos(math.radians(max_tilt_deg))

        checks = [
            ("pre_pick", "pre_topdown_dot"),
            ("pick", "pick_topdown_dot"),
            ("lift", "lift_topdown_dot"),
        ]

        for phase, key in checks:
            dot = result["ik"].get(key, None)
            if dot is None:
                raise RuntimeError(f"Falta {key}; no ejecuto sin validar verticalidad.")

            dot = float(dot)
            dot = max(-1.0, min(1.0, dot))
            tilt_deg = math.degrees(math.acos(dot))

            result["ik"][phase + "_tilt_deg_guard"] = float(tilt_deg)

            if tilt_deg > max_tilt_deg:
                raise RuntimeError(
                    f"{phase} no está vertical: "
                    f"topdown_dot={dot:.6f}, tilt={tilt_deg:.2f} deg > {max_tilt_deg:.2f} deg"
                )

    if result["ik"]["max_abs_dq_pre_to_pick"] > 0.45:
        raise SystemExit("Bajada pre_pick->pick demasiado grande.")

    max_dq_pick_to_lift_allowed = float(os.environ.get("VLA_MAX_DQ_PICK_TO_LIFT", "0.45"))
    result["ik"]["max_abs_dq_pick_to_lift_allowed"] = float(max_dq_pick_to_lift_allowed)

    if result["ik"]["max_abs_dq_pick_to_lift"] > max_dq_pick_to_lift_allowed:
        raise SystemExit("Lift articular demasiado grande.")

    if dry_run:
        print("DRY_RUN=1. No se mueve robot ni pinza.")
        node.destroy_node()
        rclpy.shutdown()
        return

    if not confirm:
        raise SystemExit("Falta CONFIRM_PHYSICAL_EXECUTION=YES")

    if dynamic_grasp_plan.get("safe_to_pick") is False:
        print(
            "Agarre dinámico inseguro: "
            f"{dynamic_grasp_plan.get('safety_reason')} "
            f"clearance={dynamic_grasp_plan.get('nearest_clearance_m')} "
            f"min_required={dynamic_grasp_plan.get('min_required_clearance_m')}",
            flush=True,
        )

        # Compatibilidad entre versiones:
        # en algunas versiones el modo se llama DRY_RUN y en otras dry_run.
        # En dry-run solo informa; en real bloquea antes de mover.
        _is_dry_run = bool(locals().get("DRY_RUN", locals().get("dry_run", False)))

        if (not _is_dry_run) and os.environ.get("VLA_ALLOW_UNSAFE_DYNAMIC_GRASP", "0") != "1":
            raise SystemExit("Agarre dinámico inseguro. No se mueve el robot.")

    print("1. Abrir pinza")
    node.set_gripper_width(open_width, wait_sec=2.5)

    print("2. Ir a pre-pick")
    # La muñeca se orienta como en el pick antes de acercar el brazo a la escena.
    # Así la garra no termina de girar justo encima del objeto.

    code, msg = node.send_goal(q_pre, duration_sec=7)
    print("result pre_pick:", code, repr(msg))
    if code != 0:
        raise SystemExit("Falló pre_pick.")

    print("3. Bajar a pick")
    code, msg = node.send_goal(q_pick, duration_sec=5)
    print("result pick:", code, repr(msg))
    if code != 0:
        raise SystemExit("Falló bajada a pick.")

    time.sleep(0.5)
    node.wait_state()
    actual_pick = node.current_tcp_xyz()
    actual_pick_err = float(np.linalg.norm(actual_pick - pick_xyz))
    print("actual_pick_xyz_m:", actual_pick.tolist())
    print("target_pick_xyz_m:", pick_xyz.tolist())
    print("actual_pick_error_m:", actual_pick_err)

    pick_max_error_m = float(os.environ.get("PICK_MAX_ERROR_M", "0.035"))


    print("pick_max_error_m:", pick_max_error_m)


    if actual_pick_err > pick_max_error_m:


        print(f"Error de posicionamiento en pick > {pick_max_error_m*100:.1f} cm; retorno a home sin cerrar la pinza.")
        node.send_goal(q_home, duration_sec=8)
        raise SystemExit("Error real en pick demasiado alto.")

    if os.environ.get("STOP_BEFORE_CLOSE", "0") == "1":
        print("STOP_BEFORE_CLOSE=1: retorno a home sin cerrar la pinza.")
        node.send_goal(q_home, duration_sec=8)
        result["execution"] = {
            "actual_pick_xyz_m": [float(x) for x in actual_pick],
            "actual_pick_error_m": actual_pick_err,
            "stopped_before_close": True,
            "returned_home": True,
        }
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        node.destroy_node()
        rclpy.shutdown()
        print("Prueba de bajada a pick terminada sin cerrar.")
        return

    print("4. Cerrar pinza")
    node.set_gripper_width(
        close_width,
        wait_sec=close_hold_sec,
    )

    print("5. Elevación segmentada")
    for i, q_step in enumerate(q_lift_steps, start=1):
        duration = 3 if i < len(q_lift_steps) else 5
        code, msg = node.send_goal(q_step, duration_sec=duration)
        print(f"result lift step {i}/{len(q_lift_steps)}:", code, repr(msg))
        if code != 0:
            raise SystemExit(f"Falló lift step {i}/{len(q_lift_steps)}.")

    time.sleep(0.5)
    node.wait_state()
    actual_lift = node.current_tcp_xyz()
    actual_lift_err = float(np.linalg.norm(actual_lift - lift_xyz))

    print("actual_lift_xyz_m:", actual_lift.tolist())
    print("target_lift_xyz_m:", lift_xyz.tolist())
    print("actual_lift_error_m:", actual_lift_err)

    if execute_place:
        if q_model_transport_steps:
            print(
                "Trayectoria adaptada desde la salida del modelo",
                flush=True,
            )

            duration = float(
                os.environ.get(
                    "VLA_MODEL_TRANSPORT_STEP_DURATION_S",
                    "3.0",
                )
            )

            for transport_index, transport_q in enumerate(
                q_model_transport_steps,
                start=1,
            ):
                code, msg = node.send_goal(
                    transport_q,
                    duration_sec=duration,
                )

                print(
                    "result model_transport "
                    f"{transport_index}/"
                    f"{len(q_model_transport_steps)}:",
                    code,
                    repr(msg),
                )

                if code != 0:
                    node.send_goal(q_home, duration_sec=8)

                    raise SystemExit(
                        "Falló waypoint específico "
                        f"{transport_index}."
                    )

        print("6. Ir a pre-place")
        code, msg = node.send_goal(q_pre_place, duration_sec=8)
        print("result pre_place:", code, repr(msg))
        if code != 0:
            node.send_goal(q_home, duration_sec=8)
            raise SystemExit("Falló pre_place.")

        print("7. Bajar a place")
        code, msg = node.send_goal(q_place, duration_sec=6)
        print("result place:", code, repr(msg))
        if code != 0:
            node.send_goal(q_home, duration_sec=8)
            raise SystemExit("Falló place.")

        actual_place = node.current_tcp_xyz()
        actual_place_err = float(np.linalg.norm(actual_place - place_xyz))
        print("actual_place_xyz_m:", actual_place.tolist())
        print("target_place_xyz_m:", place_xyz.tolist())
        print("actual_place_error_m:", actual_place_err)

        print("8. Abrir pinza en destino")
        node.set_gripper_width(open_width, wait_sec=2.5)

        print("9. Retirada vertical")
        code, msg = node.send_goal(q_retreat, duration_sec=6)
        print("result retreat:", code, repr(msg))
        if code != 0:
            node.send_goal(q_home, duration_sec=8)
            raise SystemExit("Falló retreat.")

        print("10. Volver a home")
        code, msg = node.send_goal(q_home, duration_sec=8)
        print("result home:", code, repr(msg))

        result["execution_place_yellow"] = {
            "actual_pick_xyz_m": [float(x) for x in actual_pick],
            "actual_pick_error_m": actual_pick_err,
            "actual_lift_xyz_m": [float(x) for x in actual_lift],
            "actual_lift_error_m": actual_lift_err,
            "actual_place_xyz_m": [float(x) for x in actual_place],
            "actual_place_error_m": actual_place_err,
            "placed_on_yellow": True,
            "opened_at_place": True,
            "returned_home": True,
            "home_error_code": code,
            "home_message": msg,
        }

        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        node.destroy_node()
        rclpy.shutdown()

        print("Episodio pick-and-place completado.")
        return

    if os.environ.get("RETURN_OBJECT_ORIGINAL", "0") == "1":
        print("6. Volver a la posición de origen")
        code, msg = node.send_goal(q_pick, duration_sec=5)
        print("result return_to_pick:", code, repr(msg))
        if code != 0:
            print("Falló el retorno a pick; se intenta volver a home.")
            node.send_goal(q_home, duration_sec=8)
            raise SystemExit("Falló return_to_pick.")

        print("7. Abrir pinza y soltar en origen")
        node.set_gripper_width(open_width, wait_sec=2.5)

        print("8. Retirada vertical a pre-pick")
        code, msg = node.send_goal(q_pre, duration_sec=5)
        print("result retreat_pre_pick:", code, repr(msg))
        if code != 0:
            print("Falló la retirada; se intenta volver a home.")
            node.send_goal(q_home, duration_sec=8)
            raise SystemExit("Falló retreat_pre_pick.")

        print("9. Volver a home")
        code, msg = node.send_goal(q_home, duration_sec=8)
        print("result home:", code, repr(msg))

        result["execution"] = {
            "actual_pick_xyz_m": [float(x) for x in actual_pick],
            "actual_pick_error_m": actual_pick_err,
            "actual_lift_xyz_m": [float(x) for x in actual_lift],
            "actual_lift_error_m": actual_lift_err,
            "returned_object_original": True,
            "opened_after_return": True,
            "home_error_code": code,
            "home_message": msg,
        }

        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        node.destroy_node()
        rclpy.shutdown()

        print("Secuencia de pick, elevación y retorno completada.")
        return

    print("6. Volver a home")
    code, msg = node.send_goal(q_home, duration_sec=8)
    print("result home:", code, repr(msg))

    result["execution"] = {
        "actual_pick_xyz_m": [float(x) for x in actual_pick],
        "actual_pick_error_m": actual_pick_err,
        "actual_lift_xyz_m": [float(x) for x in actual_lift],
        "actual_lift_error_m": actual_lift_err,
        "home_error_code": code,
        "home_message": msg,
    }

    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    node.destroy_node()
    rclpy.shutdown()

    print("Secuencia de pick y elevación completada; revisar el resultado visual.")


if __name__ == "__main__":
    main()
