#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path.cwd()

# Matriz usada en las pruebas de calibración: p_base_mm = T_base_from_camera * p_camera_mm
T_BASE_FROM_CAMERA_MM = np.array([
    [ 0.041, -0.998, -0.056, -219.488],
    [-0.999, -0.041,  0.000, -952.756],
    [-0.002,  0.056, -0.998,  866.163],
    [ 0.000,  0.000,  0.000,    1.000],
], dtype=float)


def parse_ply_vertices_only(path: Path):
    data = path.read_bytes()
    marker = b"end_header\n"
    idx = data.find(marker)
    if idx < 0:
        raise RuntimeError("PLY sin end_header")

    header = data[:idx + len(marker)].decode("ascii", errors="replace")
    body = data[idx + len(marker):]

    vertex_count = None
    width = None
    height = None
    props = []
    current_element = None

    for line in header.splitlines():
        line = line.strip()

        mw = re.search(r"Width\s*=\s*(\d+)", line)
        mh = re.search(r"Height\s*=\s*(\d+)", line)
        if mw:
            width = int(mw.group(1))
        if mh:
            height = int(mh.group(1))

        if line.startswith("element "):
            parts = line.split()
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
            continue

        if line.startswith("property ") and current_element == "vertex":
            parts = line.split()
            props.append((parts[1], parts[2]))

    if vertex_count is None:
        raise RuntimeError("No se encontró element vertex")

    if width is None or height is None:
        if vertex_count == 1120 * 800:
            width, height = 1120, 800
        else:
            width, height = vertex_count, 1

    type_map = {
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
    }

    dtype_fields = []
    for typ, name in props:
        if typ not in type_map:
            raise RuntimeError(f"Tipo PLY no soportado en vertex: {typ} {name}")
        dtype_fields.append((name, type_map[typ]))

    arr = np.frombuffer(body, dtype=np.dtype(dtype_fields), count=vertex_count)

    if width * height != vertex_count:
        raise RuntimeError(
            f"PLY no coherente: width={width}, height={height}, vertices={vertex_count}"
        )

    return arr, width, height, header


def get_rgb(arr, w, h):
    names = arr.dtype.names

    if all(k in names for k in ["red", "green", "blue"]):
        rn, gn, bn = "red", "green", "blue"
    elif all(k in names for k in ["Texture32Red", "Texture32Green", "Texture32Blue"]):
        rn, gn, bn = "Texture32Red", "Texture32Green", "Texture32Blue"
    else:
        raise RuntimeError(f"No encuentro RGB en PLY. Propiedades: {names}")

    r = arr[rn].reshape(h, w).astype(np.float32)
    g = arr[gn].reshape(h, w).astype(np.float32)
    b = arr[bn].reshape(h, w).astype(np.float32)

    maxv = max(
        float(np.nanmax(r)),
        float(np.nanmax(g)),
        float(np.nanmax(b)),
        1.0,
    )

    if maxv > 255.0:
        scale = 255.0 / maxv
        r *= scale
        g *= scale
        b *= scale

    rgb = np.dstack([r, g, b]).clip(0, 255).astype(np.uint8)
    return rgb

def camera_to_base(camera_xyz_mm):
    p = np.array([camera_xyz_mm[0], camera_xyz_mm[1], camera_xyz_mm[2], 1.0], dtype=float)
    q = T_BASE_FROM_CAMERA_MM @ p
    return q[:3]


def largest_components(mask, min_area=100, max_area=999999):
    num, labels, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)

    comps = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        comps.append({
            "label": i,
            "bbox_xywh": [int(x), int(y), int(w), int(h)],
            "area_px": int(area),
            "center_px": [float(cent[i][0]), float(cent[i][1])],
            "mask": labels == i,
        })

    comps.sort(key=lambda c: c["area_px"], reverse=True)
    return comps


def xyz_stats_from_mask(arr, w, h, mask):
    x = arr["x"].reshape(h, w)
    y = arr["y"].reshape(h, w)
    z = arr["z"].reshape(h, w)

    valid = mask & np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0)

    if int(valid.sum()) < 20:
        raise RuntimeError("Muy pocos puntos 3D válidos en la máscara")

    pts = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(float)

    return {
        "num_points": int(pts.shape[0]),
        "median_xyz_mm": [float(v) for v in np.median(pts, axis=0)],
        "mean_xyz_mm": [float(v) for v in np.mean(pts, axis=0)],
        "min_xyz_mm": [float(v) for v in np.min(pts, axis=0)],
        "max_xyz_mm": [float(v) for v in np.max(pts, axis=0)],
        "p10_xyz_mm": [float(v) for v in np.percentile(pts, 10, axis=0)],
        "p90_xyz_mm": [float(v) for v in np.percentile(pts, 90, axis=0)],
    }


def select_orange_component(arr, w, h, orange_comps):
    """
    Selecciona el dado naranja a partir de forma, geometría y posición.
    
    - no se elige solo por ser la componente naranja más grande;
    - se rechazan franjas estrechas, regiones degeneradas y candidatos fuera de
      la zona física validada;
    - se combina forma 2D, coherencia 3D y posición en base.
    """
    accepted = []
    rejected = []

    # Zona física validada para el pick vertical real.
    # Es amplia, pero descarta detecciones fuera del rango esperado, como Y ~= -1.21.
    x_min, x_max = -0.75, 0.15
    y_min, y_max = -1.05, -0.45
    z_min, z_max = 0.025, 0.130

    best_component = None
    best_stats = None
    best_score = None
    best_report = None

    for idx, c in enumerate(orange_comps):
        x0, y0, bw, bh = c["bbox_xywh"]
        area = int(c["area_px"])

        aspect = float(bw) / max(1.0, float(bh))
        compact = float(area) / max(1.0, float(bw * bh))

        base_item = {
            "idx": int(idx),
            "bbox_xywh": [int(x0), int(y0), int(bw), int(bh)],
            "area_px": int(area),
            "center_px": c.get("center_px"),
            "aspect": float(aspect),
            "compact": float(compact),
        }

        reason = None

        if bw < 20 or bh < 20:
            reason = f"bbox_side_too_small:{bw}x{bh}"
        elif bw > 180 or bh > 180:
            reason = f"bbox_side_too_large:{bw}x{bh}"
        elif aspect < (1.0 / 3.0) or aspect > 3.0:
            pass  # Aspecto 2D solo informativo; la aceptación continúa con geometría 3D.
        elif compact < 0.10:
            reason = f"too_sparse:{compact:.3f}"

        if reason is not None:
            item = dict(base_item)
            item["reason"] = reason
            rejected.append(item)
            continue

        try:
            stats = xyz_stats_from_mask(arr, w, h, c["mask"])
        except Exception as e:
            item = dict(base_item)
            item["reason"] = f"xyz_stats_failed:{repr(e)}"
            rejected.append(item)
            continue

        cam = np.asarray(stats["median_xyz_mm"], dtype=float)
        base_mm = camera_to_base(cam)
        base_m = base_mm / 1000.0
        bx, by, bz = map(float, base_m[:3])

        p10 = np.asarray(stats["p10_xyz_mm"], dtype=float)
        p90 = np.asarray(stats["p90_xyz_mm"], dtype=float)
        span = p90 - p10
        max_span = float(np.max(span))
        min_span = float(np.min(span))

        item = dict(base_item)
        item.update({
            "num_points": int(stats["num_points"]),
            "camera_xyz_mm": [float(v) for v in cam[:3]],
            "base_xyz_m": [float(v) for v in base_m[:3]],
            "span_p10_p90_mm": [float(v) for v in span[:3]],
            "max_span_mm": float(max_span),
            "min_span_mm": float(min_span),
        })

        if not (x_min <= bx <= x_max):
            item["reason"] = f"base_x_out:{bx:.3f} not in [{x_min:.3f},{x_max:.3f}]"
            rejected.append(item)
            continue

        if not (y_min <= by <= y_max):
            item["reason"] = f"base_y_out:{by:.3f} not in [{y_min:.3f},{y_max:.3f}]"
            rejected.append(item)
            continue

        if not (z_min <= bz <= z_max):
            item["reason"] = f"base_z_out:{bz:.3f} not in [{z_min:.3f},{z_max:.3f}]"
            rejected.append(item)
            continue

        if max_span > 140.0:
            item["reason"] = f"span_too_large:{max_span:.1f}mm"
            rejected.append(item)
            continue

        # Criterio: forma cúbica o compacta y una nube 3D coherente.
        # El tamaño por sí solo no decide la selección.
        aspect_score = -abs(float(np.log(max(1e-6, aspect))))
        size_score = -abs(float(area) - 2500.0) / 2500.0
        compact_score = float(compact)
        span_score = -max(0.0, max_span - 70.0) / 70.0

        score = (
            1.20 * compact_score
            + 0.90 * aspect_score
            + 0.50 * size_score
            + (1.0 / 3.0) * span_score
        )

        item["score"] = float(score)
        item["reason"] = "accepted"
        accepted.append(item)

        if best_score is None or score > best_score:
            best_score = float(score)
            best_component = c
            best_stats = stats
            best_report = item

    report = {
        "selected": best_report,
        "accepted": accepted,
        "rejected": rejected,
        "filters": {
            "bbox_min_side_px": 20,
            "aspect_min": (1.0 / 3.0),
            "aspect_max": 3.0,
            "base_x_range_m": [x_min, x_max],
            "base_y_range_m": [y_min, y_max],
            "base_z_range_m": [z_min, z_max],
            "max_span_mm": 140.0,
        },
    }

    return best_component, best_stats, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default=str(ROOT / "data/photoneo/escena.ply"))
    ap.add_argument("--safe-above-object-m", type=float, default=0.17)
    ap.add_argument("--safe-above-target-m", type=float, default=0.20)
    args = ap.parse_args()

    ply = Path(args.ply).expanduser().resolve()
    if not ply.exists():
        raise SystemExit(f"No existe PLY: {ply}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "validacion_robot" / f"zonas_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cálculo geométrico de la escena")
    print("PLY:", ply)
    print("Salida:", out_dir)

    arr, w, h, header = parse_ply_vertices_only(ply)
    rgb = get_rgb(arr, w, h)

    (out_dir / "ply_header.txt").write_text(header, encoding="utf-8")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "photoneo_rgb.png"), bgr)

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    x3 = arr["x"].reshape(h, w)
    y3 = arr["y"].reshape(h, w)
    z3 = arr["z"].reshape(h, w)
    finite = np.isfinite(x3) & np.isfinite(y3) & np.isfinite(z3) & (z3 > 0)

    # También se convierte a HSV para tolerar mejor los cambios de iluminación.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # Dado naranja: pequeño, compacto, R alto, G medio, B bajo.
    # HSV ayuda a diferenciarlo del amarillo grande.
    orange_mask_rgb = finite & (r > 140) & (g > 40) & (g < 210) & (b < 170) & ((r - g) > 25)
    orange_mask_hsv = finite & (hue >= 3) & (hue <= 28) & (sat > 60) & (val > 80)
    orange_mask = orange_mask_rgb & orange_mask_hsv

    # Zona amarilla: superficie grande. La condición antigua era demasiado estricta con B.
    # Por eso se combina RGB más flexible con HSV. En OpenCV H está en [0,179].
    yellow_mask_rgb = finite & (r > 130) & (g > 110) & (b < 210) & (np.abs(r - g) < 120)
    yellow_mask_hsv = finite & (hue >= 18) & (hue <= 45) & (sat > 35) & (val > 120)
    yellow_mask = yellow_mask_rgb & yellow_mask_hsv

    # Como alternativa, la zona amarilla se busca a la derecha de la zona azul;
    # es brillante, grande y no debe confundirse con el dado naranja.
    xx_pix = np.tile(np.arange(w), (h, 1))
    right_half = xx_pix > int(w * 0.35)
    yellow_fallback = finite & right_half & (r > 130) & (g > 120) & (val > 130) & (sat > 20) & (b < 230)
    yellow_mask = yellow_mask | yellow_fallback

    # Zona azul: superficie origen, B dominante.
    blue_mask_rgb = finite & (b > 90) & (b > r * 1.10) & (b > g * 1.05)
    blue_mask_hsv = finite & (hue >= 85) & (hue <= 135) & (sat > 35) & (val > 40)
    blue_mask = blue_mask_rgb | blue_mask_hsv

    orange_comps = largest_components(orange_mask, min_area=200, max_area=15000)
    yellow_comps = largest_components(yellow_mask, min_area=1000, max_area=500000)
    blue_comps = largest_components(blue_mask, min_area=1000, max_area=500000)

    if not orange_comps:
        raise RuntimeError("No detecté ningún componente naranja")
    if not yellow_comps:
        raise RuntimeError("No detecté la zona amarilla")
    if not blue_comps:
        print("No detecté zona azul grande, pero puedo seguir con naranja -> amarillo")

    orange, orange_stats, orange_selection_report = select_orange_component(arr, w, h, orange_comps)

    (out_dir / "orange_component_selection.json").write_text(
        json.dumps(orange_selection_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if orange is None:
        raise RuntimeError(
            "No detecté un dado naranja fiable dentro de la zona de trabajo validada. "
            f"Revisar {out_dir / 'orange_component_selection.json'}"
        )

    print(
        "selected "
        f"bbox={orange_selection_report['selected']['bbox_xywh']} "
        f"base={orange_selection_report['selected']['base_xyz_m']} "
        f"score={orange_selection_report['selected']['score']:.3f}",
        flush=True,
    )

    yellow = yellow_comps[0]
    blue = blue_comps[0] if blue_comps else None

    yellow_stats = xyz_stats_from_mask(arr, w, h, yellow["mask"])
    blue_stats = xyz_stats_from_mask(arr, w, h, blue["mask"]) if blue else None

    orange_camera_mm = orange_stats["median_xyz_mm"]
    yellow_camera_mm = yellow_stats["median_xyz_mm"]
    blue_camera_mm = blue_stats["median_xyz_mm"] if blue_stats else None

    orange_base_mm = camera_to_base(orange_camera_mm)
    yellow_base_mm = camera_to_base(yellow_camera_mm)
    blue_base_mm = camera_to_base(blue_camera_mm) if blue_camera_mm else None

    orange_base_m = orange_base_mm / 1000.0
    yellow_base_m = yellow_base_mm / 1000.0
    blue_base_m = blue_base_mm / 1000.0 if blue_base_mm is not None else None

    # Punto alto seguro sobre el dado.
    safe_above_orange_m = [
        float(orange_base_m[0]),
        float(orange_base_m[1]),
        float(max(0.18, min(0.45, orange_base_m[2] + args.safe_above_object_m))),
    ]

    # Punto alto seguro sobre el centro de la zona amarilla.
    safe_above_yellow_m = [
        float(yellow_base_m[0]),
        float(yellow_base_m[1]),
        float(max(0.18, min(0.45, yellow_base_m[2] + args.safe_above_target_m))),
    ]

    # Vista de comprobación.
    dbg = bgr.copy()

    ox, oy, ow, oh = orange["bbox_xywh"]
    yx, yy, yw, yh = yellow["bbox_xywh"]
    cv2.rectangle(dbg, (ox, oy), (ox + ow, oy + oh), (0, 140, 255), 3)
    cv2.putText(dbg, "orange_die", (ox, max(0, oy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

    cv2.rectangle(dbg, (yx, yy), (yx + yw, yy + yh), (0, 255, 255), 3)
    cv2.putText(dbg, "yellow_target", (yx, max(0, yy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if blue:
        bx, by, bw, bh = blue["bbox_xywh"]
        cv2.rectangle(dbg, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
        cv2.putText(dbg, "blue_source", (bx, max(0, by - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imwrite(str(out_dir / "diagnostico_zonas.png"), dbg)

    result = {
        "task": "move orange die from blue surface to yellow surface",
        "ply": str(ply),
        "image_size": [w, h],
        "orange_die": {
            "bbox_xywh": orange["bbox_xywh"],
            "area_px": orange["area_px"],
            "center_px": orange["center_px"],
            "camera_xyz_mm": orange_camera_mm,
            "base_xyz_mm": orange_base_mm.tolist(),
            "base_xyz_m": orange_base_m.tolist(),
            "safe_above_base_xyz_m": safe_above_orange_m,
            "stats": orange_stats,
        },
        "yellow_target": {
            "bbox_xywh": yellow["bbox_xywh"],
            "area_px": yellow["area_px"],
            "center_px": yellow["center_px"],
            "camera_xyz_mm": yellow_camera_mm,
            "base_xyz_mm": yellow_base_mm.tolist(),
            "base_xyz_m": yellow_base_m.tolist(),
            "safe_above_base_xyz_m": safe_above_yellow_m,
            "stats": yellow_stats,
        },
        "blue_source": None if blue is None else {
            "bbox_xywh": blue["bbox_xywh"],
            "area_px": blue["area_px"],
            "center_px": blue["center_px"],
            "camera_xyz_mm": blue_camera_mm,
            "base_xyz_mm": blue_base_mm.tolist(),
            "base_xyz_m": blue_base_m.tolist(),
            "stats": blue_stats,
        },
        "planned_safe_probe_sequence": [
            {
                "name": "above_orange_die",
                "base_xyz_m": safe_above_orange_m,
                "meaning": "punto alto seguro sobre el dado naranja",
            },
            {
                "name": "above_yellow_target",
                "base_xyz_m": safe_above_yellow_m,
                "meaning": "punto alto seguro sobre la zona amarilla",
            },
        ],
        "diag_image": str(out_dir / "diagnostico_zonas.png"),
        "note": "Solo cálculo real. No se mueve el robot.",
    }

    (out_dir / "resultado_zonas.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Resultado de la escena")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    print("No se ha movido el robot.")
    print("Diagnóstico:", out_dir / "diagnostico_zonas.png")
    print("JSON:", out_dir / "resultado_zonas.json")


if __name__ == "__main__":
    main()
