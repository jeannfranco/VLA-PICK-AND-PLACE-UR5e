#!/usr/bin/env python3

import atexit as _shape_atexit
import time as _shape_time
from pathlib import Path as _ShapePath

_shape_started_ns = _shape_time.time_ns()

def _shape_emit_camera_scene_json():
    roots = [
        _ShapePath(
            "results/validacion_robot"
        ),
        _ShapePath(
            "results/"
            "validacion_robot"
        ),
    ]

    candidates = []

    for root in roots:
        if not root.is_dir():
            continue

        for path in root.glob(
            "forma_*/"
            "resultado_forma.json"
        ):
            try:
                if path.stat().st_mtime_ns >= _shape_started_ns:
                    candidates.append(path)
            except OSError:
                pass

    if not candidates:
        return

    latest = max(
        candidates,
        key=lambda path: path.stat().st_mtime_ns,
    )

    print(
        f"{latest}",
        flush=True,
    )

_shape_atexit.register(
    _shape_emit_camera_scene_json
)


from pathlib import Path
from datetime import datetime
import importlib.util
import json
import math

import cv2
import numpy as np


ROOT = (
    Path("/workspace")
    if Path("/workspace").is_dir()
    else Path(__file__).resolve().parents[1]
)

PLY_PATH = (
    ROOT
    / "photoneo_inbox"
    / "live_scene.ply"
)

PLY_READER_PATH = (
    ROOT
    / "scripts"
    / "vision_zonas.py"
)

RESULTS_ROOT = (
    ROOT
    / "results"
    / "validacion_robot"
)


T_BASE_FROM_CAMERA_MM = np.array(
    [
        [
            0.041,
            -0.998,
            -0.056,
            -219.488,
        ],
        [
            -0.999,
            -0.041,
            0.000,
            -952.756,
        ],
        [
            -0.002,
            0.056,
            -0.998,
            866.163,
        ],
        [
            0.000,
            0.000,
            0.000,
            1.000,
        ],
    ],
    dtype=float,
)


def load_ply():
    if not PLY_PATH.is_file():
        raise RuntimeError(
            f"No existe el PLY actual: {PLY_PATH}"
        )

    if not PLY_READER_PATH.is_file():
        raise RuntimeError(
            f"No existe el lector PLY: "
            f"{PLY_READER_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "VLA_ply_reader",
        PLY_READER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "No se pudo cargar el lector PLY."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    if not hasattr(
        module,
        "parse_ply_vertices_only",
    ):
        raise RuntimeError(
            "El lector no contiene "
            "parse_ply_vertices_only()."
        )

    return module.parse_ply_vertices_only(
        PLY_PATH
    )


def to_uint8(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    finite = np.isfinite(
        values
    )

    if not finite.any():
        return np.zeros(
            values.shape,
            dtype=np.uint8,
        )

    low = float(
        np.nanpercentile(
            values[finite],
            0.1,
        )
    )

    high = float(
        np.nanpercentile(
            values[finite],
            99.9,
        )
    )

    if high <= 1.5:
        values = (
            values
            * 255.0
        )

    elif high > 255.0:
        if high <= 65535.0:
            values = (
                values
                / 257.0
            )
        else:
            values = (
                255.0
                * (
                    values
                    - low
                )
                / max(
                    high - low,
                    1e-9,
                )
            )

    return np.clip(
        np.rint(values),
        0,
        255,
    ).astype(
        np.uint8
    )


def rgb_from_ply(
    vertices,
    width,
    height,
):
    names = set(
        vertices.dtype.names
        or ()
    )

    if {
        "red",
        "green",
        "blue",
    }.issubset(names):
        keys = (
            "red",
            "green",
            "blue",
        )

    elif {
        "Texture32Red",
        "Texture32Green",
        "Texture32Blue",
    }.issubset(names):
        keys = (
            "Texture32Red",
            "Texture32Green",
            "Texture32Blue",
        )

    else:
        raise RuntimeError(
            "El PLY no contiene RGB compatible. "
            f"Campos: {sorted(names)}"
        )

    rgb = np.stack(
        [
            to_uint8(
                vertices[key]
            )
            for key in keys
        ],
        axis=1,
    )

    return rgb.reshape(
        height,
        width,
        3,
    )


def base_xyz_from_ply(
    vertices,
    width,
    height,
):
    names = set(
        vertices.dtype.names
        or ()
    )

    if not {
        "x",
        "y",
        "z",
    }.issubset(names):
        raise RuntimeError(
            "El PLY no contiene x, y, z."
        )

    camera_xyz_mm = np.stack(
        [
            np.asarray(
                vertices["x"],
                dtype=float,
            ),
            np.asarray(
                vertices["y"],
                dtype=float,
            ),
            np.asarray(
                vertices["z"],
                dtype=float,
            ),
        ],
        axis=1,
    )

    finite = (
        np.all(
            np.isfinite(
                camera_xyz_mm
            ),
            axis=1,
        )
        & (
            camera_xyz_mm[:, 2]
            > 0.0
        )
    )

    homogeneous = np.ones(
        (
            camera_xyz_mm.shape[0],
            4,
        ),
        dtype=float,
    )

    homogeneous[:, :3] = (
        camera_xyz_mm
    )

    base_xyz_m = (
        (
            T_BASE_FROM_CAMERA_MM
            @ homogeneous.T
        ).T[:, :3]
        / 1000.0
    )

    base_xyz_m[
        ~finite
    ] = np.nan

    return (
        base_xyz_m.reshape(
            height,
            width,
            3,
        ),
        finite.reshape(
            height,
            width,
        ),
    )


def clean_mask(
    mask,
    kernel_size=5,
):
    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size,
            ),
        )
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return mask


def biggest_contour(
    mask,
    minimum_area,
):
    contours, _ = (
        cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
    )

    contours = [
        contour
        for contour in contours
        if cv2.contourArea(
            contour
        ) >= minimum_area
    ]

    if not contours:
        raise RuntimeError(
            "No se encontró contorno con "
            f"área mínima {minimum_area:.1f}."
        )

    return max(
        contours,
        key=cv2.contourArea,
    )


def contour_record(
    contour,
):
    area = float(
        cv2.contourArea(
            contour
        )
    )

    x, y, width, height = (
        cv2.boundingRect(
            contour
        )
    )

    perimeter = float(
        cv2.arcLength(
            contour,
            True,
        )
    )

    circularity = (
        4.0
        * math.pi
        * area
        / max(
            perimeter
            * perimeter,
            1e-9,
        )
    )

    approximation_02 = (
        cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )
    )

    approximation_04 = (
        cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True,
        )
    )

    moments = cv2.moments(
        contour
    )

    if moments["m00"]:
        center_x = (
            moments["m10"]
            / moments["m00"]
        )

        center_y = (
            moments["m01"]
            / moments["m00"]
        )

    else:
        center_x = (
            x
            + width / 2.0
        )

        center_y = (
            y
            + height / 2.0
        )

    minimum_rectangle = (
        cv2.minAreaRect(
            contour
        )
    )

    rectangle_width = float(
        minimum_rectangle[1][0]
    )

    rectangle_height = float(
        minimum_rectangle[1][1]
    )

    minimum_rectangle_area = (
        rectangle_width
        * rectangle_height
    )

    rectangularity = (
        area
        / max(
            minimum_rectangle_area,
            1e-9,
        )
    )

    (
        _,
        enclosing_radius,
    ) = cv2.minEnclosingCircle(
        contour
    )

    enclosing_circle_area = (
        math.pi
        * float(enclosing_radius) ** 2
    )

    circle_fill_ratio = (
        area
        / max(
            enclosing_circle_area,
            1e-9,
        )
    )

    hull = cv2.convexHull(
        contour
    )

    hull_area = float(
        cv2.contourArea(
            hull
        )
    )

    solidity = (
        area
        / max(
            hull_area,
            1e-9,
        )
    )

    raw_points = (
        contour
        .reshape(-1, 2)
        .astype(float)
    )

    sampled_points = []

    if len(raw_points) >= 2:
        for index in range(
            len(raw_points)
        ):
            point_a = raw_points[
                index
            ]

            point_b = raw_points[
                (
                    index + 1
                )
                % len(raw_points)
            ]

            segment_length = float(
                np.linalg.norm(
                    point_b
                    - point_a
                )
            )

            sample_count = max(
                2,
                int(
                    math.ceil(
                        segment_length
                    )
                ),
            )

            for fraction in np.linspace(
                0.0,
                1.0,
                sample_count,
                endpoint=False,
            ):
                sampled_points.append(
                    point_a
                    + fraction
                    * (
                        point_b
                        - point_a
                    )
                )

    if sampled_points:
        sampled_points = np.asarray(
            sampled_points,
            dtype=float,
        )

        radial_distances = np.linalg.norm(
            sampled_points
            - np.array(
                [
                    center_x,
                    center_y,
                ],
                dtype=float,
            ),
            axis=1,
        )

        radial_mean = float(
            np.mean(
                radial_distances
            )
        )

        radial_cv = float(
            np.std(
                radial_distances
            )
            / max(
                radial_mean,
                1e-9,
            )
        )

    else:
        radial_cv = 1.0

    return {
        "contour":
            contour,
        "bbox_xyxy": [
            int(x),
            int(y),
            int(
                x
                + width
                - 1
            ),
            int(
                y
                + height
                - 1
            ),
        ],
        "center_uv": [
            float(center_x),
            float(center_y),
        ],
        "area_px":
            area,
        "circularity":
            float(circularity),
        "fill_ratio":
            float(
                area
                / max(
                    width
                    * height,
                    1,
                )
            ),
        "rectangularity":
            float(rectangularity),
        "circle_fill_ratio":
            float(circle_fill_ratio),
        "radial_cv":
            float(radial_cv),
        "solidity":
            float(solidity),
        "vertices":
            int(
                len(
                    approximation_04
                )
            ),
        "vertices_epsilon_02":
            int(
                len(
                    approximation_02
                )
            ),
        "vertices_epsilon_04":
            int(
                len(
                    approximation_04
                )
            ),
    }


def contour_to_mask(
    image_shape,
    contour,
):
    mask = np.zeros(
        image_shape,
        dtype=np.uint8,
    )

    cv2.drawContours(
        mask,
        [
            contour,
        ],
        -1,
        255,
        thickness=-1,
    )

    return mask


def expand_bbox(
    bbox,
    image_shape,
    scale=2.25,
):
    x1, y1, x2, y2 = bbox

    image_height, image_width = (
        image_shape
    )

    center_x = (
        x1
        + x2
    ) / 2.0

    center_y = (
        y1
        + y2
    ) / 2.0

    width = (
        x2
        - x1
        + 1
    )

    height = (
        y2
        - y1
        + 1
    )

    return [
        max(
            0,
            int(
                center_x
                - scale
                * width
                / 2.0
            ),
        ),
        max(
            0,
            int(
                center_y
                - scale
                * height
                / 2.0
            ),
        ),
        min(
            image_width - 1,
            int(
                center_x
                + scale
                * width
                / 2.0
            ),
        ),
        min(
            image_height - 1,
            int(
                center_y
                + scale
                * height
                / 2.0
            ),
        ),
    ]


def robust_xyz(
    base_xyz,
    finite,
    mask,
    z_percentile,
    label,
):
    valid = (
        (
            mask
            > 0
        )
        & finite
        & np.all(
            np.isfinite(
                base_xyz
            ),
            axis=2,
        )
    )

    points = base_xyz[
        valid
    ]

    if len(points) < 30:
        raise RuntimeError(
            f"{label}: solo "
            f"{len(points)} "
            "puntos 3D válidos."
        )

    xy = np.median(
        points[:, :2],
        axis=0,
    )

    z = np.percentile(
        points[:, 2],
        z_percentile,
    )

    return (
        [
            float(xy[0]),
            float(xy[1]),
            float(z),
        ],
        int(
            len(points)
        ),
    )


def main():
    (
        vertices,
        width,
        height,
        _,
    ) = load_ply()

    rgb = rgb_from_ply(
        vertices,
        width,
        height,
    )

    (
        base_xyz,
        finite,
    ) = base_xyz_from_ply(
        vertices,
        width,
        height,
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    hsv = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2HSV,
    )

    image_area = (
        width
        * height
    )

    yellow_mask = clean_mask(
        cv2.inRange(
            hsv,
            np.array(
                [
                    18,
                    70,
                    60,
                ]
            ),
            np.array(
                [
                    45,
                    255,
                    255,
                ]
            ),
        )
    )

    board_contour = biggest_contour(
        yellow_mask,
        image_area
        * 0.01,
    )

    board_hull = cv2.convexHull(
        board_contour
    )

    board_mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    cv2.fillConvexPoly(
        board_mask,
        board_hull,
        255,
    )

    board_area = float(
        cv2.contourArea(
            board_hull
        )
    )

    orange_mask = clean_mask(
        cv2.bitwise_and(
            cv2.inRange(
                hsv,
                np.array(
                    [
                        0,
                        90,
                        55,
                    ]
                ),
                np.array(
                    [
                        18,
                        255,
                        255,
                    ]
                ),
            ),
            board_mask,
        ),
        3,
    )

    white_mask = clean_mask(
        cv2.bitwise_and(
            cv2.inRange(
                hsv,
                np.array(
                    [
                        0,
                        0,
                        145,
                    ]
                ),
                np.array(
                    [
                        179,
                        75,
                        255,
                    ]
                ),
            ),
            board_mask,
        ),
        3,
    )

    orange_record = contour_record(
        biggest_contour(
            orange_mask,
            board_area
            * 0.0015,
        )
    )

    white_record = contour_record(
        biggest_contour(
            white_mask,
            board_area
            * 0.0015,
        )
    )

    (
        component_count,
        component_labels,
        component_stats,
        _,
    ) = (
        cv2.connectedComponentsWithStats(
            yellow_mask,
            connectivity=8,
        )
    )

    openings = []

    for component_id in range(
        1,
        component_count,
    ):
        area = int(
            component_stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < max(
            300,
            int(
                board_area
                * 0.006
            ),
        ):
            continue

        if area > int(
            board_area
            * 0.12
        ):
            continue

        component_mask = np.where(
            component_labels
            == component_id,
            255,
            0,
        ).astype(
            np.uint8
        )

        contours, _ = (
            cv2.findContours(
                component_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        )

        if not contours:
            continue

        candidate = contour_record(
            max(
                contours,
                key=cv2.contourArea,
            )
        )

        center_u = int(
            round(
                candidate[
                    "center_uv"
                ][0]
            )
        )

        center_v = int(
            round(
                candidate[
                    "center_uv"
                ][1]
            )
        )

        if not (
            0
            <= center_u
            < width
            and
            0
            <= center_v
            < height
        ):
            continue

        if not board_mask[
            center_v,
            center_u,
        ]:
            continue

        candidate[
            "mask"
        ] = component_mask

        openings.append(
            candidate
        )

    def circle_likeness(
        item,
    ):
        score = (
            1.50
            * item[
                "circularity"
            ]
            + 1.50
            * item[
                "circle_fill_ratio"
            ]
            - 1.20
            * item[
                "rectangularity"
            ]
            - 1.00
            * item[
                "radial_cv"
            ]
            + 0.15
            * min(
                item[
                    "vertices_epsilon_02"
                ],
                12,
            )
            / 12.0
        )

        return float(
            score
        )

    # Segmentación adaptativa respecto a la iluminación de la
    # captura Photoneo actual. No utiliza posiciones históricas.
    image_h, image_w = board_mask.shape[:2]
    value_channel = hsv[:, :, 2]

    board_selector = board_mask > 0
    board_values = value_channel[board_selector]

    if board_values.size < 100:
        raise RuntimeError(
            "Board mask insuficiente para detectar cajas"
        )

    otsu_threshold, _ = cv2.threshold(
        board_values.astype(np.uint8).reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    dark_threshold = int(
        np.clip(
            float(otsu_threshold) + 18.0,
            80.0,
            205.0,
        )
    )

    dark_mask = np.where(
        value_channel <= dark_threshold,
        255,
        0,
    ).astype(np.uint8)

    dark_mask = cv2.bitwise_and(
        dark_mask,
        board_mask,
    )

    # Eliminar los dos objetos manipulables de la máscara.
    object_mask = cv2.bitwise_or(
        orange_mask,
        white_mask,
    )

    object_mask = cv2.dilate(
        object_mask,
        np.ones((7, 7), dtype=np.uint8),
        iterations=1,
    )

    dark_mask = cv2.bitwise_and(
        dark_mask,
        cv2.bitwise_not(object_mask),
    )

    # Unir paredes fragmentadas de las cajas.
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        np.ones((13, 13), dtype=np.uint8),
        iterations=2,
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )

    (
        dark_component_count,
        dark_labels,
        dark_stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        dark_mask,
        connectivity=8,
    )

    board_pixels = max(
        1,
        cv2.countNonZero(board_mask),
    )

    candidates = []
    examined = []

    for label_id in range(
        1,
        dark_component_count,
    ):
        x = int(
            dark_stats[
                label_id,
                cv2.CC_STAT_LEFT,
            ]
        )
        y = int(
            dark_stats[
                label_id,
                cv2.CC_STAT_TOP,
            ]
        )
        width = int(
            dark_stats[
                label_id,
                cv2.CC_STAT_WIDTH,
            ]
        )
        height = int(
            dark_stats[
                label_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )
        pixels = int(
            dark_stats[
                label_id,
                cv2.CC_STAT_AREA,
            ]
        )

        examined.append(
            [x, y, width, height, pixels]
        )

        if pixels < max(
            350,
            int(board_pixels * 0.003),
        ):
            continue

        # Permitir cajas parcialmente fusionadas con su borde/sombra
        # tras mover la lámina, manteniendo los guards geométricos.
        if pixels > int(
            board_pixels * 0.35
        ):
            continue

        if width < 45 or height < 45:
            continue

        if (
            width > int(image_w * 0.42)
            or height > int(image_h * 0.45)
        ):
            continue

        if (
            x <= 3
            or y <= 3
            or x + width >= image_w - 3
            or y + height >= image_h - 3
        ):
            continue

        aspect_ratio = (
            float(width)
            / float(max(height, 1))
        )

        if not 0.40 <= aspect_ratio <= 2.40:
            continue

        component_mask = np.where(
            dark_labels == label_id,
            255,
            0,
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        hull = cv2.convexHull(contour)
        hull_area = float(
            cv2.contourArea(hull)
        )

        if hull_area <= 0.0:
            continue

        bbox_area = float(
            width * height
        )
        fill_ratio = (
            float(pixels)
            / max(bbox_area, 1.0)
        )

        if fill_ratio < 0.08:
            continue

        mean_value = float(
            cv2.mean(
                value_channel,
                mask=component_mask,
            )[0]
        )

        darkness = (
            255.0 - mean_value
        ) / 255.0

        area_fraction = (
            float(pixels)
            / float(board_pixels)
        )

        score = (
            2.0 * darkness
            + 2.5 * np.sqrt(
                max(area_fraction, 0.0)
            )
            + 0.25 * min(
                fill_ratio,
                1.0,
            )
        )

        rotated_rect = cv2.minAreaRect(
            hull
        )
        center_u = float(
            rotated_rect[0][0]
        )
        center_v = float(
            rotated_rect[0][1]
        )

        candidates.append(
            {
                "label": int(label_id),
                "mask": component_mask,
                "contour": contour,
                "hull": hull,
                "bbox": [
                    x,
                    y,
                    width,
                    height,
                ],
                "center": [
                    center_u,
                    center_v,
                ],
                "pixels": pixels,
                "mean_value": mean_value,
                "darkness": darkness,
                "score": float(score),
            }
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    selected = []

    for candidate in candidates:
        center_u, center_v = (
            candidate["center"]
        )

        duplicate = False

        for previous in selected:
            previous_u, previous_v = (
                previous["center"]
            )

            distance_px = float(
                np.hypot(
                    center_u - previous_u,
                    center_v - previous_v,
                )
            )

            if distance_px < 55.0:
                duplicate = True
                break

        if duplicate:
            continue

        selected.append(candidate)

        if len(selected) == 2:
            break

    print(
        f"otsu={float(otsu_threshold):.2f} "
        f"threshold={dark_threshold} "
        f"examined={examined} "
        f"accepted={[item['bbox'] for item in candidates]}",
        flush=True,
    )

    if len(selected) != 2:
        raise RuntimeError(
            "Detección adaptativa de cajas incompleta: "
            f"candidates={len(candidates)} "
            f"selected={len(selected)}. "
            "El robot no debe moverse."
        )

    live_openings = []
    selected_diag = []

    for candidate in selected:
        x, y, width, height = [
            int(value)
            for value in candidate["bbox"]
        ]

        hull = candidate["hull"]
        center_u, center_v = (
            candidate["center"]
        )

        # Detectar el hueco interior real de la caja:
        # región no negra encerrada dentro de su contorno exterior.
        # El bbox no se reduce de forma artificial y no se usan
        # posiciones históricas.

        local_black_mask = candidate["mask"][
            y:y + height,
            x:x + width,
        ].copy()

        if local_black_mask.shape != (
            height,
            width,
        ):
            raise RuntimeError(
                "Máscara negra local incompatible: "
                f"bbox={[x, y, width, height]} "
                f"shape={local_black_mask.shape}"
            )

        outer_hull_local = candidate[
            "hull"
        ].copy().astype(np.int32)

        outer_hull_local[:, :, 0] -= int(x)
        outer_hull_local[:, :, 1] -= int(y)

        outer_hull_mask = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )

        cv2.drawContours(
            outer_hull_mask,
            [outer_hull_local],
            -1,
            255,
            thickness=-1,
        )

        erosion_radius = max(
            4,
            min(
                10,
                int(
                    round(
                        min(width, height)
                        * 0.06
                    )
                ),
            ),
        )

        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * erosion_radius + 1,
                2 * erosion_radius + 1,
            ),
        )

        interior_support = cv2.erode(
            outer_hull_mask,
            erosion_kernel,
            iterations=1,
        )

        # La abertura real es la región amarilla visible,
        # encerrada dentro del contorno exterior negro.
        local_yellow_mask = yellow_mask[
            y:y + height,
            x:x + width,
        ].copy()

        if local_yellow_mask.shape != (
            height,
            width,
        ):
            raise RuntimeError(
                "Máscara amarilla local incompatible: "
                f"bbox={[x, y, width, height]} "
                f"shape={local_yellow_mask.shape}"
            )

        nonblack_inside = cv2.bitwise_and(
            interior_support,
            local_yellow_mask,
        )

        nonblack_inside = cv2.morphologyEx(
            nonblack_inside,
            cv2.MORPH_CLOSE,
            np.ones(
                (5, 5),
                dtype=np.uint8,
            ),
            iterations=2,
        )

        nonblack_inside = cv2.morphologyEx(
            nonblack_inside,
            cv2.MORPH_OPEN,
            np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
            iterations=1,
        )

        (
            opening_component_count,
            opening_labels,
            opening_stats,
            opening_centroids,
        ) = cv2.connectedComponentsWithStats(
            nonblack_inside,
            connectivity=8,
        )

        opening_candidates = []
        roi_area = float(
            width * height
        )
        roi_center = np.array(
            [
                width / 2.0,
                height / 2.0,
            ],
            dtype=np.float64,
        )
        roi_radius = max(
            1.0,
            float(
                np.hypot(
                    width / 2.0,
                    height / 2.0,
                )
            ),
        )

        for label_id in range(
            1,
            opening_component_count,
        ):
            component_x = int(
                opening_stats[
                    label_id,
                    cv2.CC_STAT_LEFT,
                ]
            )
            component_y = int(
                opening_stats[
                    label_id,
                    cv2.CC_STAT_TOP,
                ]
            )
            component_width = int(
                opening_stats[
                    label_id,
                    cv2.CC_STAT_WIDTH,
                ]
            )
            component_height = int(
                opening_stats[
                    label_id,
                    cv2.CC_STAT_HEIGHT,
                ]
            )
            component_area = int(
                opening_stats[
                    label_id,
                    cv2.CC_STAT_AREA,
                ]
            )

            area_fraction = (
                float(component_area)
                / max(roi_area, 1.0)
            )

            if not 0.015 <= area_fraction <= 0.55:
                continue

            if (
                component_width < 15
                or component_height < 15
            ):
                continue

            aspect_ratio = (
                float(component_width)
                / float(
                    max(
                        component_height,
                        1,
                    )
                )
            )

            if not 0.35 <= aspect_ratio <= 2.80:
                continue

            centroid = np.array(
                opening_centroids[
                    label_id
                ],
                dtype=np.float64,
            )

            distance_normalized = float(
                np.linalg.norm(
                    centroid
                    - roi_center
                )
                / roi_radius
            )

            fill_ratio = (
                float(component_area)
                / float(
                    max(
                        component_width
                        * component_height,
                        1,
                    )
                )
            )

            score = (
                2.0 * area_fraction
                + 0.25 * fill_ratio
                - 0.60 * distance_normalized
            )

            opening_candidates.append(
                {
                    "label": int(label_id),
                    "score": float(score),
                    "area": int(component_area),
                    "centroid": centroid,
                    "bbox": [
                        component_x,
                        component_y,
                        component_width,
                        component_height,
                    ],
                }
            )

        opening_candidates.sort(
            key=lambda item: item[
                "score"
            ],
            reverse=True,
        )

        if not opening_candidates:
            raise RuntimeError(
                "No se encontró el hueco interior real "
                f"de la caja bbox={[x, y, width, height]}"
            )

        selected_opening = opening_candidates[0]
        selected_label = selected_opening[
            "label"
        ]

        local_opening_component = np.where(
            opening_labels
            == selected_label,
            255,
            0,
        ).astype(np.uint8)

        local_opening_contours, _ = cv2.findContours(
            local_opening_component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not local_opening_contours:
            raise RuntimeError(
                "El hueco interior no produjo contorno"
            )

        inner_contour_local = max(
            local_opening_contours,
            key=cv2.contourArea,
        )

        # Se elige el punto con mayor distancia respecto a las
        # paredes negras. Para evitar escoger un extremo de una
        # meseta rectangular, se usa el centroide de todos los
        # píxeles que están prácticamente en el máximo.
        distance_map = cv2.distanceTransform(
            local_opening_component,
            cv2.DIST_L2,
            5,
        )

        max_clearance_px = float(
            distance_map.max()
        )

        if max_clearance_px < 3.0:
            raise RuntimeError(
                "Abertura sin holgura suficiente: "
                f"max_clearance_px={max_clearance_px:.3f}"
            )

        peak_mask = np.where(
            distance_map
            >= 0.95 * max_clearance_px,
            255,
            0,
        ).astype(np.uint8)

        peak_moments = cv2.moments(
            peak_mask,
            binaryImage=True,
        )

        if abs(
            float(peak_moments["m00"])
        ) < 1.0e-9:
            raise RuntimeError(
                "No se pudo calcular el centro "
                "de máxima holgura"
            )

        local_center_u = float(
            peak_moments["m10"]
            / peak_moments["m00"]
        )
        local_center_v = float(
            peak_moments["m01"]
            / peak_moments["m00"]
        )

        center_u = float(
            x + local_center_u
        )
        center_v = float(
            y + local_center_v
        )

        # Región central segura usada solo para obtener
        # el XYZ de deposición. La abertura completa se conserva
        # para clasificación, overlay y cálculo del borde.
        core_threshold = max(
            2.0,
            0.55 * max_clearance_px,
        )

        placement_core_local = np.where(
            distance_map >= core_threshold,
            255,
            0,
        ).astype(np.uint8)

        placement_core_points = int(
            cv2.countNonZero(
                placement_core_local
            )
        )

        if placement_core_points < 30:
            placement_core_local = np.where(
                distance_map
                >= 0.40 * max_clearance_px,
                255,
                0,
            ).astype(np.uint8)

            placement_core_points = int(
                cv2.countNonZero(
                    placement_core_local
                )
            )

        if placement_core_points < 30:
            raise RuntimeError(
                "Región central segura insuficiente: "
                f"points={placement_core_points}"
            )

        print(
            f"outer_bbox={[x, y, width, height]} "
            f"center_uv="
            f"{[round(center_u, 2), round(center_v, 2)]} "
            f"max_clearance_px={max_clearance_px:.3f} "
            f"core_points={placement_core_points}",
            flush=True,
        )

        inner_contour = (
            inner_contour_local.copy()
        )
        inner_contour[:, :, 0] += int(x)
        inner_contour[:, :, 1] += int(y)

        opening_mask = np.zeros_like(
            board_mask,
            dtype=np.uint8,
        )

        cv2.drawContours(
            opening_mask,
            [inner_contour],
            -1,
            255,
            thickness=-1,
        )

        local_ix, local_iy, iw, ih = (
            cv2.boundingRect(
                inner_contour_local
            )
        )

        ix = int(x + local_ix)
        iy = int(y + local_iy)

        print(
            f"outer_bbox={[x, y, width, height]} "
            f"opening_bbox={[ix, iy, ix + iw, iy + ih]} "
            f"center_uv={[round(center_u, 2), round(center_v, 2)]} "
            f"area={selected_opening['area']} "
            f"score={selected_opening['score']:.6f}",
            flush=True,
        )

        # La clasificación y el XYZ usan el contorno
        # real de la abertura, no el contorno exterior negro.
        record = contour_record(
            inner_contour
        )

        # Centro canónico de deposición y visualización:
        # máximo de holgura dentro de la abertura amarilla real.
        record[
            "detected_center_uv"
        ] = [
            float(center_u),
            float(center_v),
        ]
        record[
            "placement_center_uv"
        ] = [
            float(center_u),
            float(center_v),
        ]
        record[
            "opening_center_method"
        ] = "yellow_region_maximum_clearance"

        # robust_xyz() recorta el XYZ mediante el bbox exterior,
        # por lo que la máscara debe tener exactamente la misma
        # altura y anchura que dicho recorte.
        local_opening_mask = (
            placement_core_local.copy()
        )

        expected_shape = (
            int(height),
            int(width),
        )

        if local_opening_mask.shape != expected_shape:
            raise RuntimeError(
                "Máscara local incompatible con bbox: "
                f"bbox={[x, y, width, height]} "
                f"expected={expected_shape} "
                f"actual={local_opening_mask.shape}"
            )

        record["mask"] = local_opening_mask
        record["mask_full_image"] = opening_mask
        record[
            "mask_shape_hw"
        ] = [
            int(local_opening_mask.shape[0]),
            int(local_opening_mask.shape[1]),
        ]
        record[
            "adaptive_black_container_live"
        ] = True
        record[
            "placement_center_method"
        ] = "maximum_clearance_distance_transform"
        record[
            "placement_center_uv"
        ] = [
            float(center_u),
            float(center_v),
        ]
        record[
            "maximum_clearance_px"
        ] = float(max_clearance_px)
        record[
            "placement_core_points"
        ] = int(placement_core_points)

        print(
            f"bbox={[x, y, width, height]} "
            f"mask_shape={local_opening_mask.shape}",
            flush=True,
        )
        record[
            "outer_bbox_xywh"
        ] = list(candidate["bbox"])
        record[
            "opening_bbox_xyxy"
        ] = [
            int(ix),
            int(iy),
            int(ix + iw),
            int(iy + ih),
        ]
        record[
            "detected_center_uv"
        ] = [
            float(center_u),
            float(center_v),
        ]
        record[
            "black_mean_value"
        ] = float(
            candidate["mean_value"]
        )
        record[
            "black_component_score"
        ] = float(
            candidate["score"]
        )

        live_openings.append(record)

        selected_diag.append(
            {
                "outer_bbox": (
                    candidate["bbox"]
                ),
                "opening_bbox": [
                    int(ix),
                    int(iy),
                    int(ix + iw),
                    int(iy + ih),
                ],
                "center_uv": [
                    round(center_u, 2),
                    round(center_v, 2),
                ],
                "mean_v": round(
                    candidate["mean_value"],
                    2,
                ),
            }
        )

    openings = live_openings

    print(
        f"selected={selected_diag}",
        flush=True,
    )

    # Una caja no puede ocupar una región global.
    for item in openings:
        x, y, width, height = item[
            "outer_bbox_xywh"
        ]

        if (
            width > image_w * 0.42
            or height > image_h * 0.45
            or x <= 3
            or y <= 3
        ):
            raise RuntimeError(
                "Guard anti-tablero V5: "
                f"bbox={[x, y, width, height]}"
            )

    # Los bucles de detección de cajas usan width/height locales.
    # Se restauran las dimensiones completas antes de crear
    # máscaras de objetos y consultar el PLY global.
    height, width = board_mask.shape[:2]

    print(
        f"height={height} width={width}",
        flush=True,
    )

    for opening in openings:
        opening[
            "circle_likeness_score"
        ] = circle_likeness(
            opening
        )

    openings = sorted(
        openings,
        key=lambda item: (
            item[
                "circle_likeness_score"
            ],
            item[
                "circle_fill_ratio"
            ],
            -item[
                "rectangularity"
            ],
        ),
    )

    square_opening = openings[0]
    circular_opening = openings[1]

    # Centro geométrico independiente de sombras:
    # - rectángulo mínimo para la abertura cuadrada;
    # - círculo envolvente para la abertura circular.
    def set_geometric_opening_center(
        opening,
        shape_name,
    ):
        contour = opening["contour"]

        if shape_name == "square":
            # El contorno amarillo es irregular por sombras y
            # perspectiva. Para la abertura cuadrada se usa el
            # centro exacto de su bounding box detectado.
            bx, by, bw, bh = cv2.boundingRect(
                contour
            )

            geometric_center = (
                float(bx) + (float(bw) - 1.0) / 2.0,
                float(by) + (float(bh) - 1.0) / 2.0,
            )

            method = "opening_bounding_box_center"
        else:
            geometric_center, _ = (
                cv2.minEnclosingCircle(contour)
            )
            method = "minimum_enclosing_circle_center"

        center_u = float(geometric_center[0])
        center_v = float(geometric_center[1])

        outer_bbox = opening.get(
            "outer_bbox_xywh"
        )

        if outer_bbox is None:
            x, y, width, height = cv2.boundingRect(
                contour
            )
        else:
            x, y, width, height = [
                int(value)
                for value in outer_bbox
            ]

        full_opening_mask = opening.get(
            "mask_full_image"
        )

        if full_opening_mask is None:
            full_opening_mask = np.zeros_like(
                board_mask,
                dtype=np.uint8,
            )

            cv2.drawContours(
                full_opening_mask,
                [contour],
                -1,
                255,
                thickness=-1,
            )

        local_opening_mask = full_opening_mask[
            y:y + height,
            x:x + width,
        ].copy()

        if local_opening_mask.shape != (
            height,
            width,
        ):
            raise RuntimeError(
                "Máscara local geométrica inválida: "
                f"shape={shape_name} "
                f"bbox={[x, y, width, height]} "
                f"mask={local_opening_mask.shape}"
            )

        local_center_u = int(
            round(center_u - x)
        )
        local_center_v = int(
            round(center_v - y)
        )

        core_radius = int(
            max(
                6,
                min(
                    10,
                    round(
                        min(width, height)
                        * 0.075
                    ),
                ),
            )
        )

        placement_core = np.zeros_like(
            local_opening_mask,
            dtype=np.uint8,
        )

        cv2.circle(
            placement_core,
            (
                local_center_u,
                local_center_v,
            ),
            core_radius,
            255,
            thickness=-1,
        )

        placement_core = cv2.bitwise_and(
            placement_core,
            local_opening_mask,
        )

        core_points = int(
            cv2.countNonZero(
                placement_core
            )
        )

        if core_points < 30:
            raise RuntimeError(
                "Centro geométrico sin puntos PLY suficientes: "
                f"shape={shape_name} "
                f"center={[center_u, center_v]} "
                f"points={core_points}"
            )

        opening["mask"] = placement_core
        opening[
            "mask_full_image"
        ] = full_opening_mask
        opening[
            "placement_center_uv"
        ] = [
            center_u,
            center_v,
        ]
        opening[
            "detected_center_uv"
        ] = [
            center_u,
            center_v,
        ]
        opening[
            "opening_center_uv"
        ] = [
            center_u,
            center_v,
        ]
        opening[
            "opening_center_method"
        ] = method
        opening[
            "placement_core_points"
        ] = core_points

        print(
            f"shape={shape_name} "
            f"center_uv="
            f"{[round(center_u, 2), round(center_v, 2)]} "
            f"core_radius={core_radius} "
            f"core_points={core_points} "
            f"method={method}",
            flush=True,
        )

    set_geometric_opening_center(
        square_opening,
        "square",
    )

    set_geometric_opening_center(
        circular_opening,
        "circular",
    )


    shape_score_gap = float(
        circular_opening[
            "circle_likeness_score"
        ]
        - square_opening[
            "circle_likeness_score"
        ]
    )

    shape_assignment_low_confidence = (
        shape_score_gap
        < 0.08
    )

    print(
        "square_score=",
        round(
            square_opening[
                "circle_likeness_score"
            ],
            6,
        ),
        "square_circularity=",
        round(
            square_opening[
                "circularity"
            ],
            6,
        ),
        "square_rectangularity=",
        round(
            square_opening[
                "rectangularity"
            ],
            6,
        ),
        "square_circle_fill=",
        round(
            square_opening[
                "circle_fill_ratio"
            ],
            6,
        ),
        "square_radial_cv=",
        round(
            square_opening[
                "radial_cv"
            ],
            6,
        ),
        "square_vertices=",
        square_opening[
            "vertices_epsilon_04"
        ],
    )

    print(
        "circular_score=",
        round(
            circular_opening[
                "circle_likeness_score"
            ],
            6,
        ),
        "circular_circularity=",
        round(
            circular_opening[
                "circularity"
            ],
            6,
        ),
        "circular_rectangularity=",
        round(
            circular_opening[
                "rectangularity"
            ],
            6,
        ),
        "circular_circle_fill=",
        round(
            circular_opening[
                "circle_fill_ratio"
            ],
            6,
        ),
        "circular_radial_cv=",
        round(
            circular_opening[
                "radial_cv"
            ],
            6,
        ),
        "circular_vertices=",
        circular_opening[
            "vertices_epsilon_04"
        ],
    )

    print(
        "score_gap=",
        round(
            shape_score_gap,
            6,
        ),
        "low_confidence=",
        shape_assignment_low_confidence,
    )

    if shape_assignment_low_confidence:
        print(
            "Asignación relativa conservada; "
            "revisar overlay antes de movimiento."
        )

    dark_mask = clean_mask(
        cv2.bitwise_and(
            cv2.inRange(
                hsv,
                np.array(
                    [
                        0,
                        0,
                        0,
                    ]
                ),
                np.array(
                    [
                        179,
                        255,
                        95,
                    ]
                ),
            ),
            board_mask,
        )
    )

    objects = {}

    for (
        name,
        shape_name,
        detection,
    ) in [
        (
            "orange_cube",
            "cube",
            orange_record,
        ),
        (
            "white_cylinder",
            "cylinder",
            white_record,
        ),
    ]:
        object_mask = (
            contour_to_mask(
                (
                    height,
                    width,
                ),
                detection[
                    "contour"
                ],
            )
        )

        (
            object_xyz,
            point_count,
        ) = robust_xyz(
            base_xyz,
            finite,
            object_mask,
            90,
            name,
        )

        objects[
            name
        ] = {
            "name":
                name,
            "canonical":
                name,
            "role":
                "object",
            "object_shape":
                shape_name,
            "bbox_xyxy":
                detection[
                    "bbox_xyxy"
                ],
            "base_xyz_m":
                object_xyz,
            "center_base_xyz_m":
                object_xyz,
            "points_3d":
                point_count,
            "source":
                "photoneo_shape_geometry",
        }

    targets = {}

    for (
        name,
        shape_name,
        opening,
    ) in [
        (
            "square_box",
            "square",
            square_opening,
        ),
        (
            "circular_box",
            "circular",
            circular_opening,
        ),
    ]:
        crop_bbox = expand_bbox(
            opening[
                "bbox_xyxy"
            ],
            (
                height,
                width,
            ),
            scale=2.25,
        )

        x1, y1, x2, y2 = (
            crop_bbox
        )

        search_mask = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )

        search_mask[
            y1:y2 + 1,
            x1:x2 + 1,
        ] = 255

        rim_mask = cv2.bitwise_and(
            dark_mask,
            search_mask,
        )

        dilated_opening = (
            cv2.dilate(
                opening.get("mask_full_image", opening["mask"]),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        9,
                        9,
                    ),
                ),
            )
        )

        rim_mask = cv2.bitwise_and(
            rim_mask,
            cv2.bitwise_not(
                dilated_opening
            ),
        )

        (
            opening_xyz,
            opening_points,
        ) = robust_xyz(
            base_xyz,
            finite,
            opening.get("mask_full_image", opening["mask"]),
            50,
            f"{name}.opening",
        )

        (
            rim_xyz,
            rim_points,
        ) = robust_xyz(
            base_xyz,
            finite,
            rim_mask,
            80,
            f"{name}.rim",
        )

        target_xyz = [
            float(
                opening_xyz[0]
            ),
            float(
                opening_xyz[1]
            ),
            float(
                rim_xyz[2]
            ),
        ]

        targets[
            name
        ] = {
            "name":
                name,
            "canonical":
                name,
            "role":
                "target",
            "container_shape":
                shape_name,
            "bbox_xyxy":
                crop_bbox,
            "opening_bbox_xyxy":
                opening[
                    "bbox_xyxy"
                ],
            "opening_center_uv":
                opening[
                    "center_uv"
                ],
            "opening_circularity":
                opening[
                    "circularity"
                ],
            "opening_fill_ratio":
                opening[
                    "fill_ratio"
                ],
            "opening_rectangularity":
                opening[
                    "rectangularity"
                ],
            "opening_circle_fill_ratio":
                opening[
                    "circle_fill_ratio"
                ],
            "opening_radial_cv":
                opening[
                    "radial_cv"
                ],
            "opening_solidity":
                opening[
                    "solidity"
                ],
            "opening_circle_likeness_score":
                opening[
                    "circle_likeness_score"
                ],
            "opening_shape_assignment_low_confidence":
                shape_assignment_low_confidence,
            "opening_shape_score_gap":
                shape_score_gap,
            "opening_vertices":
                opening[
                    "vertices"
                ],
            "base_xyz_m":
                target_xyz,
            "center_base_xyz_m":
                target_xyz,
            "selected_interior_xyz_m":
                target_xyz,
            "opening_floor_xyz_m":
                opening_xyz,
            "rim_reference_xyz_m":
                rim_xyz,
            "opening_points_3d":
                opening_points,
            "rim_points_3d":
                rim_points,
            "container_interior_valid":
                True,
            "gripper_enters_container":
                False,
            "release_policy":
                "object_near_rim_gripper_above_container",
            "source":
                "photoneo_shape_geometry",
        }

    scene = {
        "schema":
            "VLA_shape_classification_scene",
        "scene_id":
            datetime.now().strftime(
                "SHAPE_%Y%m%dT%H%M%S"
            ),
        "timestamp":
            datetime.now().isoformat(),
        "source":
            "photoneo_live_ply",
        "ply_path":
            str(
                PLY_PATH
            ),
        "image_width":
            int(
                width
            ),
        "image_height":
            int(
                height
            ),
        "objects":
            objects,
        "targets":
            targets,
        "orange_cube":
            objects[
                "orange_cube"
            ],
        "white_cylinder":
            objects[
                "white_cylinder"
            ],
        "square_box":
            targets[
                "square_box"
            ],
        "circular_box":
            targets[
                "circular_box"
            ],
        "shape_correspondence": {
            "orange_cube":
                "square_box",
            "white_cylinder":
                "circular_box",
        },
        "place_policy": {
            "gripper_enters_container":
                False,
            "require_object_fully_below_rim":
                False,
            "success_rule":
                "object centroid inside opening after release",
        },
        "all_required_detected":
            True,
        "container_interior_valid":
            True,
        "robot_motion":
            False,
    }

    overlay = bgr.copy()

    colors = {
        "orange_cube":
            (
                0,
                140,
                255,
            ),
        "white_cylinder":
            (
                255,
                255,
                255,
            ),
        "square_box":
            (
                255,
                100,
                0,
            ),
        "circular_box":
            (
                255,
                0,
                255,
            ),
    }

    all_entities = {
        **objects,
        **targets,
    }

    for (
        name,
        entity,
    ) in all_entities.items():
        (
            x1,
            y1,
            x2,
            y2,
        ) = entity[
            "bbox_xyxy"
        ]

        color = colors[
            name
        ]

        cv2.rectangle(
            overlay,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            color,
            3,
        )

        cv2.putText(
            overlay,
            name,
            (
                x1,
                max(
                    20,
                    y1 - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        if name in targets:
            overlay_center_uv = entity.get(
                "placement_center_uv",
                entity.get(
                    "detected_center_uv",
                    entity["opening_center_uv"],
                ),
            )

            center_u = int(
                round(overlay_center_uv[0])
            )
            center_v = int(
                round(overlay_center_uv[1])
            )


            cv2.drawMarker(
                overlay,
                (
                    center_u,
                    center_v,
                ),
                color,
                markerType=(
                    cv2.MARKER_CROSS
                ),
                markerSize=24,
                thickness=3,
            )

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_dir = (
        RESULTS_ROOT
        / (
            "forma_"
            + timestamp
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    scene_path = (
        output_dir
        / "resultado_forma.json"
    )

    rgb_path = (
        output_dir
        / "photoneo_rgb.png"
    )

    overlay_path = (
        output_dir
        / "shape_detection_overlay.png"
    )

    scene_path.write_text(
        json.dumps(
            scene,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cv2.imwrite(
        str(
            rgb_path
        ),
        bgr,
    )

    cv2.imwrite(
        str(
            overlay_path
        ),
        overlay,
    )

    latest_path = (
        RESULTS_ROOT
        / "forma_ultima.json"
    )

    latest_path.write_text(
        json.dumps(
            scene,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "PASS"
    )

    print(
        "SCENE_JSON=",
        scene_path,
    )

    print(
        "OVERLAY=",
        overlay_path,
    )

    for (
        name,
        entity,
    ) in objects.items():
        print(
            "OBJECT",
            name,
            "bbox=",
            entity[
                "bbox_xyxy"
            ],
            "xyz=",
            entity[
                "base_xyz_m"
            ],
        )

    for (
        name,
        entity,
    ) in targets.items():
        print(
            "TARGET",
            name,
            "bbox=",
            entity[
                "bbox_xyxy"
            ],
            "opening=",
            entity[
                "opening_bbox_xyxy"
            ],
            "xyz=",
            entity[
                "base_xyz_m"
            ],
            "rim_points=",
            entity[
                "rim_points_3d"
            ],
        )

    print(
        "GRIPPER_ENTERS_CONTAINER=False"
    )

    print(
        "ROBOT_MOTION=False"
    )


main()

# La identidad física de los recipientes se valida mediante anclas 3D
# calibradas. No utiliza la posición vertical de la imagen.
# Si las cajas se desplazan, falta alguna detección o la asignación es
# ambigua, el detector termina con error antes de que pueda moverse el robot.
# Para el grounding visual se usa un crop centrado en la abertura real,
# conservando por separado la bbox exterior original.

if __name__ == "__main__":
    import copy as _copy
    import itertools as _itertools
    import json as _json
    import math as _math
    import time as _time
    from pathlib import Path as _Path

    _CONTAINER_ANCHORS_XY = {
        "square_box": (-0.2413601178817749, -0.7694806832885742),
        "circular_box": (-0.1347314877243042, -0.7672300904693603),
    }

    _MAX_ANCHOR_DISTANCE_M = 0.040
    _MIN_ASSIGNMENT_GAP_M = 0.080
    _OPENING_CROP_PADDING_PX = 14

    def _xy(entry):
        xyz = entry.get("base_xyz_m") or entry.get("center_base_xyz_m")
        if not isinstance(xyz, (list, tuple)) or len(xyz) < 2:
            raise RuntimeError(
                "target sin base_xyz_m válido"
            )
        return float(xyz[0]), float(xyz[1])

    def _distance(entry, anchor):
        x, y = _xy(entry)
        return _math.hypot(x - anchor[0], y - anchor[1])

    _project_root = _Path(__file__).resolve().parents[1]
    _results_root = (
        _project_root
        / "results"
        / "validacion_robot"
    )

    _scene_files = sorted(
        _results_root.glob(
            "forma_*/"
            "resultado_forma.json"
        ),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )

    if not _scene_files:
        raise RuntimeError(
            "no se encontró el JSON recién generado"
        )

    _scene_path = _scene_files[0]
    _age_s = _time.time() - _scene_path.stat().st_mtime

    if _age_s > 120.0:
        raise RuntimeError(
            f"JSON no reciente: age={_age_s:.3f}s path={_scene_path}"
        )

    _scene = _json.loads(
        _scene_path.read_text(encoding="utf-8")
    )

    _targets_dict = _scene.get("targets")

    if not isinstance(_targets_dict, dict):
        raise RuntimeError(
            "scene.targets no es un diccionario"
        )

    _candidates = [
        _copy.deepcopy(value)
        for value in _targets_dict.values()
        if isinstance(value, dict)
        and (
            value.get("role") == "target"
            or value.get("container_shape") in {"square", "circular"}
        )
    ]

    if len(_candidates) != 2:
        raise RuntimeError(
            f"se esperaban 2 recipientes y aparecieron {len(_candidates)}"
        )

    _hypotheses = []

    for permutation in _itertools.permutations(_candidates):
        assignment = {
            "square_box": permutation[0],
            "circular_box": permutation[1],
        }

        distances = {
            name: _distance(
                assignment[name],
                _CONTAINER_ANCHORS_XY[name],
            )
            for name in ("square_box", "circular_box")
        }

        _hypotheses.append(
            (
                sum(distances.values()),
                assignment,
                distances,
            )
        )

    _hypotheses.sort(key=lambda item: item[0])

    _best_cost, _best_assignment, _best_distances = _hypotheses[0]
    _second_cost = _hypotheses[1][0]
    _assignment_gap = _second_cost - _best_cost
    _max_distance = max(_best_distances.values())

    if _max_distance > _MAX_ANCHOR_DISTANCE_M:
        raise RuntimeError(
            f"alguna caja está demasiado lejos de su ancla: "
            f"distances={_best_distances} "
            f"limit={_MAX_ANCHOR_DISTANCE_M}"
        )

    if _assignment_gap < _MIN_ASSIGNMENT_GAP_M:
        raise RuntimeError(
            f"asignación ambigua: gap={_assignment_gap:.6f} "
            f"minimum={_MIN_ASSIGNMENT_GAP_M}"
        )

    _image_width = int(_scene.get("image_width", 1120))
    _image_height = int(_scene.get("image_height", 800))

    _locked_targets = {}

    for canonical_name in ("square_box", "circular_box"):
        target = _copy.deepcopy(
            _best_assignment[canonical_name]
        )

        original_name = target.get("name")
        original_bbox = target.get("bbox_xyxy")
        opening_bbox = target.get("opening_bbox_xyxy")

        if (
            not isinstance(opening_bbox, (list, tuple))
            or len(opening_bbox) != 4
        ):
            raise RuntimeError(
                f"{canonical_name} no tiene opening_bbox_xyxy válido"
            )

        x1, y1, x2, y2 = [
            int(round(float(value)))
            for value in opening_bbox
        ]

        pad = _OPENING_CROP_PADDING_PX

        grounding_bbox = [
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(_image_width - 1, x2 + pad),
            min(_image_height - 1, y2 + pad),
        ]

        target["detector_original_name"] = original_name
        target["name"] = canonical_name
        target["canonical"] = canonical_name
        target["role"] = "target"
        target["container_shape"] = (
            "square"
            if canonical_name == "square_box"
            else "circular"
        )

        target["outer_bbox_xyxy"] = original_bbox
        target["geometry_bbox_xyxy"] = original_bbox

        # Crop visual usado por el grounding: abertura + contexto próximo.
        target["bbox_xyxy"] = grounding_bbox
        target["grounding_bbox_xyxy"] = grounding_bbox
        target["grounding_crop_source"] = (
            "opening_bbox_xyxy_plus_padding"
        )
        target["container_identity_source"] = (
            "calibrated_3d_anchor_lock"
        )
        target["container_anchor_distance_m"] = (
            _best_distances[canonical_name]
        )

        _locked_targets[canonical_name] = target

    _scene["targets"] = _locked_targets
    _scene["square_box"] = _locked_targets["square_box"]
    _scene["circular_box"] = _locked_targets["circular_box"]

    _scene["container_identity_lock"] = {
        "schema": "VLA_SHAPE_CONTAINER_3D_IDENTITY_LOCK",
        "status": "PASS",
        "position_based_relabel": False,
        "identity_source": "calibrated_base_frame_xy",
        "anchors_xy_m": {
            key: list(value)
            for key, value in _CONTAINER_ANCHORS_XY.items()
        },
        "distances_m": _best_distances,
        "best_assignment_cost_m": _best_cost,
        "second_assignment_cost_m": _second_cost,
        "assignment_gap_m": _assignment_gap,
        "maximum_anchor_distance_m": _MAX_ANCHOR_DISTANCE_M,
        "minimum_assignment_gap_m": _MIN_ASSIGNMENT_GAP_M,
        "opening_grounding_crop": True,
    }

    _tmp_path = _scene_path.with_suffix(".json.tmp")

    _tmp_path.write_text(
        _json.dumps(
            _scene,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    _tmp_path.replace(_scene_path)

    print(
        f"status=PASS "
        f"square_xyz={_locked_targets['square_box'].get('base_xyz_m')} "
        f"circular_xyz={_locked_targets['circular_box'].get('base_xyz_m')} "
        f"distances={_best_distances} "
        f"assignment_gap={_assignment_gap:.6f} "
        f"scene={_scene_path}"
    )

    print(
        f"square_bbox={_locked_targets['square_box']['bbox_xyxy']} "
        f"circular_bbox={_locked_targets['circular_box']['bbox_xyxy']}"
    )
