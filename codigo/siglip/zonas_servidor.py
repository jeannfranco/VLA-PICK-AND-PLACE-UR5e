#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import threading
import traceback
import uuid

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]

SIGLIP_BASE_PATH = (
    ROOT
    / "scripts"
    / "siglip_base.py"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "siglip_grounding_service"
)

MODEL_ID = "google/siglip-base-patch16-224"

MIN_QUERY_PROBABILITY_MARGIN = 0.20
MIN_ROLE_ASSIGNMENT_MARGIN = 1.00

LABELS = {
    "object": (
        {
            "canonical": "orange_cube",
            "description": "an orange cube",
        },
        {
            "canonical": "white_cylinder",
            "description": "a white cylinder",
        },
    ),
    "target": (
        {
            "canonical": "blue_zone",
            "description": "a blue rectangular area",
        },
        {
            "canonical": "yellow_zone",
            "description": "a yellow rectangular area",
        },
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8776,
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cuda",
            "cpu",
        ),
        default="auto",
    )

    return parser.parse_args()


def load_siglip_base():
    specification = (
        importlib.util.spec_from_file_location(
            "siglip_base",
            SIGLIP_BASE_PATH,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"No se pudo importar {SIGLIP_BASE_PATH}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    sys.modules[
        specification.name
    ] = module

    specification.loader.exec_module(
        module
    )

    required = (
        "MODEL_ID",
        "PROMPT_TEMPLATES",
        "CROP_PADDING_RATIOS",
        "square_candidate_crop",
        "exact_bbox_candidate_crop",
        "resolve_device",
        "load_model",
        "move_inputs",
        "softmax",
    )

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            "El módulo SigLIP no contiene: "
            f"{missing}"
        )

    source = SIGLIP_BASE_PATH.read_text(
        encoding="utf-8"
    )

    if (
        "VLA_SIGLIP_ROLE_SPECIFIC_CROPS"
        not in source
    ):
        raise RuntimeError(
            "El módulo SigLIP no cumple el contrato esperado"
        )

    if module.MODEL_ID != MODEL_ID:
        raise RuntimeError(
            "Modelo distinto al validado: "
            f"{module.MODEL_ID}"
        )

    return module


def finite_bbox(
    raw: Any,
    *,
    image_width: int,
    image_height: int,
    label: str,
) -> list[float]:
    if (
        not isinstance(raw, list)
        or len(raw) != 4
    ):
        raise RuntimeError(
            f"{label}: bbox_xyxy inválida"
        )

    try:
        bbox = [
            float(value)
            for value in raw
        ]
    except Exception as error:
        raise RuntimeError(
            f"{label}: bbox no numérica"
        ) from error

    if not all(
        math.isfinite(value)
        for value in bbox
    ):
        raise RuntimeError(
            f"{label}: bbox no finita"
        )

    x1, y1, x2, y2 = bbox

    if (
        x2 <= x1
        or y2 <= y1
        or x1 < 0.0
        or y1 < 0.0
        or x2 > image_width
        or y2 > image_height
    ):
        raise RuntimeError(
            f"{label}: bbox fuera de imagen: "
            f"{bbox}, image="
            f"{image_width}x{image_height}"
        )

    return bbox


def decode_image(
    request_data: dict[str, Any],
) -> tuple[Image.Image, bytes, str]:
    encoded = request_data.get(
        "image_base64"
    )

    if not isinstance(encoded, str):
        raise RuntimeError(
            "Falta image_base64"
        )

    try:
        image_bytes = base64.b64decode(
            encoded,
            validate=True,
        )
    except Exception as error:
        raise RuntimeError(
            "image_base64 inválido"
        ) from error

    calculated_sha = hashlib.sha256(
        image_bytes
    ).hexdigest()

    declared_sha = str(
        request_data.get(
            "image_sha256",
            "",
        )
    ).strip()

    if (
        declared_sha
        and declared_sha != calculated_sha
    ):
        raise RuntimeError(
            "El SHA256 declarado no coincide "
            "con la imagen recibida"
        )

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    return (
        image,
        image_bytes,
        calculated_sha,
    )


def validate_candidates(
    role: str,
    raw_candidates: Any,
    image: Image.Image,
) -> list[dict[str, Any]]:
    if not isinstance(
        raw_candidates,
        list,
    ):
        raise RuntimeError(
            f"{role}: candidates no es una lista"
        )

    if len(raw_candidates) != 2:
        raise RuntimeError(
            f"{role}: el contrato validado exige "
            "exactamente 2 candidatos; "
            f"recibidos={len(raw_candidates)}"
        )

    result = []
    identifiers = set()

    for index, candidate in enumerate(
        raw_candidates,
        start=1,
    ):
        if not isinstance(
            candidate,
            dict,
        ):
            raise RuntimeError(
                f"{role}[{index}]: candidato inválido"
            )

        candidate_id = str(
            candidate.get(
                "candidate_id",
                "",
            )
        ).strip()

        if not candidate_id:
            raise RuntimeError(
                f"{role}[{index}]: candidate_id vacío"
            )

        if candidate_id in identifiers:
            raise RuntimeError(
                f"{role}: candidate_id duplicado: "
                f"{candidate_id}"
            )

        identifiers.add(
            candidate_id
        )

        bbox = finite_bbox(
            candidate.get(
                "bbox_xyxy"
            ),
            image_width=image.width,
            image_height=image.height,
            label=(
                f"{role}.{candidate_id}"
            ),
        )

        result.append(
            {
                "candidate_id":
                    candidate_id,
                "bbox_xyxy":
                    bbox,
            }
        )

    return result


def score_role(
    *,
    siglip_base,
    image: Image.Image,
    role: str,
    candidates: list[dict[str, Any]],
    requested_label: str,
    processor,
    model,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> dict[str, Any]:
    labels = list(
        LABELS[role]
    )

    allowed_labels = {
        label["canonical"]
        for label in labels
    }

    if requested_label not in allowed_labels:
        raise RuntimeError(
            f"{role}: requested_label no soportado: "
            f"{requested_label}"
        )

    crops = []
    crop_metadata = []

    for candidate_index, candidate in (
        enumerate(candidates)
    ):
        if role == "target":
            crop_variants = (
                {
                    "name": "exact_bbox",
                    "padding_ratio": None,
                },
            )

        else:
            crop_variants = tuple(
                {
                    "name":
                        f"padding_{padding:.2f}",
                    "padding_ratio":
                        float(padding),
                }
                for padding
                in siglip_base.CROP_PADDING_RATIOS
            )

        for variant in crop_variants:
            padding_ratio = variant[
                "padding_ratio"
            ]

            if role == "target":
                crop = (
                    siglip_base
                    .exact_bbox_candidate_crop(
                        image,
                        candidate["bbox_xyxy"],
                    )
                )

            else:
                crop = (
                    siglip_base
                    .square_candidate_crop(
                        image,
                        candidate["bbox_xyxy"],
                        float(padding_ratio),
                    )
                )

            crops.append(crop)

            crop_metadata.append(
                {
                    "candidate_index":
                        candidate_index,
                    "candidate_id":
                        candidate["candidate_id"],
                    "variant":
                        variant["name"],
                }
            )

            crop.save(
                output_dir
                / (
                    f"{role}_"
                    f"{candidate['candidate_id']}_"
                    f"{variant['name']}.png"
                )
            )

    texts = []
    text_metadata = []

    for label_index, label in (
        enumerate(labels)
    ):
        for template in (
            siglip_base.PROMPT_TEMPLATES
        ):
            texts.append(
                template.format(
                    label["description"]
                )
            )

            text_metadata.append(
                {
                    "label_index":
                        label_index,
                    "canonical":
                        label["canonical"],
                }
            )

    inputs = processor(
        images=crops,
        text=texts,
        padding="max_length",
        return_tensors="pt",
    )

    inputs = siglip_base.move_inputs(
        inputs,
        device=device,
        dtype=dtype,
    )

    with ENGINE_LOCK:
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(
                    device.type == "cuda"
                ),
            ):
                outputs = model(**inputs)

    raw_logits = (
        outputs.logits_per_image
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    score_matrix = np.zeros(
        (
            len(candidates),
            len(labels),
        ),
        dtype=float,
    )

    score_counts = np.zeros_like(
        score_matrix
    )

    for crop_index, crop_info in (
        enumerate(crop_metadata)
    ):
        candidate_index = int(
            crop_info["candidate_index"]
        )

        for text_index, text_info in (
            enumerate(text_metadata)
        ):
            label_index = int(
                text_info["label_index"]
            )

            score_matrix[
                candidate_index,
                label_index,
            ] += float(
                raw_logits[
                    crop_index,
                    text_index,
                ]
            )

            score_counts[
                candidate_index,
                label_index,
            ] += 1.0

    if np.any(
        score_counts <= 0.0
    ):
        raise RuntimeError(
            f"{role}: matriz de scores incompleta"
        )

    score_matrix = (
        score_matrix
        / score_counts
    )

    query_results = []

    for label_index, label in (
        enumerate(labels)
    ):
        candidate_scores = (
            score_matrix[
                :,
                label_index,
            ]
        )

        ranking = np.argsort(
            -candidate_scores
        )

        selected_index = int(
            ranking[0]
        )

        second_index = int(
            ranking[1]
        )

        probabilities = siglip_base.softmax(
            candidate_scores
        )

        candidate_top_label = int(
            np.argmax(
                score_matrix[
                    selected_index,
                    :,
                ]
            )
        )

        mutual_top1 = bool(
            candidate_top_label
            == label_index
        )

        candidate_label_scores = {
            labels[index]["canonical"]: float(
                score_matrix[
                    selected_index,
                    index,
                ]
            )
            for index in range(len(labels))
        }

        print(
            f"query_label={label['canonical']} "
            f"selected_candidate="
            f"{candidates[selected_index]['candidate_id']} "
            f"candidate_top_label="
            f"{labels[candidate_top_label]['canonical']} "
            f"candidate_label_scores="
            f"{candidate_label_scores} "
            f"mutual_top1="
            f"{str(mutual_top1).lower()}",
            flush=True,
        )

        probability_margin = float(
            probabilities[selected_index]
            - probabilities[second_index]
        )

        robust_pass = bool(
            mutual_top1
            and probability_margin
            >= MIN_QUERY_PROBABILITY_MARGIN
        )

        query_results.append(
            {
                "canonical":
                    label["canonical"],
                "description":
                    label["description"],
                "selected_candidate_id":
                    candidates[
                        selected_index
                    ]["candidate_id"],
                "selected_probability":
                    float(
                        probabilities[
                            selected_index
                        ]
                    ),
                "probability_margin":
                    probability_margin,
                "mutual_top1":
                    mutual_top1,
                "robust_pass":
                    robust_pass,
                "candidate_ranking": [
                    {
                        "candidate_id":
                            candidates[
                                int(index)
                            ]["candidate_id"],
                        "score":
                            float(
                                candidate_scores[
                                    int(index)
                                ]
                            ),
                        "probability":
                            float(
                                probabilities[
                                    int(index)
                                ]
                            ),
                    }
                    for index in ranking
                ],
            }
        )

    assignments = []

    for permutation in itertools.permutations(
        range(len(candidates)),
        len(labels),
    ):
        total_score = float(
            sum(
                score_matrix[
                    permutation[label_index],
                    label_index,
                ]
                for label_index
                in range(len(labels))
            )
        )

        assignments.append(
            {
                "candidate_indices":
                    list(permutation),
                "total_score":
                    total_score,
            }
        )

    assignments.sort(
        key=lambda item:
            item["total_score"],
        reverse=True,
    )

    best_assignment = assignments[0]
    second_assignment = assignments[1]

    assignment_margin = float(
        best_assignment["total_score"]
        - second_assignment["total_score"]
    )

    label_to_candidate = {
        labels[label_index]["canonical"]:
            candidates[
                best_assignment[
                    "candidate_indices"
                ][label_index]
            ]["candidate_id"]
        for label_index
        in range(len(labels))
    }

    query_by_label = {
        query["canonical"]:
            query
        for query in query_results
    }

    assignment_query_agreement = all(
        label_to_candidate[
            label["canonical"]
        ]
        == query_by_label[
            label["canonical"]
        ]["selected_candidate_id"]
        for label in labels
    )

    role_pass = bool(
        assignment_margin
        >= MIN_ROLE_ASSIGNMENT_MARGIN
        and assignment_query_agreement
        and all(
            query["robust_pass"]
            for query in query_results
        )
    )

    selected_candidate_id = (
        label_to_candidate[
            requested_label
        ]
    )

    requested_query = (
        query_by_label[
            requested_label
        ]
    )

    requested_assignment_query_agreement = bool(
        selected_candidate_id
        == requested_query[
            "selected_candidate_id"
        ]
    )

    return {
        "role":
            role,
        "requested_label":
            requested_label,
        "selected_candidate_id":
            selected_candidate_id,
        "candidate_order": [
            candidate["candidate_id"]
            for candidate in candidates
        ],
        "label_order": [
            label["canonical"]
            for label in labels
        ],
        "score_matrix":
            score_matrix.tolist(),
        "queries":
            query_results,
        "requested_query":
            requested_query,
        "label_to_candidate":
            label_to_candidate,
        "best_assignment_score":
            float(
                best_assignment[
                    "total_score"
                ]
            ),
        "second_assignment_score":
            float(
                second_assignment[
                    "total_score"
                ]
            ),
        "assignment_margin":
            assignment_margin,
        "minimum_query_probability_margin":
            MIN_QUERY_PROBABILITY_MARGIN,
        "minimum_assignment_margin":
            MIN_ROLE_ASSIGNMENT_MARGIN,
        "assignment_query_agreement":
            assignment_query_agreement,
        "requested_assignment_query_agreement":
            requested_assignment_query_agreement,
        "assignment_query_agreement_scope":
            "global_diagnostic_only",
        "execution_assignment_guard_scope":
            "requested_query",
        "pass":
            role_pass,
    }


def draw_overview(
    image: Image.Image,
    roles: dict[
        str,
        list[dict[str, Any]],
    ],
    output_path: Path,
) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    for role, candidates in roles.items():
        for candidate in candidates:
            bbox = tuple(
                candidate["bbox_xyxy"]
            )

            draw.rectangle(
                bbox,
                outline="white",
                width=3,
            )

            draw.text(
                (
                    bbox[0] + 3,
                    bbox[1] + 3,
                ),
                (
                    f"{role}:"
                    f"{candidate['candidate_id']}"
                ),
                fill="white",
            )

    canvas.save(
        output_path
    )


def process_grounding_request(
    request_data: dict[str, Any],
) -> dict[str, Any]:
    (
        image,
        image_bytes,
        image_sha256,
    ) = decode_image(
        request_data
    )

    roles_request = request_data.get(
        "roles"
    )

    if not isinstance(
        roles_request,
        dict,
    ):
        raise RuntimeError(
            "Falta roles"
        )

    timestamp = (
        datetime.now(timezone.utc)
        .strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )
    )

    output_dir = (
        OUTPUT_ROOT
        / (
            f"{timestamp}_"
            f"{uuid.uuid4().hex[:8]}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    input_path = (
        output_dir
        / "input_image.png"
    )

    input_path.write_bytes(
        image_bytes
    )

    validated_roles = {}

    for role in (
        "object",
        "target",
    ):
        role_request = roles_request.get(
            role
        )

        if not isinstance(
            role_request,
            dict,
        ):
            raise RuntimeError(
                f"Falta role={role}"
            )

        validated_roles[role] = (
            validate_candidates(
                role,
                role_request.get(
                    "candidates"
                ),
                image,
            )
        )

    draw_overview(
        image,
        validated_roles,
        output_dir
        / "neutral_candidate_overview.png",
    )

    role_results = []

    for role in (
        "object",
        "target",
    ):
        role_request = roles_request[
            role
        ]

        requested_label = str(
            role_request.get(
                "requested_label",
                "",
            )
        ).strip()

        role_results.append(
            score_role(
                siglip_base=SIGLIP_BASE,
                image=image,
                role=role,
                candidates=
                    validated_roles[role],
                requested_label=
                    requested_label,
                processor=PROCESSOR,
                model=MODEL,
                device=DEVICE,
                dtype=DTYPE,
                output_dir=output_dir,
            )
        )

    role_by_name = {
        result["role"]:
            result
        for result in role_results
    }

    all_pass = all(
        result["pass"]
        for result in role_results
    )

    response = {
        "schema":
            "VLA_siglip_grounding_service",
        "model_id":
            MODEL_ID,
        "contract":
            "neutral_photoneo_candidates_"
            "plus_frozen_siglip_crop_text_ranking",
        "candidate_semantic_names_received":
            False,
        "source_candidate_names_used":
            False,
        "image_sha256":
            image_sha256,
        "image_size": [
            image.width,
            image.height,
        ],
        "roles":
            role_by_name,
        "selected_object_candidate_id":
            role_by_name[
                "object"
            ]["selected_candidate_id"],
        "selected_target_candidate_id":
            role_by_name[
                "target"
            ]["selected_candidate_id"],
        "all_pass":
            all_pass,
        "output_dir":
            str(output_dir),
    }

    report_path = (
        output_dir
        / "siglip_grounding_service_response.json"
    )

    report_path.write_text(
        json.dumps(
            response,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"object="
        f"{response['selected_object_candidate_id']} "
        f"target="
        f"{response['selected_target_candidate_id']} "
        f"pass={str(all_pass).lower()} "
        f"report={report_path}",
        flush=True,
    )

    return response


class Handler(BaseHTTPRequestHandler):
    server_version = (
        "SigLIPGrounding/1.0"
    )

    def send_json(
        self,
        status: int,
        payload: dict[str, Any],
    ) -> None:
        raw = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(len(raw)),
        )
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json(
                404,
                {
                    "ok": False,
                    "error": "not_found",
                },
            )
            return

        self.send_json(
            200,
            {
                "ok": True,
                "service":
                    "VLA_siglip_grounding",
                "schema":
                    "VLA_siglip_grounding_health",
                "model_id":
                    MODEL_ID,
                "device":
                    DEVICE.type,
                "dtype":
                    str(DTYPE),
                "contract":
                    "VLA_SIGLIP_ROLE_SPECIFIC_CROPS",
                "minimum_query_probability_margin":
                    MIN_QUERY_PROBABILITY_MARGIN,
                "minimum_role_assignment_margin":
                    MIN_ROLE_ASSIGNMENT_MARGIN,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/ground":
            self.send_json(
                404,
                {
                    "ok": False,
                    "error": "not_found",
                },
            )
            return

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            if (
                content_length <= 0
                or content_length
                > 25_000_000
            ):
                raise RuntimeError(
                    "Content-Length inválido"
                )

            raw = self.rfile.read(
                content_length
            )

            request_data = json.loads(
                raw.decode("utf-8")
            )

            if not isinstance(
                request_data,
                dict,
            ):
                raise RuntimeError(
                    "El request no es un objeto JSON"
                )

            response = (
                process_grounding_request(
                    request_data
                )
            )

            self.send_json(
                200,
                response,
            )

        except Exception as error:
            print(
                f"{type(error).__name__}: {error}",
                flush=True,
            )

            traceback.print_exc()

            self.send_json(
                422,
                {
                    "schema":
                        "VLA_siglip_grounding_error",
                    "ok": False,
                    "error_type":
                        type(error).__name__,
                    "error":
                        str(error),
                },
            )

    def log_message(
        self,
        format_string: str,
        *arguments,
    ) -> None:
        print(
            + format_string % arguments,
            flush=True,
        )


ARGS = parse_args()

SIGLIP_BASE = load_siglip_base()

DEVICE = SIGLIP_BASE.resolve_device(
    ARGS.device
)

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

print(
    f"model={MODEL_ID} "
    f"device={DEVICE.type}",
    flush=True,
)

PROCESSOR, MODEL, DTYPE = (
    SIGLIP_BASE.load_model(
        DEVICE
    )
)

ENGINE_LOCK = threading.Lock()

print(
    f"model={MODEL_ID} "
    f"device={DEVICE.type} "
    f"dtype={DTYPE} "
    f"url=http://{ARGS.host}:{ARGS.port}",
    flush=True,
)

SERVER = ThreadingHTTPServer(
    (
        ARGS.host,
        ARGS.port,
    ),
    Handler,
)

SERVER.serve_forever()
