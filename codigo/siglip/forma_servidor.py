#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
import argparse
import base64
import hashlib
import json
import math
import os
import threading

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


ROOT = Path("/workspace") if Path("/workspace").is_dir() else Path(__file__).resolve().parents[1]
MODEL_ID = os.environ.get("VLA_SHAPE_SIGLIP_MODEL_ID", "google/siglip-base-patch16-224")
CONTRACT = "VLA_SIGLIP_SHAPE_CONTAINERS"
MIN_QUERY_MARGIN = float(os.environ.get("VLA_SHAPE_MIN_QUERY_PROBABILITY_MARGIN", "0.20"))
MIN_ASSIGNMENT_MARGIN = float(os.environ.get("VLA_SHAPE_MIN_ROLE_ASSIGNMENT_MARGIN", "1.00"))
LOCAL_FILES_ONLY = os.environ.get("VLA_SHAPE_SIGLIP_LOCAL_FILES_ONLY", "1") == "1"

OBJECT_PROMPTS = {
    "orange_cube": ["an orange cube", "an orange die", "a cube-shaped orange object"],
    "white_cylinder": ["a white cylinder", "a white cylindrical object", "a white round cylinder"],
}
TARGET_PROMPTS = {
    "square_box": [
        "a black container with a square opening",
        "a square-shaped black receptacle",
        "a square opening in a black box",
    ],
    "circular_box": [
        "a black container with a circular opening",
        "a round black receptacle",
        "a circular opening in a black container",
    ],
}
RESULTS_ROOT = ROOT / "results/clasificacion_forma/servicio"


class ShapeSiglipServiceError(RuntimeError):
    pass


def finite_bbox(raw: Any, *, label: str, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ShapeSiglipServiceError(f"{label}: bbox_xyxy inválida")
    try:
        values = [float(value) for value in raw]
    except Exception as error:
        raise ShapeSiglipServiceError(f"{label}: bbox no numérica") from error
    if not all(math.isfinite(value) for value in values):
        raise ShapeSiglipServiceError(f"{label}: bbox no finita")
    width, height = image_size
    x1, y1, x2, y2 = [int(round(value)) for value in values]
    x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
    x2, y2 = max(x1 + 1, min(x2, width)), max(y1 + 1, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ShapeSiglipServiceError(f"{label}: bbox degenerada")
    return x1, y1, x2, y2


def padded_square(
    bbox: tuple[int, int, int, int],
    *,
    image_size: tuple[int, int],
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * padding)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return finite_bbox(
        [cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0],
        label="padded_square",
        image_size=image_size,
    )


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    exponentials = np.exp(values - np.max(values))
    denominator = float(np.sum(exponentials))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ShapeSiglipServiceError("Softmax no válida")
    return exponentials / denominator


def decode_image(payload: dict[str, Any]) -> tuple[Image.Image, str]:
    encoded = payload.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ShapeSiglipServiceError("Falta image_base64")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as error:
        raise ShapeSiglipServiceError("image_base64 no válida") from error
    digest = hashlib.sha256(image_bytes).hexdigest()
    expected = str(payload.get("image_sha256", "")).strip()
    if expected and expected != digest:
        raise ShapeSiglipServiceError("El SHA-256 de la imagen no coincide")
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as error:
        raise ShapeSiglipServiceError("No se pudo decodificar la imagen") from error
    return image, digest


class ShapeSiglipModel:
    def __init__(self, device: str) -> None:
        self.device = device
        self.lock = threading.Lock()
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=LOCAL_FILES_ONLY)
        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            local_files_only=LOCAL_FILES_ONLY,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()

    def score_role(
        self,
        *,
        image: Image.Image,
        role: str,
        candidates: list[dict[str, Any]],
        prompts_by_class: dict[str, list[str]],
    ) -> dict[str, Any]:
        if not isinstance(candidates, list) or len(candidates) != 2:
            count = len(candidates) if isinstance(candidates, list) else "invalid"
            raise ShapeSiglipServiceError(f"{role}: se requieren dos candidatos; recibidos={count}")

        image_size = image.size
        candidate_ids: list[str] = []
        bboxes: dict[str, tuple[int, int, int, int]] = {}

        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                raise ShapeSiglipServiceError(f"{role}[{index}]: candidato inválido")
            unexpected = set(candidate) - {"candidate_id", "bbox_xyxy"}
            if unexpected:
                raise ShapeSiglipServiceError(
                    f"{role}[{index}]: claves no neutrales: {sorted(unexpected)}"
                )
            candidate_id = str(candidate.get("candidate_id", "")).strip()
            expected_id = f"{role}_{index:02d}"
            if candidate_id != expected_id:
                raise ShapeSiglipServiceError(
                    f"{role}[{index}]: candidate_id={candidate_id!r}; esperado={expected_id!r}"
                )
            candidate_ids.append(candidate_id)
            bboxes[candidate_id] = finite_bbox(
                candidate.get("bbox_xyxy"),
                label=f"{role}.{candidate_id}",
                image_size=image_size,
            )

        class_names = list(prompts_by_class)
        prompt_records = [
            {"class_name": class_name, "prompt_index": prompt_index, "prompt": prompt}
            for class_name in class_names
            for prompt_index, prompt in enumerate(prompts_by_class[class_name])
        ]
        texts = [record["prompt"] for record in prompt_records]
        logits = np.zeros((2, len(class_names)), dtype=float)
        crop_trace: dict[str, list[dict[str, Any]]] = {}

        for candidate_index, candidate_id in enumerate(candidate_ids):
            bbox = bboxes[candidate_id]
            if role == "object":
                crop_specs = [
                    ("padding_010", padded_square(bbox, image_size=image_size, padding=0.10)),
                    ("padding_030", padded_square(bbox, image_size=image_size, padding=0.30)),
                    ("padding_060", padded_square(bbox, image_size=image_size, padding=0.60)),
                ]
            else:
                crop_specs = [("container_context", bbox)]

            crops = [image.crop(crop_bbox).convert("RGB") for _, crop_bbox in crop_specs]
            crop_trace[candidate_id] = [
                {"crop_label": crop_label, "bbox_xyxy": list(crop_bbox)}
                for crop_label, crop_bbox in crop_specs
            ]
            inputs = self.processor(
                text=texts,
                images=crops,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with self.lock, torch.inference_mode():
                raw = self.model(**inputs).logits_per_image.detach().float().cpu().numpy()

            for class_index, class_name in enumerate(class_names):
                prompt_indices = [
                    index
                    for index, record in enumerate(prompt_records)
                    if record["class_name"] == class_name
                ]
                logits[candidate_index, class_index] = float(np.mean(raw[:, prompt_indices]))

        class_results: dict[str, dict[str, Any]] = {}
        for class_index, class_name in enumerate(class_names):
            query_logits = logits[:, class_index]
            probabilities = softmax(query_logits)
            order = np.argsort(query_logits)[::-1]
            best, second = int(order[0]), int(order[1])
            probability_margin = float(probabilities[best] - probabilities[second])
            class_results[class_name] = {
                "class_name": class_name,
                "selected_candidate_id": candidate_ids[best],
                "selected_probability": float(probabilities[best]),
                "second_candidate_id": candidate_ids[second],
                "second_probability": float(probabilities[second]),
                "probability_margin": probability_margin,
                "query_logit_margin": float(query_logits[best] - query_logits[second]),
                "query_margin_definition": "softmax_across_same_role_candidates",
                "query_margin_pass": probability_margin >= MIN_QUERY_MARGIN,
            }

        candidate_results: dict[str, dict[str, Any]] = {}
        for candidate_index, candidate_id in enumerate(candidate_ids):
            candidate_logits = logits[candidate_index, :]
            order = np.argsort(candidate_logits)[::-1]
            best, second = int(order[0]), int(order[1])
            assignment_margin = float(candidate_logits[best] - candidate_logits[second])
            candidate_results[candidate_id] = {
                "candidate_id": candidate_id,
                "selected_class": class_names[best],
                "selected_logit": float(candidate_logits[best]),
                "second_class": class_names[second],
                "second_logit": float(candidate_logits[second]),
                "role_assignment_margin": assignment_margin,
                "assignment_margin_pass": assignment_margin >= MIN_ASSIGNMENT_MARGIN,
            }

        checks = []
        for class_name in class_names:
            query = class_results[class_name]
            assignment = candidate_results[query["selected_candidate_id"]]
            mutual_top1 = assignment["selected_class"] == class_name
            checks.append(
                {
                    "class_name": class_name,
                    "selected_candidate_id": query["selected_candidate_id"],
                    "mutual_top1": mutual_top1,
                    "query_probability_margin": query["probability_margin"],
                    "role_assignment_margin": assignment["role_assignment_margin"],
                    "pass": bool(
                        mutual_top1
                        and query["query_margin_pass"]
                        and assignment["assignment_margin_pass"]
                    ),
                }
            )

        bijection_pass = len(
            {result["selected_candidate_id"] for result in class_results.values()}
        ) == len(class_names)

        return {
            "role": role,
            "candidate_semantic_names_received": False,
            "candidate_ids": candidate_ids,
            "class_names": class_names,
            "prompt_records": prompt_records,
            "aggregated_logits": logits.tolist(),
            "class_results": class_results,
            "candidate_results": candidate_results,
            "mutual_top1_checks": checks,
            "bijection_pass": bijection_pass,
            "all_pass": bool(bijection_pass and all(check["pass"] for check in checks)),
            "crop_trace": crop_trace,
        }


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ShapeSiglipServiceError("CUDA no está disponible")
    if requested not in {"cuda", "cpu"}:
        raise ShapeSiglipServiceError(f"Dispositivo no soportado: {requested!r}")
    return requested


class Application:
    def __init__(self, device: str) -> None:
        self.device = device
        self.model = ShapeSiglipModel(device)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "schema": "VLA_shape_siglip_health",
            "model_id": MODEL_ID,
            "contract": CONTRACT,
            "device": self.device,
            "minimum_query_probability_margin": MIN_QUERY_MARGIN,
            "minimum_role_assignment_margin": MIN_ASSIGNMENT_MARGIN,
            "query_margin_definition": "softmax_across_same_role_candidates",
            "candidate_semantic_names_received": False,
        }

    def ground(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ShapeSiglipServiceError("Payload no es un objeto JSON")
        if str(payload.get("contract", "")).strip() != CONTRACT:
            raise ShapeSiglipServiceError("Contrato solicitado no válido")

        requested_object = str(payload.get("requested_object_label", "")).strip()
        requested_target = str(payload.get("requested_target_label", "")).strip()
        if requested_object not in OBJECT_PROMPTS:
            raise ShapeSiglipServiceError(f"Objeto no soportado: {requested_object!r}")
        if requested_target not in TARGET_PROMPTS:
            raise ShapeSiglipServiceError(f"Destino no soportado: {requested_target!r}")

        image, image_sha256 = decode_image(payload)
        roles = payload.get("roles")
        if not isinstance(roles, dict):
            raise ShapeSiglipServiceError("Falta roles")
        object_payload, target_payload = roles.get("object"), roles.get("target")
        if not isinstance(object_payload, dict) or not isinstance(target_payload, dict):
            raise ShapeSiglipServiceError("Roles object/target inválidos")

        object_result = self.model.score_role(
            image=image,
            role="object",
            candidates=object_payload.get("candidates"),
            prompts_by_class=OBJECT_PROMPTS,
        )
        target_result = self.model.score_role(
            image=image,
            role="target",
            candidates=target_payload.get("candidates"),
            prompts_by_class=TARGET_PROMPTS,
        )

        object_query = object_result["class_results"][requested_object]
        target_query = target_result["class_results"][requested_target]
        object_id = object_query["selected_candidate_id"]
        target_id = target_query["selected_candidate_id"]
        object_result["requested_query"] = object_query
        target_result["requested_query"] = target_query
        object_result["assignment_margin"] = object_result["candidate_results"][object_id][
            "role_assignment_margin"
        ]
        target_result["assignment_margin"] = target_result["candidate_results"][target_id][
            "role_assignment_margin"
        ]
        all_pass = bool(object_result["all_pass"] and target_result["all_pass"])

        result = {
            "ok": all_pass,
            "all_pass": all_pass,
            "schema": "VLA_shape_siglip_grounding_response",
            "contract": CONTRACT,
            "model_id": MODEL_ID,
            "device": self.device,
            "image_sha256": image_sha256,
            "requested_object_label": requested_object,
            "requested_target_label": requested_target,
            "roles": {"object": object_result, "target": target_result},
            "selected_object_candidate_id": object_id,
            "selected_target_candidate_id": target_id,
            "candidate_semantic_names_received": False,
            "minimum_query_probability_margin": MIN_QUERY_MARGIN,
            "minimum_role_assignment_margin": MIN_ASSIGNMENT_MARGIN,
            "query_margin_definition": "softmax_across_same_role_candidates",
        }

        run_dir = RESULTS_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "shape_siglip_service_response.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not all_pass:
            raise ShapeSiglipServiceError(f"Grounding FAIL; evidencia={run_dir}")
        return result


APPLICATION: Application | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "ShapeSigLIP/1.0"

    def write_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.write_json(404, {"ok": False, "error": "not_found"})
        elif APPLICATION is None:
            self.write_json(503, {"ok": False, "error": "service_not_ready"})
        else:
            self.write_json(200, APPLICATION.health())

    def do_POST(self) -> None:
        if self.path != "/ground":
            self.write_json(404, {"ok": False, "error": "not_found"})
            return
        if APPLICATION is None:
            self.write_json(503, {"ok": False, "error": "service_not_ready"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 80 * 1024 * 1024:
                raise ShapeSiglipServiceError("Content-Length inválido")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.write_json(200, APPLICATION.ground(payload))
        except ShapeSiglipServiceError as error:
            self.write_json(
                422,
                {"ok": False, "schema": "VLA_shape_siglip_error", "error": str(error), "contract": CONTRACT},
            )
        except Exception as error:
            self.write_json(
                500,
                {
                    "ok": False,
                    "schema": "VLA_shape_siglip_error",
                    "error": f"{type(error).__name__}: {error}",
                    "contract": CONTRACT,
                },
            )

    def log_message(self, format: str, *args: Any) -> None:
        print(format % args, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    global APPLICATION
    APPLICATION = Application(resolve_device(args.device))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    health = APPLICATION.health()
    print(f"model={health['model_id']}", flush=True)
    print(f"contract={health['contract']}", flush=True)
    print(f"device={health['device']}", flush=True)
    print(f"url=http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
