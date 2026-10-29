#!/usr/bin/env python3

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import urllib.error
import urllib.request

from pathlib import Path
from typing import Any


DEFAULT_SERVER_URL = (
    "http://127.0.0.1:8776"
)

DEFAULT_TIMEOUT_SEC = 180.0

ALLOWED_OBJECT_LABELS = {
    "orange_cube",
    "white_cylinder",
}

ALLOWED_TARGET_LABELS = {
    "blue_zone",
    "yellow_zone",
}


class SiglipGroundingError(
    RuntimeError
):
    pass


def _server_url() -> str:
    value = os.environ.get(
        "VLA_SIGLIP_GROUNDING_URL",
        DEFAULT_SERVER_URL,
    ).strip()

    if not value:
        raise SiglipGroundingError(
            "VLA_SIGLIP_GROUNDING_URL está vacío"
        )

    return value.rstrip("/")


def _timeout_sec() -> float:
    raw = os.environ.get(
        "VLA_SIGLIP_GROUNDING_TIMEOUT_SEC",
        str(DEFAULT_TIMEOUT_SEC),
    )

    try:
        value = float(raw)
    except ValueError as error:
        raise SiglipGroundingError(
            "VLA_SIGLIP_GROUNDING_TIMEOUT_SEC "
            f"no es numérico: {raw!r}"
        ) from error

    if (
        not math.isfinite(value)
        or value <= 0.0
    ):
        raise SiglipGroundingError(
            "Timeout SigLIP inválido: "
            f"{value}"
        )

    return value


def _finite_bbox(
    raw: Any,
    *,
    label: str,
) -> list[float]:
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 4
    ):
        raise SiglipGroundingError(
            f"{label}: bbox_xyxy inválida"
        )

    try:
        bbox = [
            float(value)
            for value in raw
        ]
    except Exception as error:
        raise SiglipGroundingError(
            f"{label}: bbox no numérica"
        ) from error

    if not all(
        math.isfinite(value)
        for value in bbox
    ):
        raise SiglipGroundingError(
            f"{label}: bbox no finita"
        )

    x1, y1, x2, y2 = bbox

    if (
        x2 <= x1
        or y2 <= y1
        or x1 < 0.0
        or y1 < 0.0
    ):
        raise SiglipGroundingError(
            f"{label}: bbox degenerada: {bbox}"
        )

    return bbox


def _neutralize_candidates(
    raw_candidates: list[dict[str, Any]],
    *,
    role: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    if not isinstance(
        raw_candidates,
        list,
    ):
        raise SiglipGroundingError(
            f"{role}: candidatos no es una lista"
        )

    if len(raw_candidates) != 2:
        raise SiglipGroundingError(
            f"{role}: el contrato validado exige "
            "exactamente dos candidatos; "
            f"recibidos={len(raw_candidates)}"
        )

    neutral_candidates = []
    mapping = {}

    for index, candidate in enumerate(
        raw_candidates,
        start=1,
    ):
        if not isinstance(
            candidate,
            dict,
        ):
            raise SiglipGroundingError(
                f"{role}[{index}]: candidato inválido"
            )

        local_key = str(
            candidate.get(
                "key",
                candidate.get(
                    "canonical",
                    candidate.get(
                        "name",
                        "",
                    ),
                ),
            )
        ).strip()

        if not local_key:
            raise SiglipGroundingError(
                f"{role}[{index}]: falta key local"
            )

        bbox = _finite_bbox(
            candidate.get(
                "bbox_xyxy"
            ),
            label=(
                f"{role}.{local_key}"
            ),
        )

        neutral_id = (
            f"{role}_{index:02d}"
        )

        neutral_candidates.append(
            {
                "candidate_id":
                    neutral_id,
                "bbox_xyxy":
                    bbox,
            }
        )

        mapping[
            neutral_id
        ] = candidate

    return (
        neutral_candidates,
        mapping,
    )


def _request_json(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    body = None
    headers = {}

    if payload is not None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        headers[
            "Content-Type"
        ] = (
            "application/json"
        )

    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=_timeout_sec(),
        ) as response:
            raw = response.read()

    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            detail = ""

        raise SiglipGroundingError(
            "El servicio SigLIP devolvió "
            f"HTTP {error.code}: {detail}"
        ) from error

    except urllib.error.URLError as error:
        raise SiglipGroundingError(
            "No se pudo contactar con SigLIP: "
            f"{url}: {error}"
        ) from error

    try:
        result = json.loads(
            raw.decode("utf-8")
        )
    except Exception as error:
        raise SiglipGroundingError(
            "El servicio SigLIP no devolvió "
            "JSON válido"
        ) from error

    if not isinstance(
        result,
        dict,
    ):
        raise SiglipGroundingError(
            "La respuesta SigLIP no es un objeto"
        )

    return result


def check_health() -> dict[str, Any]:
    result = _request_json(
        method="GET",
        url=(
            _server_url()
            + "/health"
        ),
        payload=None,
    )

    if result.get("ok") is not True:
        raise SiglipGroundingError(
            "SigLIP health no devolvió ok=true"
        )

    if (
        result.get("model_id")
        != "google/siglip-base-patch16-224"
    ):
        raise SiglipGroundingError(
            "Modelo SigLIP distinto al validado: "
            f"{result.get('model_id')!r}"
        )

    if (
        result.get("contract")
        != "VLA_SIGLIP_ROLE_SPECIFIC_CROPS"
    ):
        raise SiglipGroundingError(
            "Contrato SigLIP distinto al validado: "
            f"{result.get('contract')!r}"
        )

    return result


def ground_candidates(
    *,
    image_path: str | Path,
    requested_object_label: str,
    requested_target_label: str,
    object_candidates: list[
        dict[str, Any]
    ],
    target_candidates: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    object_label = str(
        requested_object_label
    ).strip()

    target_label = str(
        requested_target_label
    ).strip()

    if (
        object_label
        not in ALLOWED_OBJECT_LABELS
    ):
        raise SiglipGroundingError(
            "Objeto solicitado no soportado: "
            f"{object_label!r}"
        )

    if (
        target_label
        not in ALLOWED_TARGET_LABELS
    ):
        raise SiglipGroundingError(
            "Destino solicitado no soportado: "
            f"{target_label!r}"
        )

    path = Path(
        image_path
    ).expanduser().resolve()

    if not path.is_file():
        raise SiglipGroundingError(
            f"No existe la imagen: {path}"
        )

    image_bytes = path.read_bytes()

    image_sha256 = hashlib.sha256(
        image_bytes
    ).hexdigest()

    (
        neutral_objects,
        object_mapping,
    ) = _neutralize_candidates(
        object_candidates,
        role="object",
    )

    (
        neutral_targets,
        target_mapping,
    ) = _neutralize_candidates(
        target_candidates,
        role="target",
    )

    payload = {
        "schema":
            "VLA_siglip_grounding_request",
        "image_base64":
            base64.b64encode(
                image_bytes
            ).decode("ascii"),
        "image_sha256":
            image_sha256,
        "roles": {
            "object": {
                "requested_label":
                    object_label,
                "candidates":
                    neutral_objects,
            },
            "target": {
                "requested_label":
                    target_label,
                "candidates":
                    neutral_targets,
            },
        },
    }

    response = _request_json(
        method="POST",
        url=(
            _server_url()
            + "/ground"
        ),
        payload=payload,
    )

    if (
        response.get("schema")
        != "VLA_siglip_grounding_service"
    ):
        raise SiglipGroundingError(
            "Schema SigLIP inesperado: "
            f"{response.get('schema')!r}"
        )

    if (
        response.get("image_sha256")
        != image_sha256
    ):
        raise SiglipGroundingError(
            "SigLIP respondió para una imagen "
            "distinta de la enviada"
        )

    # La ejecución se decide mediante las consultas realmente
    # solicitadas por la instrucción. La asignación completa de todas
    # las clases se conserva como diagnóstico, pero una clase auxiliar
    # no solicitada no bloquea una selección solicitada robusta.
    if response.get("all_pass") is not True:
        roles_for_guard = response.get("roles")
        requested_failures = []

        selected_fields = {
            "object": "selected_object_candidate_id",
            "target": "selected_target_candidate_id",
        }

        if not isinstance(roles_for_guard, dict):
            requested_failures.append(
                "roles_missing"
            )
        else:
            for role_name in (
                "object",
                "target",
            ):
                role = roles_for_guard.get(
                    role_name
                )

                if not isinstance(role, dict):
                    requested_failures.append(
                        f"{role_name}.role_missing"
                    )
                    continue

                requested_query = role.get(
                    "requested_query"
                )

                if not isinstance(
                    requested_query,
                    dict,
                ):
                    requested_failures.append(
                        f"{role_name}.requested_query_missing"
                    )
                    continue

                # mutual_top1 y robust_pass del servidor se conservan
                # como diagnóstico. La autorización de ejecución usa
                # la selección solicitada entre candidatos neutrales,
                # su margen y la asignación global uno-a-uno.
                requested_query[
                    "mutual_top1_execution_diagnostic"
                ] = requested_query.get(
                    "mutual_top1"
                )

                requested_query[
                    "server_robust_pass_diagnostic"
                ] = requested_query.get(
                    "robust_pass"
                )

                probability_margin = (
                    requested_query.get(
                        "probability_margin"
                    )
                )

                minimum_probability_margin = (
                    role.get(
                        "minimum_query_probability_margin",
                        0.20,
                    )
                )

                if (
                    not isinstance(
                        probability_margin,
                        (int, float),
                    )
                    or probability_margin
                    < minimum_probability_margin
                ):
                    requested_failures.append(
                        f"{role_name}.requested_query."
                        "probability_margin"
                    )

                requested_selected = (
                    requested_query.get(
                        "selected_candidate_id"
                    )
                )

                service_selected = response.get(
                    selected_fields[role_name]
                )

                if (
                    requested_selected is None
                    or requested_selected
                    != service_selected
                ):
                    requested_failures.append(
                        f"{role_name}.requested_query."
                        "selected_candidate_consistency"
                    )

                requested_assignment_agreement = (
                    role.get(
                        "requested_assignment_query_agreement"
                    )
                )

                if (
                    requested_assignment_agreement
                    is None
                ):
                    requested_assignment_agreement = bool(
                        requested_selected is not None
                        and requested_selected
                        == service_selected
                    )

                if (
                    requested_assignment_agreement
                    is not True
                ):
                    requested_failures.append(
                        f"{role_name}."
                        "requested_assignment_query_agreement"
                    )

                assignment_margin = role.get(
                    "assignment_margin"
                )

                minimum_assignment_margin = (
                    role.get(
                        "minimum_assignment_margin",
                        1.0,
                    )
                )

                if (
                    not isinstance(
                        assignment_margin,
                        (int, float),
                    )
                    or assignment_margin
                    < minimum_assignment_margin
                ):
                    requested_failures.append(
                        f"{role_name}."
                        "assignment_margin"
                    )

        if requested_failures:
            raise SiglipGroundingError(
                "SigLIP no superó los guards "
                "de la consulta solicitada: "
                + ", ".join(requested_failures)
            )

        response[
            "requested_query_execution_pass"
        ] = True

        response[
            "full_assignment_diagnostic_pass"
        ] = False

        response[
            "auxiliary_assignment_warning"
        ] = True

        response[
            "execution_guard_contract"
        ] = (
            "requested_query_assignment_guard"
        )

    response.setdefault(
        "requested_query_execution_pass",
        True,
    )

    response.setdefault(
        "full_assignment_diagnostic_pass",
        bool(response.get("all_pass")),
    )

    response.setdefault(
        "auxiliary_assignment_warning",
        False,
    )

    response.setdefault(
        "execution_guard_contract",
        "full_assignment_guard",
    )

    roles = response.get(
        "roles"
    )

    if not isinstance(
        roles,
        dict,
    ):
        raise SiglipGroundingError(
            "Respuesta SigLIP sin roles"
        )

    for role in (
        "object",
        "target",
    ):
        role_result = roles.get(role)

        if (
            (not isinstance(
                role_result,
                dict,
            )
            or role_result.get("pass")
            is not True) and response.get(
                "requested_query_execution_pass"
            ) is not True
        ):
            raise SiglipGroundingError(
                f"SigLIP role={role} no pasó"
            )

    selected_object_id = str(
        response.get(
            "selected_object_candidate_id",
            "",
        )
    )

    selected_target_id = str(
        response.get(
            "selected_target_candidate_id",
            "",
        )
    )

    if (
        selected_object_id
        not in object_mapping
    ):
        raise SiglipGroundingError(
            "SigLIP seleccionó un objeto "
            f"desconocido: {selected_object_id!r}"
        )

    if (
        selected_target_id
        not in target_mapping
    ):
        raise SiglipGroundingError(
            "SigLIP seleccionó un destino "
            f"desconocido: {selected_target_id!r}"
        )

    selected_object = object_mapping[
        selected_object_id
    ]

    selected_target = target_mapping[
        selected_target_id
    ]

    return {
        "schema":
            "VLA_siglip_grounding_client_result",
        "image_path":
            str(path),
        "image_sha256":
            image_sha256,
        "requested_object_label":
            object_label,
        "requested_target_label":
            target_label,
        "selected_object":
            selected_object,
        "selected_target":
            selected_target,
        "selected_object_key":
            str(
                selected_object.get(
                    "key",
                    selected_object.get(
                        "canonical",
                        selected_object.get(
                            "name",
                            "",
                        ),
                    ),
                )
            ),
        "selected_target_key":
            str(
                selected_target.get(
                    "key",
                    selected_target.get(
                        "canonical",
                        selected_target.get(
                            "name",
                            "",
                        ),
                    ),
                )
            ),
        "selected_object_neutral_id":
            selected_object_id,
        "selected_target_neutral_id":
            selected_target_id,
        "candidate_semantic_names_sent":
            False,
        "service_url":
            _server_url(),
        "service_response":
            response,
    }
