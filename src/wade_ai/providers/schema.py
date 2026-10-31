"""Vendor-neutral strict JSON schema for model review output.

Shared by all providers that use JSON-schema structured outputs so every
reviewer is held to the same contract.
"""

import copy
from typing import Any

from wade_ai.core.models import ModelReviewResponse

# Gateway-side metadata that the model must never generate.
_INTERNAL_FIELDS = ("raw_model_name",)


def make_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform a Pydantic JSON schema to meet strict structured-output requirements.

    Ensures:
    1. Every object schema has additionalProperties: false.
    2. Every property in properties is explicitly listed in required.
    3. Internal gateway fields (e.g. raw_model_name) are excluded.
    Nullable fields keep their anyOf [..., {"type": "null"}] form.
    """
    s = copy.deepcopy(schema)

    for field in _INTERNAL_FIELDS:
        s.get("properties", {}).pop(field, None)
        if field in s.get("required", []):
            s["required"].remove(field)

    def _recurse(node: Any) -> None:
        if not isinstance(node, dict):
            return

        if "properties" in node:
            node["additionalProperties"] = False
            node["required"] = sorted(node["properties"].keys())
            for prop in node["properties"].values():
                _recurse(prop)

        if "items" in node:
            _recurse(node["items"])

        if "$defs" in node:
            for def_schema in node["$defs"].values():
                _recurse(def_schema)

        for key in ("anyOf", "allOf", "oneOf"):
            if isinstance(node.get(key), list):
                for sub in node[key]:
                    _recurse(sub)

    _recurse(s)
    return s


def review_response_schema() -> dict[str, Any]:
    """Return the strict, model-facing schema for ModelReviewResponse."""
    return make_strict_json_schema(ModelReviewResponse.model_json_schema())
