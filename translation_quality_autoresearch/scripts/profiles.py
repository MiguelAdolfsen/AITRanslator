from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = REPO_ROOT / "translation_quality_autoresearch" / "profiles"
DEFAULT_MODEL_PATH = ".models/qwen/Qwen3.5-9B-Q8_0.gguf"
DEFAULT_PROJECTOR_PATH = ".models/qwen/mmproj-Qwen3.5-9B-BF16.gguf"
CONTROLLED_PROFILE_KEYS = {
    "schema_version",
    "name",
    "description",
    "vision_facts_enabled",
    "artifact_mode",
    "crop_mode",
    "chunk_size",
    "validation_policy_version",
    "context_render_format",
    "enabled_fields",
    "prompt",
    "models",
}


def load_profile(path_or_name: str | Path | None) -> dict[str, Any]:
    path = resolve_profile_path(path_or_name or "best")
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["_profile_path"] = str(path)
    validate_profile(profile)
    apply_profile_defaults(profile)
    return profile


def resolve_profile_path(path_or_name: str | Path) -> Path:
    value = Path(path_or_name)
    if value.exists():
        return value
    if value.suffix == "":
        named = PROFILE_DIR / f"{value}.json"
        if named.exists():
            return named
    raise FileNotFoundError(f"profile not found: {path_or_name}")


def validate_profile(profile: dict[str, Any]) -> None:
    unknown = sorted(set(profile) - CONTROLLED_PROFILE_KEYS - {"_profile_path"})
    if unknown:
        raise ValueError(f"profile contains uncontrolled keys: {unknown}")
    if int(profile.get("schema_version", 0) or 0) != 1:
        raise ValueError("profile.schema_version must be 1")
    if not str(profile.get("name", "")).strip():
        raise ValueError("profile.name is required")
    if str(profile.get("artifact_mode", "numbered_page")) not in {"numbered_page", "page_image"}:
        raise ValueError("profile.artifact_mode must be numbered_page or page_image")
    if str(profile.get("crop_mode", "numbered_full_page")) not in {
        "numbered_full_page",
        "line_crop",
        "bubble_crop",
        "full_page_plus_crop",
    }:
        raise ValueError("profile.crop_mode is not supported")
    chunk_size = int(profile.get("chunk_size", 4) or 4)
    if chunk_size < 1 or chunk_size > 12:
        raise ValueError("profile.chunk_size must be between 1 and 12")
    enabled_fields = profile.get("enabled_fields", [])
    if not isinstance(enabled_fields, list) or not all(isinstance(value, str) for value in enabled_fields):
        raise ValueError("profile.enabled_fields must be a string list")


def apply_profile_defaults(profile: dict[str, Any]) -> None:
    profile.setdefault("vision_facts_enabled", True)
    profile.setdefault("artifact_mode", "numbered_page")
    profile.setdefault("crop_mode", "numbered_full_page")
    profile.setdefault("chunk_size", 4)
    profile.setdefault("validation_policy_version", "production_v1")
    profile.setdefault("context_render_format", "production_v1")
    profile.setdefault("enabled_fields", [])
    profile.setdefault("prompt", {"task": "", "extra_instructions": []})
    models = profile.setdefault("models", {})
    models.setdefault("qwen_judge_model", DEFAULT_MODEL_PATH)
    models.setdefault("qwen_critic_model", DEFAULT_MODEL_PATH)
    models.setdefault("qwen_fallback_model", DEFAULT_MODEL_PATH)
    models.setdefault("vision_model", DEFAULT_MODEL_PATH)
    models.setdefault("vision_projector", DEFAULT_PROJECTOR_PATH)


def profile_hash(profile: dict[str, Any]) -> str:
    payload = public_profile(profile)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: copy.deepcopy(value)
        for key, value in profile.items()
        if key in CONTROLLED_PROFILE_KEYS
    }
    apply_profile_defaults(payload)
    return payload


def write_profile(path: Path, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = public_profile(profile)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def model_path(profile: dict[str, Any], key: str) -> Path:
    models = profile.get("models") if isinstance(profile.get("models"), dict) else {}
    value = str(models.get(key) or DEFAULT_MODEL_PATH)
    return Path(value)

