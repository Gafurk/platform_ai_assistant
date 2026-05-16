"""ServiceRegistry — single interface for loading and querying service configs.

Loads config/services.yaml and config/keywords.yaml once at startup (singleton).
All intent detection, keyword lookups, and flow-type queries go through here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FlowConfig:
    type: str                            # "linear" | "scenario" | "faq"
    max_steps: Optional[int] = None
    requires_entity: bool = False
    scenarios: list[str] = field(default_factory=list)


@dataclass
class ServiceConfig:
    service_id: str
    display_name: dict[str, str]         # {"ru": "...", "kz": "..."}
    description: dict[str, str]
    detection_patterns: dict[str, list[str]]  # {"filename": [...], "content": [...]}
    flow: FlowConfig
    navigation_path: dict[str, str]
    entities: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry (singleton)
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


class ServiceRegistry:
    """Singleton that loads services.yaml + keywords.yaml once."""

    _instance: Optional["ServiceRegistry"] = None

    def __new__(cls) -> "ServiceRegistry":
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._services: dict[str, ServiceConfig] = {}
            instance._flow_keywords: dict[str, dict[str, list[str]]] = {}
            instance._filter_keywords: dict[str, dict[str, list[str]]] = {}
            instance._load()
            cls._instance = instance
        return cls._instance

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        self._load_services()
        self._load_keywords()

    def _load_services(self) -> None:
        path = _CONFIG_DIR / "services.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for service_id, svc in (data.get("services") or {}).items():
            flow_raw = svc.get("flow") or {}
            flow = FlowConfig(
                type=flow_raw.get("type", "faq"),
                max_steps=flow_raw.get("max_steps"),
                requires_entity=flow_raw.get("requires_entity", False),
                scenarios=flow_raw.get("scenarios") or [],
            )
            self._services[service_id] = ServiceConfig(
                service_id=service_id,
                display_name=svc.get("display_name") or {},
                description=svc.get("description") or {},
                detection_patterns=svc.get("detection_patterns") or {},
                flow=flow,
                navigation_path=svc.get("navigation_path") or {},
                entities=svc.get("entities") or [],
            )

    def _load_keywords(self) -> None:
        path = _CONFIG_DIR / "keywords.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for service_id, kw in (data.get("keywords") or {}).items():
            self._flow_keywords[service_id] = kw.get("flow_keywords") or {}
            self._filter_keywords[service_id] = kw.get("filter_keywords") or {}

    # ------------------------------------------------------------------
    # Service queries
    # ------------------------------------------------------------------

    def get_service(self, service_id: str) -> Optional[ServiceConfig]:
        return self._services.get(service_id)

    def list_services(self) -> dict[str, ServiceConfig]:
        return dict(self._services)

    def get_display_name(self, service_id: str, language: str = "ru") -> str:
        svc = self._services.get(service_id)
        if not svc:
            return service_id
        return svc.display_name.get(language) or svc.display_name.get("ru") or service_id

    def get_flow_type(self, service_id: str) -> str:
        svc = self._services.get(service_id)
        return svc.flow.type if svc else "faq"

    def get_navigation_path(self, service_id: str, language: str = "ru") -> str:
        svc = self._services.get(service_id)
        if not svc:
            return ""
        return svc.navigation_path.get(language) or svc.navigation_path.get("ru") or ""

    # ------------------------------------------------------------------
    # Keyword queries
    # ------------------------------------------------------------------

    def get_flow_keywords(self, service_id: str, language: str) -> list[str]:
        """Keywords used for intent classification and FAQ-interruption gate."""
        return list(self._flow_keywords.get(service_id, {}).get(language) or [])

    def get_all_flow_keywords(self, service_id: str) -> list[str]:
        """Combined RU + KZ flow keywords for a service."""
        kw = self._flow_keywords.get(service_id) or {}
        return list(kw.get("ru") or []) + list(kw.get("kz") or [])

    def get_filter_keywords(self, service_id: str) -> list[str]:
        """Filter keywords from keywords.yaml (retained for reference/tooling)."""
        kw = self._filter_keywords.get(service_id) or {}
        return list(kw.get("ru") or []) + list(kw.get("kz") or [])

    # ------------------------------------------------------------------
    # Intent detection (replaces _slug_intent in ingestion/ingest.py)
    # ------------------------------------------------------------------

    def detect_intent(self, text: str, filename: str = "") -> Optional[str]:
        """Detect service intent from document text and/or filename.

        Checks filename patterns first (more reliable), then content patterns.
        Returns the matching service_id, or None if nothing matches.

        Space-pads the combined string so patterns like ' ту ' don't match
        inside longer words.
        """
        combined = " " + ((text or "") + " " + (filename or "")).lower() + " "

        for service_id, svc in self._services.items():
            patterns = svc.detection_patterns

            for pat in patterns.get("filename") or []:
                if pat.lower() in " " + filename.lower() + " ":
                    return service_id

            for pat in patterns.get("content") or []:
                if pat.lower() in combined:
                    return service_id

        return None

    # ------------------------------------------------------------------
    # Intent classification helpers (used by flows/registry.py)
    # ------------------------------------------------------------------

    def build_intent_keywords_map(self) -> dict[str, list[str]]:
        """Return {service_id: [all ru+kz flow keywords]} for all services.

        Used by flows/registry.py to rebuild INTENT_KEYWORDS from config
        instead of maintaining a separate hardcoded dict.
        """
        return {sid: self.get_all_flow_keywords(sid) for sid in self._services}

    # ------------------------------------------------------------------
    # Page-intent map (static, can be extended to YAML later)
    # ------------------------------------------------------------------

    _PAGE_INTENT_MAP: dict[str, list[str]] = {
        "tu_application": [
            "технических условий", "техусловия",
            "шаг 1", "шаг 2", "шаг 3", "шаг 4", "шаг 5",
        ],
        "real_estate": [
            "объект недвижимости", "объекты недвижимости",
            "добавление объекта",
        ],
    }

    def detect_intent_from_page(self, page: str) -> Optional[str]:
        lower = page.lower()
        for service_id, keywords in self._PAGE_INTENT_MAP.items():
            if any(kw in lower for kw in keywords):
                return service_id
        return None
