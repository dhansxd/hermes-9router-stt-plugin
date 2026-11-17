"""Native Hermes speech-to-text backend for a 9Router gateway.

Settings: stt.9router-stt (legacy stt.9router remains supported).
Model: argument > config > scoped NINEROUTER_STT_MODEL > live catalog.
Only explicit fallback_models are retried, on HTTP 429 or a quota failure.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:20128"
DEFAULT_TIMEOUT = 180
_MODELS_CACHE_TTL = 300.0
_MIME = {
    ".mp3": "audio/mpeg", ".mpga": "audio/mpeg", ".mpeg": "audio/mpeg",
    ".wav": "audio/wav", ".m4a": "audio/mp4", ".mp4": "audio/mp4",
    ".webm": "audio/webm", ".ogg": "audio/ogg", ".oga": "audio/ogg",
    ".opus": "audio/ogg", ".flac": "audio/flac", ".aac": "audio/aac",
}


def _load_settings() -> Dict[str, Any]:
    from hermes_cli.config import load_config

    cfg = load_config() or {}
    stt = cfg.get("stt") if isinstance(cfg, dict) else None
    if not isinstance(stt, dict):
        return {}
    legacy = stt.get("9router")
    canonical = stt.get("9router-stt")
    settings = dict(legacy) if isinstance(legacy, dict) else {}
    if isinstance(canonical, dict):
        settings.update(canonical)
    return settings


def _scoped_env(name: str) -> str:
    try:
        from agent.secret_scope import get_secret
    except ImportError:  # Compatibility with hosts predating scoped secrets.
        return os.environ.get(name, "").strip()
    # A scoped miss or UnscopedSecretError must never borrow the launch profile's env.
    return str(get_secret(name) or "").strip()


def _base_url(settings: Dict[str, Any]) -> str:
    return (str(settings.get("base_url") or "").strip()
            or _scoped_env("NINEROUTER_URL") or DEFAULT_BASE_URL).rstrip("/")


def _api_key() -> str:
    return _scoped_env("NINEROUTER_KEY")


def _is_quota_error(body: Any, status: int) -> bool:
    if status == 429:
        return True
    if 300 <= status < 400 or (400 <= status < 500 and status not in (402, 403)):
        return False
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        return False
    message = str(err.get("message") or "").lower()
    code = str(err.get("code") or "").lower()
    # 9Router 0.5.119 also labels ACL-denied HTTP 403 as insufficient_quota.
    # Its code alone is therefore not evidence of quota exhaustion on 403.
    if status == 403:
        return "quota" in message or "rate limit" in message
    return ("quota" in message or "rate limit" in message
            or code in ("429", "rate_limit_exceeded", "insufficient_quota"))


def _keep_gateway_auth(request):
    """Suppress Requests' ambient netrc auth; retain proxy and CA environment support."""
    return request


def _session():
    import requests

    session = requests.Session()
    session.auth = _keep_gateway_auth
    return session


class NineRouterTranscriptionProvider(TranscriptionProvider):
    """OpenAI-compatible file transcription; no network work during discovery."""

    def __init__(self) -> None:
        self._models_cache = None
        self._models_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "9router-stt"

    @property
    def display_name(self) -> str:
        return "9Router"

    def is_available(self) -> bool:
        try:
            import requests  # noqa: F401
            return True
        except ImportError:  # pragma: no cover
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": m, "display": m.split("/")[-1]} for m in self._fetch_models()]

    def default_model(self) -> Optional[str]:
        settings = _load_settings()
        model = (str(settings.get("model") or "").strip()
                 or _scoped_env("NINEROUTER_STT_MODEL"))
        if model:
            return model
        models = self._fetch_models(settings)
        return models[0] if models else None

    def _model_chain(
        self, model_arg: Optional[str], settings: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> List[str]:
        settings = _load_settings() if settings is None else settings
        primary = (str(model_arg or "").strip()
                   or str(settings.get("model") or "").strip()
                   or _scoped_env("NINEROUTER_STT_MODEL"))
        if not primary:
            models = self._fetch_models(settings, api_key)
            primary = models[0] if models else ""
        # Fallbacks cannot silently become the primary when the catalog is unavailable.
        if not primary:
            return []
        chain = [primary]
        raw = settings.get("fallback_models") or []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            raise ValueError("fallback_models must be a list or string")
        for candidate in raw:
            candidate = str(candidate).strip()
            if candidate and candidate not in chain:
                chain.append(candidate)
        return chain

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "9Router",
            "badge": "",
            "tag": "Speech-to-text via a local or remote 9Router gateway",
            "env_vars": [
                {"key": "NINEROUTER_KEY", "prompt": "9Router API key (skip for auth-free local gateway)"},
            ],
        }

    def _fetch_models(
        self, settings: Optional[Dict[str, Any]] = None, api_key: Optional[str] = None,
    ) -> List[str]:
        try:
            from hermes_constants import get_hermes_home

            settings = _load_settings() if settings is None else settings
            base = _base_url(settings)
            api_key = _api_key() if api_key is None else api_key
            identity = (str(get_hermes_home()), base, hashlib.sha256(api_key.encode()).digest())
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            # One bounded cache entry. Lock the fetch too, so concurrent callers share it.
            with self._models_lock:
                cached = self._models_cache
                if cached is not None and cached[0] == identity and time.monotonic() - cached[2] < _MODELS_CACHE_TTL:
                    return list(cached[1])
                with _session() as session:
                    response = session.get(
                        f"{base}/v1/models/stt", headers=headers,
                        timeout=15, allow_redirects=False,
                    )
                    try:
                        if not 200 <= response.status_code < 300:
                            return []
                        body = response.json()
                        entries = body.get("data") if isinstance(body, dict) else None
                        if not isinstance(entries, list):
                            return []
                        models = list(dict.fromkeys(
                            entry["id"].strip() for entry in entries
                            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
                            and entry["id"].strip()
                        ))
                    finally:
                        response.close()
                self._models_cache = (identity, tuple(models), time.monotonic())
                return list(models)
        except Exception:  # Optional catalog: no raw URL, body or credentials in logs.
            logger.debug("9Router STT model catalog unavailable")
            return []

    def transcribe(
        self, file_path: str, *, model: Optional[str] = None,
        language: Optional[str] = None, **extra: Any,
    ) -> Dict[str, Any]:
        model_id = model
        try:
            settings = _load_settings()
            base = _base_url(settings)
            api_key = _api_key()
            timeout = float(settings.get("timeout", DEFAULT_TIMEOUT))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeout must be positive and finite")
            chain = self._model_chain(model, settings, api_key)
            if not chain:
                return self._error("No STT model available on 9Router (catalog empty and no model configured)")
            language = language or str(settings.get("language") or "").strip() or None
            prompt = extra.get("prompt")
            prompt = prompt if isinstance(prompt, str) and prompt.strip() else None
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            result = None
            for model_id in chain:
                result = self._post_transcription(
                    base, file_path, model_id, language, headers, timeout, prompt=prompt,
                )
                retryable = result.pop("_retryable", False)
                if result["success"] or not retryable:
                    return result
                logger.warning("9Router STT hit 429/quota; trying the next configured fallback")
            return result
        except Exception:  # Provider ABC requires an error envelope, including setup failures.
            return self._error(
                "9Router request failed; check gateway connectivity, profile credentials and STT settings",
                model_id,
            )

    def _error(self, message: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        result = {"success": False, "transcript": "", "provider": self.name, "error": message}
        if model_id:
            result["model"] = model_id
        return result

    def _post_transcription(
        self, base: str, file_path: str, model_id: str,
        language: Optional[str], headers: Dict[str, str], timeout: float,
        *, prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        name = os.path.basename(file_path)
        mime = _MIME.get(os.path.splitext(name)[1].lower(), "application/octet-stream")
        data = {"model": model_id, "response_format": "json"}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        with open(file_path, "rb") as fh, _session() as session:
            response = session.post(
                f"{base}/v1/audio/transcriptions", headers=headers,
                files={"file": (name, fh, mime)}, data=data,
                timeout=timeout, allow_redirects=False,
            )
            try:
                status = response.status_code
                try:
                    body = response.json()
                except ValueError:
                    result = self._error(f"HTTP {status}: non-JSON transcription response", model_id)
                    result["_retryable"] = status == 429
                    return result
                if not 200 <= status < 300 or (isinstance(body, dict) and body.get("error") is not None):
                    result = self._error(f"HTTP {status}: 9Router transcription failed", model_id)
                    result["_retryable"] = _is_quota_error(body, status)
                    return result
                text = body.get("text") if isinstance(body, dict) else None
                if not isinstance(text, str):
                    return self._error("9Router returned an invalid transcription response", model_id)
                text = text.strip()
                if not text:
                    return self._error("9Router returned empty transcription", model_id)
                return {"success": True, "transcript": text, "provider": self.name, "model": model_id}
            finally:
                response.close()


def register(ctx) -> None:  # pragma: no cover
    ctx.register_transcription_provider(NineRouterTranscriptionProvider())
