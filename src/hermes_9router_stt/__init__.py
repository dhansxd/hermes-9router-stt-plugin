"""9Router speech-to-text backend for Hermes Agent.

Transcription via a 9Router gateway (``POST {base_url}/v1/audio/transcriptions``,
multipart ``file`` + ``model``, response ``{"text": ...}``).
Model selection: ``model`` arg → ``stt.9router.model`` → ``NINEROUTER_STT_MODEL``
→ first model from the live ``/v1/models/stt`` catalog (cached, TTL 300s).

Config (``stt.9router`` in config.yaml, non-secret):
    base_url        — gateway URL (default: ``NINEROUTER_URL`` env or ``http://localhost:20128``)
    model           — default model id
    fallback_models — tried in order, ONLY on 429/quota errors of the previous model
    timeout         — per-request timeout seconds (default 180)
    language        — default BCP-47 hint (default: none; gemini may translate when set)

Secret: ``NINEROUTER_KEY`` (.env). Auth-free local gateways work without it.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.transcription_provider import TranscriptionProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:20128"
DEFAULT_TIMEOUT = 180
_MODELS_CACHE_TTL = 300.0
# ponytail: multipart via requests; Hermes runtime always has it.
_MIME = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
         ".webm": "audio/webm", ".ogg": "audio/ogg", ".flac": "audio/flac"}


def _load_settings() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
    except ImportError:  # tests / non-Hermes context: env + defaults only
        return {}
    cfg = load_config() or {}
    scoped = cfg.get("stt") if isinstance(cfg, dict) else None
    scoped = scoped.get("9router") if isinstance(scoped, dict) else None
    return scoped if isinstance(scoped, dict) else {}


def _is_quota_error(body: Any) -> bool:
    """429/quota-style failure worth retrying on another model (user-chosen policy)."""
    if not isinstance(body, dict):
        return False
    err = body.get("error")
    if isinstance(err, dict):
        msg = str(err.get("message") or "").lower()
        code = str(err.get("code") or "").lower()
        return ("429" in msg or "quota" in msg or "exceeded your current quota" in msg
                or code in ("429", "rate_limit_exceeded", "insufficient_quota"))
    return False


def _base_url(settings: Dict[str, Any]) -> str:
    url = (
        str(settings.get("base_url") or "").strip()
        or os.environ.get("NINEROUTER_URL", "").strip()
        or DEFAULT_BASE_URL
    )
    return url.rstrip("/")


def _api_key() -> str:
    # ponytail: get_secret is the scoped path; os.environ is the single-profile fallback.
    try:
        from agent.secret_scope import get_secret

        key = get_secret("NINEROUTER_KEY")
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("NINEROUTER_KEY", "").strip()


class NineRouterTranscriptionProvider(TranscriptionProvider):
    """Speech-to-text through a 9Router gateway (OpenAI Whisper-compatible API)."""

    def __init__(self) -> None:
        self._models_cache: Optional[Tuple[List[str], float]] = None

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
        model = str(settings.get("model") or "").strip()
        if model:
            return model
        env_model = os.environ.get("NINEROUTER_STT_MODEL", "").strip()
        if env_model:
            return env_model
        models = self._fetch_models()
        return models[0] if models else None

    def _model_chain(self, model_arg: Optional[str]) -> List[str]:
        """Primary → fallback_models (deduped, order preserved). Fallbacks fire on 429/quota only."""
        settings = _load_settings()
        chain: List[str] = []
        for candidate in (
            model_arg,
            str(settings.get("model") or "").strip(),
            os.environ.get("NINEROUTER_STT_MODEL", "").strip(),
        ):
            if candidate and candidate not in chain:
                chain.append(candidate)
        raw = settings.get("fallback_models")
        if isinstance(raw, str):
            raw = [raw]
        for candidate in raw or []:
            candidate = str(candidate).strip()
            if candidate and candidate not in chain:
                chain.append(candidate)
        if not chain:
            models = self._fetch_models()
            if models:
                chain.append(models[0])
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

    def _fetch_models(self) -> List[str]:
        if self._models_cache is not None and time.monotonic() - self._models_cache[1] < _MODELS_CACHE_TTL:
            return self._models_cache[0]
        settings = _load_settings()
        base = _base_url(settings)
        headers = {"Authorization": f"Bearer {_api_key()}"} if _api_key() else {}
        try:
            import requests

            response = requests.get(f"{base}/v1/models/stt", headers=headers, timeout=15)
            body = response.json()
            models = [
                str(entry.get("id"))
                for entry in (body.get("data") or [])
                if isinstance(entry, dict) and entry.get("id")
            ]
            self._models_cache = (models, time.monotonic())
            return models
        except Exception as exc:  # noqa: BLE001 — catalog is optional; transcribe() reports real errors
            logger.debug("9Router STT model catalog unavailable: %s", exc)
            return []

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Transcribe ``file_path`` via 9Router; envelope: {success, transcript, provider, error?}."""
        chain = self._model_chain(model)
        if not chain:
            return {"success": False, "transcript": "", "provider": self.name,
                    "error": "No STT model available on 9Router (catalog empty and no model configured)"}

        settings = _load_settings()
        base = _base_url(settings)
        timeout = int(settings.get("timeout") or DEFAULT_TIMEOUT)
        # ponytail: gemini STT may TRANSLATE when language is forced; only send when
        # explicitly configured — auto-detect by default (audit finding 2026-09-21).
        language = language or str(settings.get("language") or "").strip() or None

        headers = {}
        api_key = _api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        last_error = ""
        for model_id in chain:
            try:
                result = self._post_transcription(base, file_path, model_id, language, headers, timeout)
            except Exception as exc:  # noqa: BLE001 — provider must not raise
                return {"success": False, "transcript": "", "provider": self.name,
                        "model": model_id, "error": f"9Router request failed: {exc}"}
            if result.get("success"):
                return result
            last_error = str(result.get("error") or "")
            if _is_quota_error(result.get("_body")):
                logger.warning("9Router STT model %s hit 429/quota; falling back to next model in chain", model_id)
                continue
            return result  # hard error: no fallback (user policy)
        return {"success": False, "transcript": "", "provider": self.name,
                "error": last_error or "All 9Router STT models failed"}

    def _post_transcription(
        self, base: str, file_path: str, model_id: str,
        language: Optional[str], headers: Dict[str, str], timeout: int,
    ) -> Dict[str, Any]:
        import requests

        name = os.path.basename(file_path)
        ext = os.path.splitext(name)[1].lower()
        mime = _MIME.get(ext, "application/octet-stream")
        data = {"model": model_id, "response_format": "json"}
        if language:
            data["language"] = language
        with open(file_path, "rb") as fh:
            files = {"file": (name, fh, mime)}
            response = requests.post(
                f"{base}/v1/audio/transcriptions", headers=headers,
                files=files, data=data, timeout=timeout,
            )
        try:
            body = response.json()
        except ValueError:
            return {"success": False, "transcript": "", "provider": self.name,
                    "model": model_id, "error": f"HTTP {response.status_code}: non-JSON response"}
        err = body.get("error") if isinstance(body, dict) else None
        if err is not None or response.status_code >= 400:
            message = str(err.get("message") or err) if isinstance(err, dict) else f"HTTP {response.status_code}"
            return {"success": False, "transcript": "", "provider": self.name,
                    "model": model_id, "error": message, "_body": body}
        text = str(body.get("text") or "").strip()
        if not text:
            return {"success": False, "transcript": "", "provider": self.name,
                    "model": model_id, "error": "9Router returned empty transcription"}
        return {"success": True, "transcript": text, "provider": self.name, "model": model_id}


def register(ctx) -> None:  # pragma: no cover — wired by Hermes plugin discovery
    ctx.register_transcription_provider(NineRouterTranscriptionProvider())
