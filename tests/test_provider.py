"""Tests for the 9Router STT provider. Offline: monkeypatched transport.

Run: python3 -m pytest tests/ -q   (or the stdlib runner below)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Minimal stubs so the provider module imports outside the Hermes tree.
for mod, attrs in {
    "agent": None,
    "agent.transcription_provider": {
        "TranscriptionProvider": object,
    },
}.items():
    if mod not in sys.modules:
        import types

        m = types.ModuleType(mod)
        if attrs:
            for k, v in attrs.items():
                setattr(m, k, v)
        sys.modules[mod] = m


class _Resp:
    def __init__(self, body: dict, status: int = 200):
        self._body = body
        self.status_code = status

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class _FakeRequests:
    _handler = None

    @classmethod
    def set_handler(cls, fn):
        cls._handler = fn

    @staticmethod
    def request(*a, **kw):
        raise NotImplementedError

    @staticmethod
    def get(url, **kw):
        return _FakeRequests._handler("GET", url, kw)

    @staticmethod
    def post(url, headers=None, files=None, data=None, timeout=None):
        return _FakeRequests._handler("POST", url,
                                      {"headers": headers, "files": files, "data": data, "timeout": timeout})


def _install_handler(handler):
    import requests as real

    _FakeRequests.set_handler(handler)
    sys.modules["requests"] = _FakeRequests
    return real


def _load_provider():
    for name in list(sys.modules):
        if name.startswith("hermes_9router_stt"):
            del sys.modules[name]
    import hermes_9router_stt as mod

    return mod, mod.NineRouterTranscriptionProvider()


def main() -> int:
    src_path = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src_path))

    # 1. success path
    calls = []

    def ok_handler(method, url, kw):
        calls.append((method, url, kw))
        if url.endswith("/v1/models/stt"):
            return _Resp({"data": [{"id": "gemini/gemini-3.8-flash"}, {"id": "gemini/gemini-2.5-flash"}]})
        return _Resp({"text": "nine router speech to text audit test"})

    real = _install_handler(ok_handler)
    mod, provider = _load_provider()
    result = provider.transcribe("/tmp/stt_test.wav", model="gemini/gemini-3.8-flash")
    assert result["success"] is True, result
    assert result["transcript"] == "nine router speech to text audit test"
    assert result["provider"] == "9router"
    assert result["model"] == "gemini/gemini-3.8-flash"
    post = [c for c in calls if c[0] == "POST"][0]
    assert post[1].endswith("/v1/audio/transcriptions")

    # 2. quota error -> fallback to next model (config-provided fallback chain)
    mod, provider = _load_provider()
    _settings = {"fallback_models": ["gemini/gemini-2.5-flash"]}
    mod._load_settings = lambda: _settings
    fallback_calls = []

    def quota_handler(method, url, kw):
        fallback_calls.append((method, url, kw))
        if url.endswith("/v1/models/stt"):
            return _Resp({"data": [{"id": "a"}, {"id": "b"}]})
        model_used = (kw.get("data") or {}).get("model")
        if model_used == "gemini/gemini-3.8-flash":
            body = {"error": {"message": "You exceeded your current quota", "code": 429}}
            return _Resp(body, 429)
        return _Resp({"text": "fallback transcript"})

    _install_handler(quota_handler)
    result = provider.transcribe("/tmp/stt_test.wav", model="gemini/gemini-3.8-flash")
    assert result["success"] is True, result
    assert result["transcript"] == "fallback transcript"
    assert result["model"] == "gemini/gemini-2.5-flash"
    posts = [c for c in fallback_calls if c[0] == "POST"]
    assert len(posts) == 2, posts  # primary + fallback
    del mod._load_settings

    # 3. hard error (auth) -> no fallback
    def auth_err_handler(method, url, kw):
        if url.endswith("/v1/models/stt"):
            return _Resp({"data": [{"id": "a"}, {"id": "b"}]})
        body = {"error": {"message": "Invalid API key", "type": "authentication_error", "code": "invalid_api_key"}}
        return _Resp(body, 401)

    _install_handler(auth_err_handler)
    mod, provider = _load_provider()
    result = provider.transcribe("/tmp/stt_test.wav", model="a")
    assert result["success"] is False, result
    assert "Invalid API key" in result["error"]

    # 4. empty prompt / no model maybe not applicable; test empty catalog
    def empty_catalog_handler(method, url, kw):
        return _Resp({"data": []})

    _install_handler(empty_catalog_handler)
    mod, provider = _load_provider()
    result = provider.transcribe("/tmp/stt_test.wav")
    assert result["success"] is False, result
    assert "No STT model" in result["error"]

    # 5. empty transcription
    def empty_text_handler(method, url, kw):
        if url.endswith("/v1/models/stt"):
            return _Resp({"data": [{"id": "a"}]})
        return _Resp({"text": ""})

    _install_handler(empty_text_handler)
    mod, provider = _load_provider()
    result = provider.transcribe("/tmp/stt_test.wav", model="a")
    assert result["success"] is False, result
    assert "empty transcription" in result["error"]

    # reset
    sys.modules["requests"] = real
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())