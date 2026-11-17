"""Offline regressions using the real Hermes ABC, scopes and requests objects.

Run with the pinned Hermes interpreter:
    python tests/test_regressions.py

To prove the original defects:
    PLUGIN_SOURCE=/path/to/baseline python tests/test_regressions.py --baseline
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

import requests

SOURCE = Path(
    os.environ.get("PLUGIN_SOURCE", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, str(SOURCE / "src"))

from agent.secret_scope import (
    UnscopedSecretError,
    reset_multiplex_context,
    reset_secret_scope,
    set_multiplex_context,
    set_secret_scope,
)
from agent.transcription_provider import TranscriptionProvider
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
import hermes_9router_stt as plugin

assert issubclass(plugin.NineRouterTranscriptionProvider, TranscriptionProvider)


@contextmanager
def profile_scope(home: Path, secrets: dict[str, str]):
    home_token = set_hermes_home_override(home)
    multiplex_token = set_multiplex_context(True)
    secret_token = set_secret_scope(secrets, profile_home=str(home))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_multiplex_context(multiplex_token)
        reset_hermes_home_override(home_token)


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audio = self.root / "sample.wav"
        with wave.open(str(self.audio), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * 160)
        self.audio_bytes = self.audio.read_bytes()

        self.env_patch = patch.dict(
            os.environ,
            {
                "NINEROUTER_URL": "",
                "NINEROUTER_STT_MODEL": "",
                "NINEROUTER_KEY": "",
            },
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

        self.settings = {
            "base_url": "http://gateway.invalid",
            "model": "primary",
        }
        self.settings_patch = patch.object(
            plugin, "_load_settings", side_effect=lambda: dict(self.settings)
        )
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)

        self.provider = plugin.NineRouterTranscriptionProvider()
        self.calls = []
        self.responses = []
        self.handler = self.success_handler

        def request(session, method, url, **kwargs):
            # Perform real requests authentication preparation, without sending.
            prepared = session.prepare_request(
                requests.Request(method, url, headers=kwargs.get("headers"))
            )
            call = {
                "method": method.upper(),
                "url": url,
                "kwargs": kwargs,
                "prepared_headers": dict(prepared.headers),
            }
            if kwargs.get("files"):
                stream = kwargs["files"]["file"][1]
                call["stream"] = stream
                call["audio_bytes"] = stream.read()
            self.calls.append(call)

            body, status = self.handler(call)
            response = requests.Response()
            response.status_code = status
            response.url = url
            response.request = prepared
            response.encoding = "utf-8"
            response._content = (
                body if isinstance(body, bytes)
                else json.dumps(body).encode("utf-8")
            )
            response._content_consumed = True
            response.close = Mock(wraps=response.close)
            self.responses.append(response)
            return response

        self.network_patch = patch(
            "requests.sessions.Session.request", new=request
        )
        self.network_patch.start()
        self.addCleanup(self.network_patch.stop)

    @staticmethod
    def success_handler(call):
        if call["method"] == "GET":
            return {"data": [{"id": "catalog-primary"}]}, 200
        return {"text": "offline transcript"}, 200

    def posts(self):
        return [call for call in self.calls if call["method"] == "POST"]

    def test_primary_precedence(self):
        os.environ["NINEROUTER_STT_MODEL"] = "environment"
        self.settings["fallback_models"] = ["fallback", "fallback"]
        self.assertEqual(
            self.provider._model_chain("explicit"),
            ["explicit", "fallback"],
        )
        self.assertEqual(
            self.provider._model_chain(None),
            ["primary", "fallback"],
        )
        self.settings.pop("model")
        self.assertEqual(
            self.provider._model_chain(None),
            ["environment", "fallback"],
        )

    def test_catalog_primary_precedes_fallbacks(self):
        self.settings.pop("model")
        self.settings["fallback_models"] = ["fallback"]
        self.assertEqual(
            self.provider._model_chain(None),
            ["catalog-primary", "fallback"],
        )

    def test_http_429_non_json_falls_back(self):
        self.settings["fallback_models"] = ["fallback"]

        def handler(call):
            if call["kwargs"].get("data", {}).get("model") == "primary":
                return b"Too many requests", 429
            return {"text": "fallback transcript"}, 200

        self.handler = handler
        result = self.provider.transcribe(str(self.audio))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["model"], "fallback")
        self.assertEqual(
            [call["kwargs"]["data"]["model"] for call in self.posts()],
            ["primary", "fallback"],
        )
        self.assertEqual(
            [call["audio_bytes"] for call in self.posts()],
            [self.audio_bytes, self.audio_bytes],
        )

    def test_hard_401_quota_message_is_not_replayed(self):
        self.settings["fallback_models"] = ["fallback"]

        def handler(call):
            if len(self.posts()) == 1:
                return {"error": {"message": "quota access requires login"}}, 401
            return {"text": "should not be requested"}, 200

        self.handler = handler
        result = self.provider.transcribe(str(self.audio))
        self.assertFalse(result["success"], result)
        self.assertEqual(len(self.posts()), 1)

    def test_hard_403_insufficient_quota_code_is_not_replayed(self):
        self.settings["fallback_models"] = ["fallback"]

        def handler(call):
            if len(self.posts()) == 1:
                return {
                    "error": {
                        "code": "insufficient_quota",
                        "message": 'This API key is not allowed to use model "fixture/quota-model"',
                    }
                }, 403
            return {"text": "should not be requested"}, 200

        self.handler = handler
        result = self.provider.transcribe(str(self.audio))
        self.assertFalse(result["success"], result)
        self.assertEqual(len(self.posts()), 1)

    def test_missing_scoped_key_does_not_use_process_key(self):
        os.environ["NINEROUTER_KEY"] = "synthetic-launch-key"
        with profile_scope(self.root / "profile", {}):
            self.assertEqual(plugin._api_key(), "")

    def test_empty_scoped_key_does_not_use_process_key(self):
        os.environ["NINEROUTER_KEY"] = "synthetic-launch-key"
        with profile_scope(
            self.root / "profile", {"NINEROUTER_KEY": ""}
        ):
            self.assertEqual(plugin._api_key(), "")

    def test_unscoped_multiplex_key_fails_closed(self):
        os.environ["NINEROUTER_KEY"] = "synthetic-launch-key"
        multiplex_token = set_multiplex_context(True)
        secret_token = set_secret_scope(None)
        try:
            with self.assertRaises(UnscopedSecretError):
                plugin._api_key()
        finally:
            reset_secret_scope(secret_token)
            reset_multiplex_context(multiplex_token)

    def test_catalog_changes_with_endpoint(self):
        def handler(call):
            model = (
                "first" if call["url"].startswith("http://first.invalid/")
                else "second"
            )
            return {"data": [{"id": model}]}, 200

        self.handler = handler
        self.settings["base_url"] = "http://first.invalid"
        self.assertEqual(self.provider._fetch_models(), ["first"])
        self.settings["base_url"] = "http://second.invalid"
        self.assertEqual(self.provider._fetch_models(), ["second"])

    def test_catalog_changes_with_key(self):
        def handler(call):
            key = call["kwargs"].get("headers", {}).get("Authorization", "")
            return {"data": [{"id": key.rsplit(" ", 1)[-1]}]}, 200

        self.handler = handler
        os.environ["NINEROUTER_KEY"] = "synthetic-first"
        self.assertEqual(self.provider._fetch_models(), ["synthetic-first"])
        os.environ["NINEROUTER_KEY"] = "synthetic-second"
        self.assertEqual(self.provider._fetch_models(), ["synthetic-second"])

    def test_catalog_isolated_between_concurrent_profiles(self):
        def handler(call):
            return {"data": [{"id": get_hermes_home().name}]}, 200

        self.handler = handler

        def read_profile(name):
            with profile_scope(
                self.root / name, {"NINEROUTER_KEY": "synthetic-shared-key"}
            ):
                return self.provider._fetch_models()

        # A warm singleton cache must not make B inherit A's catalog.
        self.assertEqual(read_profile("profile-a"), ["profile-a"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(read_profile, "profile-a")
            b = pool.submit(read_profile, "profile-b")
            self.assertEqual(a.result(timeout=5), ["profile-a"])
            self.assertEqual(b.result(timeout=5), ["profile-b"])

    def test_catalog_returns_independent_lists(self):
        models = self.provider._fetch_models()
        models.append("caller-poison")
        self.assertEqual(
            self.provider._fetch_models(), ["catalog-primary"]
        )

    def test_transcript_requires_string(self):
        for invalid_text in (["not", "text"], {"text": "nested"}, 123, None):
            self.handler = lambda call, value=invalid_text: (
                {"text": value}, 200
            )
            result = self.provider.transcribe(str(self.audio))
            self.assertFalse(result["success"], result)
            self.assertEqual(result["transcript"], "")

    def test_errors_do_not_expose_upstream_data(self):
        marker = "synthetic-PRIVATE-upstream-value"
        self.handler = lambda call: (
            {
                "error": {
                    "message": marker,
                    "code": "invalid_request",
                },
                "extra": {"credential": marker},
            },
            400,
        )
        result = self.provider.transcribe(str(self.audio))
        self.assertFalse(result["success"], result)
        self.assertNotIn(marker, json.dumps(result))
        self.assertNotIn("_body", result)

        def failing_transport(call):
            raise requests.ConnectionError(marker)

        self.handler = failing_transport
        result = self.provider.transcribe(str(self.audio))
        self.assertFalse(result["success"], result)
        self.assertNotIn(marker, json.dumps(result))

    def test_gateway_auth_and_redirect_policy(self):
        os.environ["NINEROUTER_KEY"] = "synthetic-gateway-key"
        with patch(
            "requests.sessions.get_netrc_auth",
            return_value=("synthetic-netrc-user", "synthetic-netrc-password"),
        ):
            self.assertEqual(
                self.provider._fetch_models(), ["catalog-primary"]
            )
            result = self.provider.transcribe(str(self.audio))
        self.assertTrue(result["success"], result)
        self.assertEqual(len(self.calls), 2)
        for call in self.calls:
            self.assertEqual(
                call["prepared_headers"].get("Authorization"),
                "Bearer synthetic-gateway-key",
            )
            self.assertIs(call["kwargs"].get("allow_redirects"), False)

    def test_invalid_timeout_returns_error_envelope(self):
        self.settings["timeout"] = "invalid-timeout"
        result = self.provider.transcribe(str(self.audio))
        self.assertFalse(result["success"], result)
        self.assertEqual(result["transcript"], "")

    def test_files_and_responses_are_closed(self):
        self.provider._fetch_models()
        result = self.provider.transcribe(str(self.audio))
        self.assertTrue(result["success"], result)
        self.assertTrue(self.posts()[0]["stream"].closed)
        self.assertTrue(self.responses)
        for response in self.responses:
            self.assertTrue(response.close.called)

    def test_canonical_settings_namespace(self):
        self.settings_patch.stop()
        expected = {"base_url": "http://canonical.invalid", "model": "canonical"}
        with patch(
            "hermes_cli.config.load_config",
            return_value={"stt": {"9router-stt": expected}},
        ):
            self.assertEqual(plugin._load_settings(), expected)

    # Controls: both must already pass against the original baseline.

    def test_success_and_multipart_contract(self):
        result = self.provider.transcribe(str(self.audio), language="id")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["transcript"], "offline transcript")
        self.assertEqual(result["provider"], "9router-stt")
        call = self.posts()[0]
        self.assertEqual(
            call["url"],
            "http://gateway.invalid/v1/audio/transcriptions",
        )
        self.assertEqual(
            call["kwargs"]["data"],
            {
                "model": "primary",
                "response_format": "json",
                "language": "id",
            },
        )
        self.assertEqual(call["kwargs"]["files"]["file"][0], "sample.wav")
        self.assertEqual(call["kwargs"]["files"]["file"][2], "audio/wav")
        self.assertEqual(call["audio_bytes"], self.audio_bytes)

    def test_legacy_settings_compatibility(self):
        self.settings_patch.stop()
        expected = {"base_url": "http://legacy.invalid", "model": "legacy"}
        with patch(
            "hermes_cli.config.load_config",
            return_value={"stt": {"9router": expected}},
        ):
            self.assertEqual(plugin._load_settings(), expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    names = unittest.defaultTestLoader.getTestCaseNames(RegressionTests)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RegressionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    if not args.baseline:
        return 0 if result.wasSuccessful() else 1

    controls = {
        "test_success_and_multipart_contract",
        "test_legacy_settings_compatibility",
    }
    expected = set(names) - controls
    failures = {
        test._testMethodName for test, _ in result.failures
    }
    errors = {
        test._testMethodName for test, _ in result.errors
    }
    observed = failures | errors
    allowed_errors = {"test_invalid_timeout_returns_error_envelope"}

    if (
        result.testsRun != len(names)
        or observed != expected
        or errors - allowed_errors
        or result.skipped
        or result.unexpectedSuccesses
    ):
        print("BASELINE REPRODUCTION MISMATCH")
        print("Missing expected defects:", sorted(expected - observed))
        print("Unexpected failures:", sorted(observed - expected))
        print("Unexpected setup/runtime errors:", sorted(errors - allowed_errors))
        return 1

    print(
        f"BASELINE REPRODUCED: {len(expected)} defects; "
        f"{len(controls)} controls passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
