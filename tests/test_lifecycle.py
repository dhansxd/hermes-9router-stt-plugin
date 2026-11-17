"""Real Hermes lifecycle checks. Execute only inside a disposable cloud runner.

STT_CLOUD_TESTS=1 PLUGIN_REVISION=<40-char commit> python tests/test_lifecycle.py
The pinned official Hermes runtime must be on PYTHONPATH.
No real transcription endpoint is called.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import wave

import requests


def run(*args, ok=True, cwd=None):
    result = subprocess.run(
        [str(arg) for arg in args], cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240,
    )
    print(result.stdout)
    if ok:
        assert result.returncode == 0, (args, result.returncode)
    return result


def main():
    assert os.environ.get("STT_CLOUD_TESTS") == "1", "Use a disposable cloud runner"
    source = Path(__file__).resolve().parents[1]
    revision = os.environ.get("PLUGIN_REVISION") or run("git", "-C", source, "rev-parse", "HEAD").stdout.strip()
    assert len(revision) == 40
    # Set the disposable home before importing any Hermes module.
    with tempfile.TemporaryDirectory(prefix="stt-lifecycle-") as tmp:
        root = Path(tmp)
        home = root / "profile"
        home.mkdir()
        os.environ["HERMES_HOME"] = str(home)
        for key in ("NINEROUTER_KEY", "NINEROUTER_URL", "NINEROUTER_STT_MODEL"):
            os.environ.pop(key, None)

        from hermes_cli.config import load_config, save_config
        from hermes_cli.plugins import get_plugin_manager
        from agent.transcription_registry import get_provider

        def cli(*args, ok=True):
            return run(sys.executable, "-m", "hermes_cli.main", "plugins", *args, ok=ok)

        def head(path):
            return run("git", "-C", path, "rev-parse", "HEAD").stdout.strip()

        def discover(present):
            with patch("requests.sessions.Session.request", side_effect=AssertionError("Discovery contacted network")):
                manager.discover_and_load(force=True)
                provider = get_provider("9router-stt")
                assert (provider is not None) == present
                if provider is not None:
                    assert provider.is_available()
            return provider

        cli("install", "dhansxd/hermes-9router-stt-plugin", "--ref", revision, "--enable", "--yes-deps")
        installed = home / "plugins" / "9router-stt"
        assert (installed / "plugin.yaml").is_file(), list((home / "plugins").iterdir())
        assert head(installed) == revision
        cli("doctor", "9router-stt", "--ci")
        manager = get_plugin_manager()
        discover(True)
        assert manager.unload("9router-stt")
        assert get_provider("9router-stt") is None
        discover(True)

        cfg = load_config()
        cfg["stt"] = {
            "enabled": True, "provider": "9router-stt", "prompt": "Hermes 9Router",
            "9router-stt": {
                "base_url": "http://gateway.invalid", "model": "fixture/model",
                "language": "id",
            },
        }
        save_config(cfg)
        audio_path = root / "native.wav"
        with wave.open(str(audio_path), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * 160)
        calls = []

        def transport(session, method, url, **kwargs):
            assert method.upper() == "POST", (method, url)
            assert url == "http://gateway.invalid/v1/audio/transcriptions"
            assert kwargs["allow_redirects"] is False
            assert kwargs["data"]["model"] == "fixture/model"
            assert kwargs["data"]["language"] == "id"
            assert kwargs["data"]["prompt"] == "Hermes 9Router"
            assert kwargs["files"]["file"][1].read() == audio_path.read_bytes()
            calls.append(url)
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps({"text": "native cloud transcript"}).encode()
            response._content_consumed = True
            return response

        from tools.transcription_tools import transcribe_audio
        with patch("requests.sessions.Session.request", new=transport):
            result = transcribe_audio(str(audio_path))
        assert result["success"] and result["transcript"] == "native cloud transcript", result
        assert len(calls) == 1

        cli("disable", "9router-stt")
        discover(False)
        cli("enable", "9router-stt", "--yes-deps")
        discover(True)
        pinned = cli("update", "9router-stt", ok=False)
        assert pinned.returncode != 0 and "pinned" in pinned.stdout.lower(), pinned.stdout
        assert head(installed) == revision
        cli("remove", "9router-stt")
        discover(False)
        assert not installed.exists()

        # The official manager supports file:// Git sources. Only this runner's
        # disposable origin moves, allowing a real unpinned update before main moves.
        origin = root / "origin.git"
        run("git", "clone", "--bare", source, origin)
        run("git", "--git-dir", origin, "update-ref", "refs/heads/ci", revision)
        run("git", "--git-dir", origin, "symbolic-ref", "HEAD", "refs/heads/ci")
        cli("install", origin.as_uri(), "--enable", "--yes-deps")
        assert head(installed) == revision
        discover(True)
        writer = root / "writer"
        run("git", "clone", origin.as_uri(), writer)
        with (writer / "README.md").open("a") as readme:
            readme.write("\n")
        run("git", "-C", writer, "add", "README.md")
        run("git", "-C", writer, "-c", "user.name=STT CI", "-c",
            "user.email=ci@example.invalid", "commit", "-m", "Cloud lifecycle update fixture")
        updated = head(writer)
        assert updated != revision
        run("git", "-C", writer, "push", "origin", "HEAD:ci")
        cli("update", "9router-stt")
        assert head(installed) == updated
        discover(True)
        cli("remove", "9router-stt")
        discover(False)
        assert not installed.exists()
        plugins = (load_config() or {}).get("plugins") or {}
        aliases = {"9router-stt", "hermes-9router-stt-plugin"}
        for key in ("enabled", "disabled"):
            assert not aliases.intersection(plugins.get(key) or []), plugins

    print("REAL HERMES LIFECYCLE PASSED: Git install, doctor, native STT, unload, disable, enable, pinned refusal, update, remove")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
