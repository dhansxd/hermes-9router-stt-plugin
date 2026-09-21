# hermes-9router-stt-plugin

Speech-to-text backend plugin for [Hermes Agent](https://hermes-agent.nousresearch.com) that routes STT through a [9Router](https://github.com/decolua/9router) gateway (OpenAI Whisper-compatible `/v1/audio/transcriptions`).

Replaces the local STT backend (`stt.provider`) with any model your 9Router gateway exposes — Gemini 2.5/3.x Flash and more — with 429/quota-gated model fallback.

## Install

1. Copy this plugin into `$HERMES_HOME/plugins/stt/9router-stt/`:

```bash
mkdir -p ~/.hermes/plugins/stt/9router-stt
cp __init__.py plugin.yaml ~/.hermes/plugins/stt/9router-stt/
cp -r src/hermes_9router_stt ~/.hermes/plugins/stt/9router-stt/
```

2. Add to `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - stt/9router-stt

stt:
  enabled: true
  provider: 9router-stt
  9router-stt:
    base_url: http://localhost:20128   # your 9Router gateway
    model: gemini/gemini-2.5-flash     # primary STT model
    fallback_models:                   # tried in order, ONLY on 429/quota errors
      - gemini/gemini-3.8-flash
    timeout: 120
```

3. Set the gateway key (if auth is enabled) in `~/.hermes/.env`:

```
NINEROUTER_KEY=your-9router-api-key
```

## Model selection

Priority: `model` argument → `stt.9router-stt.model` → `NINEROUTER_STT_MODEL` env → first entry of `/v1/models/stt` catalog (TTL 300s).

Fallback chain: `fallback_models` are tried in order when the active model returns a 429/quota error. Hard errors (auth, invalid request) return immediately.

## Notes

- Audio formats: whatever 9Router accepts (mp3, wav, m4a, webm, ogg, flac)
- `language` hint passes through when set (`stt.language`)
- Cross-platform: pure Python (`requests`), no OS-specific code — macOS/Linux/Windows/Termux (point `base_url` at a reachable 9Router)

## License

MIT
