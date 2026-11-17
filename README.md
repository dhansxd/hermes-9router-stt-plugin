# hermes-9router-stt-plugin

Native speech-to-text backend for [Hermes Agent](https://hermes-agent.nousresearch.com/docs/user-guide/features/plugins), using a [9Router](https://github.com/decolua/9router) gateway's OpenAI-compatible `/v1/audio/transcriptions` endpoint.

## Install and configure

Use Hermes' plugin manager with the complete repository; no symlinks or core changes:

```bash
hermes plugins install dhansxd/hermes-9router-stt-plugin --enable --yes-deps
hermes plugins doctor 9router-stt --ci
```

In the active Hermes profile's `config.yaml`:

```yaml
stt:
  enabled: true
  provider: 9router-stt
  9router-stt:
    base_url: http://localhost:20128
    model: gemini/gemini-2.5-flash
    fallback_models:
      - gemini/gemini-2.5-flash-lite
    timeout: 120
```

Use model IDs exposed by your gateway's `/v1/models/stt`. Omit `model` for automatic catalog selection. The legacy `stt.9router` settings remain supported; matching keys in `stt.9router-stt` take precedence.

Set `NINEROUTER_KEY` in the active profile's `.env` if the gateway requires authentication. An auth-free local gateway needs no key. Optional `NINEROUTER_URL` and `NINEROUTER_STT_MODEL` also use Hermes' profile-scoped environment resolution.

## Lifecycle

Use [Hermes' official commands](https://hermes-agent.nousresearch.com/docs/user-guide/features/plugins):

```bash
hermes plugins disable 9router-stt
hermes plugins enable 9router-stt
hermes plugins update 9router-stt
hermes plugins remove 9router-stt
```

Disable/remove makes this backend unavailable; choose another `stt.provider` when you want to use STT without it. An install pinned with `--ref FULL_COMMIT_SHA` deliberately cannot update until reinstalled without the pin. Restart long-running Hermes processes after lifecycle changes as directed by Hermes.

## Model and request behavior

One primary is selected: argument → config → `NINEROUTER_STT_MODEL` → first catalog model. Only `fallback_models` are additional candidates, in order with duplicates removed.

Fallback occurs on HTTP 429, including non-JSON responses, or a quota/rate-limit error. Authentication, ACL permission, invalid-request and redirect errors return immediately. In 9Router 0.5.119, HTTP 403 can be labeled `insufficient_quota` even for ACL denial; that code alone does not trigger fallback.

Catalog results are cached for 300 seconds per active profile, gateway and credential identity; concurrent callers share the fetch. Discovery/registration never contact the gateway. Requests suppress ambient `.netrc` authentication and automatic redirects, while preserving standard proxy and CA environment support. Errors return status/category without upstream bodies, raw exceptions or credentials.

`language` is optional and defaults to gateway auto-detection. Hermes vocabulary `prompt` hints pass through. Audio MIME types include mp3, mpga, mpeg, wav, m4a, mp4, webm, ogg, oga, opus, flac and aac; actual format/model support depends on 9Router and its provider. Hermes' native dispatcher validates and chunks uploads; direct provider callers remain responsible for file sizes because Requests buffers multipart uploads in memory.

## Cloud regression checks

Cloud verification targets the unchanged official Hermes revision `ae2a81ccc00e9b7c06301523e5bb09608397149c` (v0.23.2), with its locked minimal Python 3.14 runtime. The regression scripts compare the original plugin baseline, run transport/profile regressions with mocked network, and exercise the real Hermes plugin manager, native STT dispatcher and CLI install/enable/disable/update/remove in a disposable runner profile. No gateway secrets or paid provider calls are needed.

This validates the pinned Hermes host and mocked 9Router 0.5.119 API contract. It does not establish transcription quality, real provider availability, or native execution on every OS.

## License

MIT
