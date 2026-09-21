"""9Router STT plugin — Hermes loads `register(ctx)` from this root package.

The provider implementation lives in ``hermes_9router_stt``; this entry point
is what Hermes plugin discovery imports (it scans the plugin dir for ``__init__.py``).
"""
from .hermes_9router_stt import NineRouterTranscriptionProvider, register

__all__ = ["NineRouterTranscriptionProvider", "register"]