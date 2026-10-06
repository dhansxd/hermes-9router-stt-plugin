"""Fresh directory-install regression. Run with Hermes Python."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('stt_plugin_install', root / '__init__.py', submodule_search_locations=[str(root)])
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
registered = []
mod.register(SimpleNamespace(register_transcription_provider=registered.append))
assert len(registered) == 1 and registered[0].name == '9router-stt'
print('REGISTRATION PASSED')
