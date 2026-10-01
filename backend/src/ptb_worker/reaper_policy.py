"""Per-request REAPER policy; lazy native construction also permits Python-only hosts."""
from dataclasses import replace
from phonetic_core.ports.acoustic import python_reaper
from phonetic_core.ports.errors import BackendAborted


class PolicyReaper:
    def __init__(self, policy, native_factory, python_backend=python_reaper):
        if policy not in ('native_required','native_then_python','python_only'):
            raise ValueError('Invalid active REAPER policy')
        self.policy, self.native_factory, self.python_backend = policy, native_factory, python_backend
        self.native_sha256 = None

    def __call__(self, *args, **kwargs):
        if self.policy == 'python_only':
            return self.python_backend(*args, **kwargs)
        try:
            native = self.native_factory()
            result = native(*args, **kwargs)
            self.native_sha256 = native.sha256
            return result
        except (BackendAborted, MemoryError):
            raise
        except (OSError, ValueError, RuntimeError):
            if self.policy == 'native_required':
                raise
            return replace(self.python_backend(*args, **kwargs), reason='native_failed')
