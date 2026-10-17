"""Desktop ports for the shared VTL session. No scientific implementation is copied."""
import sys
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.session import Runtime as Session
from .audio_output import LiveAudio, output_devices
from .profile import ProfileStore
from pathlib import Path


class Runtime(Session):
    def __init__(self, resource_dir, profile_dir, *, playback_allowed=True, legacy_profile=None):
        self.resource_entry=None;engine=None
        if sys.platform=='win32':
            from ptb_worker.native_runtime import DirectoryEntry
            self.resource_entry=DirectoryEntry(resource_dir)
            resource_dir=self.resource_entry.path
        try:
            engine=Engine(resource_dir=resource_dir)
            profile=ProfileStore(profile_dir,legacy_directory=legacy_profile)
            live=LiveAudio(engine,playback_allowed=playback_allowed,config_path=Path(profile_dir)/'audio-settings.json')
            super().__init__(engine,profile,live,output_devices=output_devices)
        except BaseException:
            try:
                if engine:engine.close()
            finally:
                if self.resource_entry:self.resource_entry.close()
            raise

    def close(self):
        if self.closed:return
        try:super().close()
        finally:
            if self.resource_entry:self.resource_entry.close()
