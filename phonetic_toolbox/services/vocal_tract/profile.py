"""Writable user state, separate from bundled read-only resources."""
import json
import os
from pathlib import Path
import threading
from phonetic_toolbox.core.vocal_tract.trajectory import validate_frames, validate_pitch_curve


def profile_directory():
    return Path(os.environ.get('LOCALAPPDATA', Path.home()/'.local/share'))/'PhoneticToolbox/vocal_tract'


class ProfileStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def load_frames(self, engine):
        return self.load_sequence(engine)['frames']

    def load_sequence(self, engine):
        with self.lock:
            path = self.directory/'keyframes.json'
            if not path.exists():
                return {'frames':[], 'pitch_curve':[]}
            # Report malformed state; never silently replace a user's saved poses.
            saved=json.loads(path.read_text(encoding='utf-8'))
            return {'frames':validate_frames(engine,saved['frames'],for_storage=True),
                    'pitch_curve':validate_pitch_curve(saved.get('pitch_curve',[]))}

    def save_frames(self, frames, pitch_curve=None):
        with self.lock:
            target = self.directory/'keyframes.json'
            temporary = target.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps({'version':2,'frames':frames,'pitch_curve':validate_pitch_curve(pitch_curve or [])},ensure_ascii=False,allow_nan=False),encoding='utf-8')
            temporary.replace(target)
