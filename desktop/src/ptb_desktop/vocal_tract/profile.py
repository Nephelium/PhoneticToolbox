"""Writable user state, separate from bundled read-only resources."""
import json
import os
from pathlib import Path
import threading
from phonetic_core.vocal_tract.trajectory import validate_frames, validate_pitch_curve, validate_pressure_curve
from phonetic_core.vocal_tract.perturbation import validate_perturbation
from phonetic_core.vocal_tract.suprasegmentals import validate_features
from phonetic_core.vocal_tract.sequence_controls import validate_global_controls
from phonetic_core.vocal_tract.preset_library import LibraryProfile, validate_presets, favorites

DEFAULT_FAVORITES=['a','i','u','e','o','E','y','2','l','n','t']


def profile_directory():
    return Path(os.environ.get('LOCALAPPDATA', Path.home()/'.local/share'))/'PhoneticToolbox/vocal_tract'


class ProfileStore(LibraryProfile):
    def __init__(self, directory, *, legacy_directory=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.legacy=Path(legacy_directory) if legacy_directory else None
        self.root_directory=self.directory;self.legacy_root=self.legacy

    def set_speaker(self,speaker):
        if speaker not in ('JD2','W02'):raise ValueError('无效的声道模型')
        self.directory=self.root_directory if speaker=='JD2' else self.root_directory/'speakers'/speaker
        self.directory.mkdir(parents=True,exist_ok=True)
        self.legacy=self.legacy_root if speaker=='JD2' else None

    def load_frames(self, engine):
        return self.load_sequence(engine)['frames']

    def load_sequence(self, engine):
        with self.lock:
            path = self.directory/'keyframes.json'
            if not path.exists() and self.legacy:path=self.legacy/'keyframes.json'
            if not path.exists():
                return {'frames':[], 'pitch_curve':[]}
            # Report malformed state; never silently replace a user's saved poses.
            if path.stat().st_size>8_000_000:raise ValueError('姿势文件超过读取容量')
            saved=json.loads(path.read_text(encoding='utf-8'))
            result={'frames':validate_frames(engine,saved['frames'],for_storage=True),
                    'pitch_curve':validate_pitch_curve(saved.get('pitch_curve',[]))}
            pressure=validate_pressure_curve(saved.get('pressure_curve',[]))
            if pressure:result['pressure_curve']=pressure
            perturbation=validate_perturbation(saved.get('perturbation'))
            if perturbation is not None:result['perturbation']=perturbation
            controls=validate_global_controls(engine,saved.get('global_controls'))
            if controls:result['global_controls']=controls
            features=validate_features(engine,saved.get('features'))
            if features:result['features']=features
            return result

    def save_frames(self, frames, pitch_curve=None, pressure_curve=None, perturbation=None, global_controls=None,features=None):
        with self.lock:
            target = self.directory/'keyframes.json'
            temporary = target.with_suffix(f'.{os.getpid()}.tmp')
            pressure=validate_pressure_curve(pressure_curve or [])
            perturbation=validate_perturbation(perturbation)
            temporary.write_text(json.dumps({'version':3,'frames':frames,'pitch_curve':validate_pitch_curve(pitch_curve or []),**({'pressure_curve':pressure} if pressure else {}),**({'perturbation':perturbation} if perturbation is not None else {}),**({'global_controls':global_controls} if global_controls else {}),**({'features':features} if features else {})},ensure_ascii=False,allow_nan=False),encoding='utf-8')
            temporary.replace(target)

    def validate_presets(self,engine,presets):return validate_presets(engine,presets)

    def _read_library(self):
        with self.lock:
            path=self.directory/'preset-libraries.json'
            if not path.exists():return None
            if path.stat().st_size>8_000_000:raise ValueError('构型套装超过读取容量')
            return json.loads(path.read_text('utf-8'))

    def _write_library(self,value):
        with self.lock:
            path=self.directory/'preset-libraries.json';temp=path.with_suffix(f'.{os.getpid()}.tmp')
            temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')
            temp.replace(path)

    def _legacy_presets(self,engine):
        with self.lock:
            path=self.directory/'presets.json'
            if not path.exists() and self.legacy:path=self.legacy/'presets.json'
            if not path.exists():return []
            if path.stat().st_size>8_000_000:raise ValueError('构形库超过读取容量')
            saved=json.loads(path.read_text('utf-8'))
            if saved.get('version') not in (1,2):raise ValueError('构形库版本不匹配')
            return self.validate_presets(engine,saved['presets'])

    def _legacy_favorites(self):
        with self.lock:
            path=self.directory/'favorites.json'
            if not path.exists() and self.legacy:path=self.legacy/'favorites.json'
            if not path.exists():return list(DEFAULT_FAVORITES)
            if path.stat().st_size>100_000:raise ValueError('常用构形设置过大')
            saved=json.loads(path.read_text('utf-8'))
            if saved.get('version')!=1:raise ValueError('常用构形设置版本不匹配')
            return self.validate_favorites(saved.get('ids'))

    validate_favorites=staticmethod(favorites)
