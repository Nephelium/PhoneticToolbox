"""Writable user state, separate from bundled read-only resources."""
import json
import os
from pathlib import Path
import threading
from phonetic_core.vocal_tract.trajectory import validate_frames, validate_pitch_curve

DEFAULT_FAVORITES=['a','i','u','e','o','E','y','2','l','n','t']


def profile_directory():
    return Path(os.environ.get('LOCALAPPDATA', Path.home()/'.local/share'))/'PhoneticToolbox/vocal_tract'


class ProfileStore:
    def __init__(self, directory, *, legacy_directory=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.legacy=Path(legacy_directory) if legacy_directory else None

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
            return {'frames':validate_frames(engine,saved['frames'],for_storage=True),
                    'pitch_curve':validate_pitch_curve(saved.get('pitch_curve',[]))}

    def save_frames(self, frames, pitch_curve=None):
        with self.lock:
            target = self.directory/'keyframes.json'
            temporary = target.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps({'version':3,'frames':frames,'pitch_curve':validate_pitch_curve(pitch_curve or [])},ensure_ascii=False,allow_nan=False),encoding='utf-8')
            temporary.replace(target)

    def validate_presets(self,engine,presets):
        if not isinstance(presets,list):raise ValueError('无效的构形库')
        result=[];ids=set()
        for item in presets:
            if not isinstance(item,dict):raise ValueError('无效的构形')
            key=item.get('id')
            if not isinstance(key,str) or not 1<=len(key)<=64 or key in ids:raise ValueError('构形标识重复或无效')
            pose=validate_frames(engine,[item],for_storage=True)[0]
            if pose.get('aspiration'):raise ValueError('送气帧不作为构形保存')
            if key in engine.presets:raise ValueError('自存构形标识不能覆盖内置构形')
            if not pose['name'].strip():raise ValueError('请填写构形名称')
            ids.add(key);result.append({**pose,'id':key})
        if len(json.dumps(result,ensure_ascii=False).encode('utf-8'))>8_000_000:raise ValueError('构形库超过读取容量')
        return result

    def load_presets(self,engine):
        with self.lock:
            path=self.directory/'presets.json'
            if not path.exists():return []
            if path.stat().st_size>8_000_000:raise ValueError('构形库超过读取容量')
            saved=json.loads(path.read_text('utf-8'))
            if saved.get('version') not in (1,2):raise ValueError('构形库版本不匹配')
            return self.validate_presets(engine,saved['presets'])

    def save_presets(self,engine,presets):
        validated=self.validate_presets(engine,presets)
        with self.lock:
            path=self.directory/'presets.json';tmp=path.with_suffix(f'.{os.getpid()}.tmp')
            tmp.write_text(json.dumps({'version':2,'presets':validated},ensure_ascii=False,allow_nan=False),encoding='utf-8')
            tmp.replace(path)
        return validated

    def load_favorites(self):
        with self.lock:
            path=self.directory/'favorites.json'
            if not path.exists():return list(DEFAULT_FAVORITES)
            if path.stat().st_size>100_000:raise ValueError('常用构形设置过大')
            saved=json.loads(path.read_text('utf-8'))
            if saved.get('version')!=1:raise ValueError('常用构形设置版本不匹配')
            return self.validate_favorites(saved.get('ids'))

    @staticmethod
    def validate_favorites(ids):
        if not isinstance(ids,list) or len(ids)>500 or any(not isinstance(k,str) or not 1<=len(k)<=64 for k in ids) or len(set(ids))!=len(ids):raise ValueError('无效的常用构形设置')
        return list(ids)

    def save_favorites(self,engine,ids):
        ids=self.validate_favorites(ids)
        available=set(engine.presets)|{p['id'] for p in self.load_presets(engine)}
        if any(k not in available for k in ids):raise ValueError('常用构形已不存在，请重新选择')
        with self.lock:
            path=self.directory/'favorites.json';tmp=path.with_suffix(f'.{os.getpid()}.tmp')
            tmp.write_text(json.dumps({'version':1,'ids':ids},ensure_ascii=False),encoding='utf-8');tmp.replace(path)
        return ids
