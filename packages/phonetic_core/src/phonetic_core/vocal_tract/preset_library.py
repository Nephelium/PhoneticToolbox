"""Shared, data-only validation and edits for named configuration libraries."""
import copy
import json
import uuid
from collections import Counter
from .trajectory import validate_frames
from .sequence_controls import normalized_symbol

DEFAULT_FAVORITES=['a','i','u','e','o','E','y','2','l','n','t']


def name(value, limit=80):
    if not isinstance(value,str) or not 1<=len(value.strip())<=limit or any(ord(c)<32 or ord(c)==127 for c in value):
        raise ValueError(f'名称须为 1–{limit} 字的单行文本')
    value=normalized_symbol(value)
    if not value:raise ValueError('请填写构型名称')
    return value


def validate_presets(engine, items, *, unique=False):
    if not isinstance(items,list) or len(items)>2000:raise ValueError('每套最多保存 2000 个构型')
    result=[];ids=set();names=set()
    for item in items:
        if not isinstance(item,dict):raise ValueError('无效的构型')
        key=item.get('id')
        if not isinstance(key,str) or not 1<=len(key)<=64 or key in ids or key in engine.presets:raise ValueError('构型标识重复或无效')
        pose=validate_frames(engine,[item],for_storage=True)[0]
        if pose.get('silent') or pose.get('aspiration'):raise ValueError('静音或送气间隔不能保存为构型')
        label=name(pose['name'])
        if unique and label in names:raise ValueError('构型名称重复，请先更名：'+label)
        ids.add(key);names.add(label);result.append({**pose,'name':label,'id':key})
    return result


def favorites(ids):
    if not isinstance(ids,list) or len(ids)>500 or any(not isinstance(k,str) or not 1<=len(k)<=64 for k in ids) or len(set(ids))!=len(ids):raise ValueError('无效的常用构型设置')
    return list(ids)


def initial_library(presets, ids):
    return dict(version=1,active='default',sets=[dict(id='default',name='默认构型',presets=copy.deepcopy(presets),favorites=favorites(ids))])


def active_set(value):
    if not isinstance(value,dict) or value.get('version')!=1 or not isinstance(value.get('sets'),list) or not 1<=len(value['sets'])<=100:raise ValueError('构型套装记录无效，原数据已保留')
    result=next((s for s in value['sets'] if isinstance(s,dict) and s.get('id')==value.get('active')),None)
    if result is None:raise ValueError('找不到当前构型套装')
    return result


def validate_library(engine, value):
    active_set(value);sets=[];ids=set();names=set()
    for item in value['sets']:
        if not isinstance(item,dict):raise ValueError('无效的构型套装')
        key=item.get('id');label=name(item.get('name'),120)
        if not isinstance(key,str) or not 1<=len(key)<=64 or key in ids:raise ValueError('构型套装标识重复或无效')
        if label in names:raise ValueError('套装名称已存在：'+label)
        presets=validate_presets(engine,item.get('presets'));common=favorites(item.get('favorites',DEFAULT_FAVORITES))
        available=set(engine.presets)|{p['id'] for p in presets}
        if any(k not in available for k in common):raise ValueError('常用构型已不存在，请重新选择')
        ids.add(key);names.add(label);sets.append(dict(id=key,name=label,presets=presets,favorites=common))
    result=dict(version=1,active=value['active'],sets=sets)
    if len(json.dumps(result,ensure_ascii=False,allow_nan=False).encode())>8_000_000:raise ValueError('构型套装合计超过 8 MB，请分开保存')
    return result


def edit_library(engine,value,operation):
    result=validate_library(engine,value);current=active_set(result);action=operation.get('action')
    if action=='select':
        result['active']=operation.get('id');active_set(result)
    elif action=='rename':current['name']=name(operation.get('name'),120)
    elif action in ('create','copy'):
        result['sets'].append(dict(id=uuid.uuid4().hex,name=name(operation.get('name'),120),
            presets=copy.deepcopy(current['presets']) if action=='copy' else [],
            favorites=list(current['favorites']) if action=='copy' else list(DEFAULT_FAVORITES)))
        result['active']=result['sets'][-1]['id']
    else:raise ValueError('无效的构型套装操作')
    return validate_library(engine,result)


class LibraryProfile:
    """Adapters implement only raw storage and legacy reads; shared semantics here."""
    def load_library(self,engine):
        raw=self._read_library()
        if raw is None:
            presets=self._legacy_presets(engine)
            available=set(engine.presets)|{p['id'] for p in presets}
            # Older saves could leave favorites pointing to removed poses. The
            # old UI omitted those buttons; preserve that view and source bytes.
            raw=initial_library(presets,[k for k in self._legacy_favorites() if k in available])
        return validate_library(engine,raw)

    def save_library(self,engine,value):
        result=validate_library(engine,value);self._write_library(result);return copy.deepcopy(result)

    def load_presets(self,engine):return active_set(self.load_library(engine))['presets']

    def save_presets(self,engine,items):
        items=validate_presets(engine,items);library=self.load_library(engine)
        current=active_set(library)
        before=Counter(p['name'] for p in current['presets']);after=Counter(p['name'] for p in items)
        duplicates={k:n-1 for k,n in after.items() if n>1}
        # Permit one-at-a-time repair of several old duplicate groups, while
        # refusing any newly introduced collision or an unchanged conflict.
        if duplicates and (any(n>max(0,before[k]-1) for k,n in duplicates.items()) or
                           sum(duplicates.values())>=sum(max(0,n-1) for n in before.values())):
            raise ValueError('构型名称重复，请先更名：'+next(iter(duplicates)))
        current['presets']=items
        available=set(engine.presets)|{p['id'] for p in items}
        current['favorites']=[k for k in current['favorites'] if k in available]
        self.save_library(engine,library);return copy.deepcopy(items)

    def load_favorites(self):
        raw=self._read_library()
        return favorites(active_set(raw)['favorites']) if raw is not None else self._legacy_favorites()

    def save_favorites(self,engine,ids):
        library=self.load_library(engine);active_set(library)['favorites']=favorites(ids)
        self.save_library(engine,library);return list(ids)
