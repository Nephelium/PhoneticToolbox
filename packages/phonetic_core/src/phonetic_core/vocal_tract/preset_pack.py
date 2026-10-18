"""Portable, model-bound articulatory configurations; data only, no paths."""
import hashlib
import json
from .trajectory import validate_frames
from .preset_library import name
from .stop_timing import STOP_TIMING_METHOD

FORMAT='phonetic-toolbox-vocal-presets'


def validate_pack(engine,value):
    if not isinstance(value,dict) or value.get('format')!=FORMAT or value.get('version') not in (1,2,3):
        raise ValueError('不支持此构型包格式或版本')
    speaker=value.get('speaker')
    if speaker!=engine.speaker:
        raise ValueError('此构型包属于'+('女性' if speaker=='W02' else '男性' if speaker=='JD2' else '未知')+'声道模型，请先切换模型再导入')
    if value.get('speaker_sha256')!=hashlib.sha256(engine.speaker_path.read_bytes()).hexdigest():
        raise ValueError('构型包的声道解剖版本不同，无法直接套用')
    method=value.get('geometry_method')
    if method not in ('m10/r18','m10/r19','m10/r20',engine.geometry_method) or value.get('parameters')!=engine.names:
        raise ValueError('构型包的几何版本或参数顺序不受支持')
    title=value.get('title','声道构型包')
    if not isinstance(title,str) or not 1<=len(title.strip())<=120 or any(ord(c)<32 for c in title):
        raise ValueError('请填写不超过 120 字的单行构型包名称')
    items=value.get('presets')
    if not isinstance(items,list) or not 1<=len(items)<=2000:raise ValueError('构型包应包含 1–2000 个音标')
    result=[];symbols=set()
    for item in items:
        if not isinstance(item,dict):raise ValueError('无效的构型')
        if value['version']==1 and (any(item.get('lip_shape',[])) or item.get('stop_timing')):raise ValueError('独立唇部和爆破时序需要版本 2 构型包')
        pose=validate_frames(engine,[item],for_storage=True)[0]
        symbol=name(pose['name'])
        if pose.get('aspiration') or pose.get('silent'):
            raise ValueError('构型不能使用静音或送气间隔')
        if symbol in symbols:raise ValueError('构型包包含重复音标：'+symbol)
        if item['params']!=pose['params']:raise ValueError('构型包的器官参数超出模型范围')
        symbols.add(symbol);pose['name']=symbol;pose.pop('id',None);pose.pop('frame_id',None);result.append(pose)
    output=dict(format=FORMAT,version=3,title=title.strip(),speaker=speaker,
                speaker_sha256=value['speaker_sha256'],geometry_method=method,
                parameters=list(engine.names),presets=result)
    if len(json.dumps(output,ensure_ascii=False,allow_nan=False).encode())>8_000_000:raise ValueError('构型包超过 8 MB')
    if method!=engine.geometry_method:output.update(target_geometry_method=engine.geometry_method,geometry_notice=f'此包使用 {method}，导入后按当前 {engine.geometry_method} 重算，面积与声音可能变化。')
    if any(p.get('stop_timing') for p in result):
        timing=value.get('stop_timing_method','m10/stop-timing/1')
        if timing not in ('m10/stop-timing/1',STOP_TIMING_METHOD):raise ValueError('不支持此爆破时序方法')
        output['stop_timing_method']=timing
        if timing!=STOP_TIMING_METHOD:output['geometry_notice']=(output.get('geometry_notice','')+' 旧爆破时序数值保留，按修正后的前浊与起声方法重算。').strip()
    return output


def make_pack(engine,value):
    return validate_pack(engine,dict(format=FORMAT,version=3,title=value.get('title','声道构型包'),
        speaker=engine.speaker,speaker_sha256=hashlib.sha256(engine.speaker_path.read_bytes()).hexdigest(),
        geometry_method=engine.geometry_method,stop_timing_method=STOP_TIMING_METHOD,parameters=engine.names,presets=value.get('presets')))
