"""Model-owned IPA construction and non-destructive global articulation."""
import copy
import math
import unicodedata
from .lips import LIP_KEYS, LIP_LIMITS, validate_lip_shape
from .source import validate_source
from .perturbation import DEFAULT_PERTURBATION, validate_perturbation


EXTRA_LIMITS = {'lip_width': (.55, 1.6), 'larynx_height': (-1., 1.),
                'f0':(60.,350.),'pressure_pa':(0.,1600.),'jitter':(0.,100.),'shimmer':(0.,100.),**dict(zip(LIP_KEYS,LIP_LIMITS))}


def validate_global_controls(engine, value=None):
    if value is None or value == {}:
        return {}
    if not isinstance(value, dict) or len(value) > len(engine.names) + len(EXTRA_LIMITS):
        raise ValueError('无效的全局构音设置')
    limits = {name: (float(lo), float(hi)) for name, lo, hi in
              zip(engine.names, engine.minimum, engine.maximum)} | EXTRA_LIMITS
    result = {}
    for key, item in value.items():
        if key not in limits or not isinstance(item, dict) or set(item) != {'mode', 'value'}:
            raise ValueError('无效的全局构音参数')
        mode, number = item['mode'], item['value']
        if mode not in ('fixed', 'offset') or type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError('全局构音调整需要有限的固定值或偏移值')
        lo, hi = limits[key]
        low, high = (lo, hi) if mode == 'fixed' else (lo - hi, hi - lo)
        if not low <= number <= high:
            raise ValueError('全局构音调整超出范围：' + key)
        result[key] = {'mode': mode, 'value': float(number)}
    return result


def apply_global_controls(engine, pose, controls):
    """Apply after interpolation, before the native contact solver; never mutate."""
    if not controls:
        return pose
    result = copy.deepcopy(pose)
    result['params'] = list(result['params'])
    limited = []
    for key, item in controls.items():
        if key in EXTRA_LIMITS:
            lo, hi = EXTRA_LIMITS[key]
            if key in LIP_KEYS:original=validate_lip_shape(result.get('lip_shape'))[LIP_KEYS.index(key)]
            elif key=='pressure_pa':original=validate_source(result.get('source'))['pressure_pa']
            elif key in ('jitter','shimmer'):original=(validate_perturbation(result.get('perturbation')) or DEFAULT_PERTURBATION)[key]
            else:original = result.get(key, 1. if key == 'lip_width' else 0.)
        else:
            index = engine.names.index(key)
            lo, hi = engine.minimum[index], engine.maximum[index]
            original = result['params'][index]
        wanted = item['value'] if item['mode'] == 'fixed' else original + item['value']
        actual = float(min(hi, max(lo, wanted)))
        if actual != wanted:
            limited.append(key)
        if key in EXTRA_LIMITS:
            if key in LIP_KEYS:
                result['lip_shape']=validate_lip_shape(result.get('lip_shape'));result['lip_shape'][LIP_KEYS.index(key)]=actual
            elif key=='pressure_pa':result['source']={**validate_source(result.get('source')),'pressure_pa':actual}
            elif key in ('jitter','shimmer'):result['perturbation']={**(validate_perturbation(result.get('perturbation')) or DEFAULT_PERTURBATION),key:actual}
            elif key=='f0' and (result.get('silent') or result.get('aspiration')):continue
            else:result[key] = actual
        else:
            result['params'][index] = actual
            if key in ('TRX', 'TRY'):
                result['manual_root'] = True
    result['global_limited'] = limited
    return result


def normalized_symbol(value):
    value = unicodedata.normalize('NFC', value.strip())
    if len(value) >= 2 and (value[0], value[-1]) in (('/', '/'), ('[', ']')):
        value = value[1:-1].strip()
    return value


def ipa_default_duration(symbol):
    """Editable authoring starting points in seconds, not language norms or VOT."""
    plain = ''.join(c for c in unicodedata.normalize('NFD', symbol)
                    if not unicodedata.combining(c))
    base = plain[:1]
    stops = 'pbtdʈɖcɟkɡgqɢʔɓɗʄɠʛʘǀǃǂǁ'
    fricatives = 'fvs zʃʒʂʐɕʑθðçʝxɣχʁħʕhɦɸβɬɮ'.replace(' ', '')
    if base in stops and any(c in fricatives for c in plain[1:]):
        duration = .14  # Affricate, with or without an IPA tie bar.
    elif base in stops:
        duration = .08
    elif base in fricatives:
        duration = .16 if base in 'vzʒʐʑðʝɣʁʕɦβɮ' else .18
    elif base in 'mɱnɳɲŋɴ':
        duration = .12
    elif base in 'ɾɽ':
        duration = .06
    elif base in 'rʀʙɹɻjɰwʍɥlɭʎʟ':
        duration = .10
    else:
        duration = .20  # Vowels and user-defined configuration labels.
    if 'ʰ' in plain and base in stops:
        duration += .04
    if 'ː' in plain:
        duration *= 2
    elif 'ˑ' in plain:
        duration *= 1.5
    return round(duration, 3)


def ipa_sequence(engine, text, presets, duration=None,*,stop_timing=False):
    """Resolve all symbols before returning frames. No missing-symbol fallback."""
    from .trajectory import validate_frames
    from .stop_timing import default_stop_timing
    if type(stop_timing) is not bool:raise ValueError('无效的爆破时序选项')
    if not isinstance(text, str) or len(text) > 4096 or any(ord(c) < 32 and not c.isspace() for c in text):
        raise ValueError('请输入不超过 4096 字的音标序列')
    if duration is not None and (type(duration) not in (int, float) or not math.isfinite(duration) or not .05 <= duration <= 3):
        raise ValueError('单音时长应为 0.05–3 秒')
    if not isinstance(presets, list) or len(presets) > 2000:
        raise ValueError('构型库过大或无效')
    catalog = {}
    for item in presets:
        if not isinstance(item, dict) or not isinstance(item.get('name'), str):
            raise ValueError('无效的已存构型')
        key = normalized_symbol(item['name'])
        if key and not any(c.isspace() for c in key):
            catalog[key] = item
    symbols = sorted(catalog, key=len, reverse=True)
    text = normalized_symbol(text)
    if not text:
        raise ValueError('请输入国际音标')
    tokens, missing = [], []
    for word in text.split():
        offset = 0
        while offset < len(word):
            match = next((s for s in symbols if word.startswith(s, offset)), None)
            # A combining mark, tie bar, length/aspiration/stress/tone marker
            # may not be silently detached from an otherwise recognized base.
            end = offset + len(match) if match else offset + 1
            while end < len(word) and (unicodedata.combining(word[end]) or word[end] in 'ʰʷʲˠˤːˑ˞'):
                end += 1
            if match and end == offset + len(match) and not match.endswith(('͡', '͜')):
                tokens.append(match)
            else:
                unknown = word[offset:end]
                # Tie bars attach the following base and its diacritics too.
                if any(c in unknown for c in ('͡', '͜')) and end < len(word):
                    end += 1
                    while end < len(word) and unicodedata.combining(word[end]):
                        end += 1
                    unknown = word[offset:end]
                if unknown not in missing:
                    missing.append(unknown)
            offset = end
    if missing:
        return {'frames': [], 'missing': missing, 'symbols': tokens}
    frames = []
    for symbol in tokens:
        frame = copy.deepcopy(catalog[symbol])
        old_duration=frame.get('duration',.2)
        frame.update(name=symbol, duration=ipa_default_duration(symbol) if duration is None else float(duration))
        frame.pop('id', None)
        frame.pop('frame_id',None)
        if frame.get('stop_timing'):
            if duration is None:frame['duration']=old_duration
            else:frame['stop_timing']={k:v*frame['duration']/old_duration for k,v in frame['stop_timing'].items()}
        frames.append(frame)
    if stop_timing:
        for frame in frames[:-1]:
            plain=''.join(c for c in unicodedata.normalize('NFD',frame['name']) if not unicodedata.combining(c))
            if frame.get('stop_timing') or not plain or plain[0] not in 'pbtdʈɖcɟkɡgqɢʔ' or any(c not in 'ʰʷʲˠˤːˑ' for c in plain[1:]):continue
            frame['stop_timing']=default_stop_timing(frame['duration'])
            if plain[0] in 'bdɖɟɡgɢ':
                timing=frame['stop_timing'];timing['closure']=frame['duration']-timing['release'];timing['aspiration']=0.
                timing['vot']=-min(.03,timing['closure'])
    return {'frames': validate_frames(engine, frames, for_storage=True),
            'missing': [], 'symbols': tokens}
