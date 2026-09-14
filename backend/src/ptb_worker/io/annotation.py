"""M12 bounded file adapters. Never execute serialized pickle functions."""
import json
import math
import pickle
import numpy as np
from phonetic_core.annotation import parse_document
from .legacy_pickle import read_numeric_pickle, Array
from .lip import convert_local_legacy_lip, decode_lip
from .limits import Limits


def validate_textgrid(payload):
    if type(payload) != str or len(payload.encode('utf-8')) > 2_000_000:
        raise ValueError('TextGrid 超过 2 MB 或格式无效。')
    return parse_document(payload)


def lip_preview(payload, name, companion=None):
    if name.lower().endswith('.pkl'):
        return json.loads(convert_local_legacy_lip(payload, companion=companion))
    decode_lip(payload)
    return json.loads(payload)


def update_lip_offset(payload, name, offset):
    if type(offset) not in (float, int) or not math.isfinite(offset) or abs(offset) > 3600:
        raise ValueError('唇偏必须是 −3600 至 3600 秒的有限数值。')
    if name.lower().endswith('.pkl'):
        # Symbolic parsing validates every value and never resolves input globals.
        tree = read_numeric_pickle(payload, Limits())
        def restore(value):
            if type(value) == Array:
                return np.frombuffer(value.raw, dtype=np.dtype(value.dtype.endian+value.dtype.code)).copy().reshape(value.shape, order=value.order)
            if type(value) == list:
                return [restore(v) for v in value]
            if type(value) == tuple:
                return tuple(restore(v) for v in value)
            if type(value) == dict:
                return {k: restore(v) for k, v in value.items()}
            return value
        data = restore(tree)
        if type(data) != dict or type(data.get('metadata', {})) != dict:
            raise ValueError('唇形 metadata 格式无效，未写入。')
        data.setdefault('metadata', {})['lip_manual_offset'] = float(offset)
        # Only the validated inert graph and fixed numeric arrays are serialized.
        raw = pickle.dumps(data, protocol=4)
        read_numeric_pickle(raw, Limits())
    else:
        decode_lip(payload)
        data = json.loads(payload)
        data['data'].setdefault('metadata', {})['lip_manual_offset'] = float(offset)
        raw = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        decode_lip(raw)
    if len(raw) > 16_000_000:
        raise ValueError('唇形保存超过 16 MB。')
    return raw
