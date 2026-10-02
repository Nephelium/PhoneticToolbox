"""Immutable sample-frame EDL; all channels share half-open intervals."""
from copy import deepcopy


def frames(spans):
    return sum(int(s['end']) - int(s['start']) for s in spans)


def select(spans, start, end):
    total = frames(spans)
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        raise ValueError('选区必须使用整数采样帧')
    if not 0 <= start <= end <= total:
        raise ValueError('选区超出当前版本')
    out, offset = [], 0
    for span in spans:
        length = span['end'] - span['start']
        left, right = max(start, offset), min(end, offset + length)
        if right > left:
            out.append({**deepcopy(span), 'start': span['start'] + left-offset, 'end': span['start'] + right-offset})
        offset += length
    return out


def remove(spans, start, end):
    select(spans, start, end)
    return select(spans, 0, start) + select(spans, end, frames(spans))


def insert(spans, at, clipboard):
    return select(spans, 0, at) + deepcopy(clipboard) + select(spans, at, frames(spans))
