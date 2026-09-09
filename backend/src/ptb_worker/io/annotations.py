"""Bound text before decoding, delegate pure interval parsing to core."""
from phonetic_core.acoustic.textgrid import parse_textgrid
from .limits import Limits, LimitError, FormatError


def decode_textgrid(payload: bytes, limits=Limits()):
    if not isinstance(payload,bytes): raise TypeError('TextGrid bytes required')
    if len(payload)>min(limits.input_bytes,limits.text_bytes): raise LimitError('text_bytes_exceeded')
    try:
        text=payload.decode('utf-16' if payload[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig')
        return parse_textgrid(text,max_chars=limits.text_bytes,max_items=limits.text_items,max_tiers=limits.tiers)
    except (ValueError,UnicodeError) as exc: raise FormatError('Invalid or excessive TextGrid: '+str(exc)) from exc
