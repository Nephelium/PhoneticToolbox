"""Bounded Qt task envelopes; shared role limits include Base64 expansion."""
import json

M11_INPUT_LIMITS = {'audio':64_000_000, 'dictionary':16_000_000, 'transcript':2_000_000}


def decode_task_request(raw):
    if not isinstance(raw,str) or len(raw)>86_000_000:
        raise ValueError('request_size')
    body=json.loads(raw)
    if not isinstance(body,dict):raise ValueError('request_shape')
    op=body.get('op')
    limit=8_100_000 if op=='annotation_save' else 2_800_000 if op=='m14_import' else 1_000_000
    if op=='m11_import':
        role=body.get('role')
        if role not in M11_INPUT_LIMITS:raise ValueError('request_role')
        encoded=body.get('base64')
        encoded_limit=((M11_INPUT_LIMITS[role]+2)//3)*4
        if not isinstance(encoded,str) or len(encoded)>encoded_limit:raise ValueError('request_size')
        limit=encoded_limit+16_384
    if len(raw)>limit:raise ValueError('request_size')
    return body
