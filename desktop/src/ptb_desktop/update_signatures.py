"""Staged update trust policy. Empty keys mean signing is not activated yet.

Only reviewed, frozen application code may set these public keys. Never load a
trust root from a downloaded manifest, renderer request or writable preferences.
"""
import base64
import json

TRUSTED_UPDATE_KEYS: dict[str,str] = {}
DOMAIN = b'PhoneticToolbox release manifest signature v1\0'


class SignatureError(ValueError):
    pass


def manifest_payload(manifest):
    if not isinstance(manifest,dict):raise SignatureError('signature_manifest_invalid')
    body={k:v for k,v in manifest.items() if k!='signature'}
    def validate(value):
        if value is None or isinstance(value,(str,bool)):return
        if type(value) is int and abs(value)<=9007199254740991:return
        if isinstance(value,list):
            for item in value:validate(item)
            return
        if isinstance(value,dict) and all(isinstance(key,str) for key in value):
            for item in value.values():validate(item)
            return
        raise SignatureError('signature_manifest_invalid')
    validate(body)
    return DOMAIN+json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')


def verify_manifest(manifest, trusted_keys):
    if not trusted_keys:raise SignatureError('signature_trust_not_configured')
    signature=manifest.get('signature') if isinstance(manifest,dict) else None
    if not isinstance(signature,dict) or set(signature)!={'algorithm','keyId','value'} or signature['algorithm']!='Ed25519':
        raise SignatureError('signature_required')
    key_id=signature['keyId']
    if not isinstance(key_id,str) or key_id not in trusted_keys:raise SignatureError('signature_unknown_key')
    try:
        public=base64.b64decode(trusted_keys[key_id],validate=True)
        raw=base64.b64decode(signature['value'],validate=True)
        if len(public)!=32 or len(raw)!=64:raise ValueError()
    except (TypeError,ValueError):raise SignatureError('signature_invalid_encoding') from None
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
    except ImportError:raise SignatureError('signature_runtime_unavailable') from None
    try:Ed25519PublicKey.from_public_bytes(public).verify(raw,manifest_payload(manifest))
    except (InvalidSignature,ValueError):raise SignatureError('signature_invalid') from None
