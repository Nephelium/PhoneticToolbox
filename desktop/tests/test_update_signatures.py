"""Staged trust tests use the public RFC 8032 test vector, never release keys."""
import base64
import copy
import importlib.util
from pathlib import Path
import pytest
from ptb_desktop.update_signatures import SignatureError, manifest_payload, verify_manifest
from test_updates import make_service, manifest, github_release


@pytest.fixture
def signed():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    spec=importlib.util.spec_from_file_location('sign_update_manifest',Path(__file__).resolve().parents[2]/'release/sign_update_manifest.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    key=Ed25519PrivateKey.from_private_bytes(bytes.fromhex('9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60'))
    trust={'public-rfc-fixture':base64.b64encode(key.public_key().public_bytes_raw()).decode()}
    return module.sign_manifest(manifest(),key,'public-rfc-fixture'),trust


def test_valid_and_key_order_independent(signed):
    value,trust=signed
    verify_manifest(dict(reversed(list(value.items()))),trust)


@pytest.mark.parametrize('field',['notes','notesByLocale','version','packages'])
def test_tamper_rejected(signed,field):
    value,trust=signed;value=copy.deepcopy(value)
    value[field]='changed'
    with pytest.raises(SignatureError):verify_manifest(value,trust)


def test_missing_unknown_and_malformed_signatures(signed):
    value,trust=signed
    with pytest.raises(SignatureError):verify_manifest(manifest(),trust)
    with pytest.raises(SignatureError):verify_manifest(value,{'other':next(iter(trust.values()))})
    value['signature']['value']='invalid'
    with pytest.raises(SignatureError):verify_manifest(value,trust)


def test_unsafe_numeric_payload_rejected():
    for value in (float('nan'),1.1,9007199254740992):
        with pytest.raises(SignatureError):manifest_payload({'size':value})


def test_service_signed_manifest_and_no_unsigned_fallback(tmp_path,signed):
    value,trust=signed
    service,_=make_service(tmp_path/'valid',server=value,trusted_keys=trust)
    assert service.check(manual=True)['status']=='available'
    service,_=make_service(tmp_path/'unsigned',server=manifest(),github=[github_release()],trusted_keys=trust)
    assert service.check(manual=True)['status']!='available'


def test_stage_default_preserves_existing_channel(tmp_path):
    service,_=make_service(tmp_path)
    assert service.check(manual=True)['status']=='available'
