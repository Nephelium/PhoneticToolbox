"""Permission policy tests do not change any filesystem ACL."""
import pytest
from ptb_api.private_config import _validate_windows_acl


def acl(sid='owner',rights=0x1f01ff,**changes):
    rule=dict(sid=sid,rights=rights,allow=True,inheritOnly=False);rule.update(changes)
    return dict(user='owner',owner='owner',rules=[rule])


@pytest.mark.parametrize('sid',['owner','S-1-5-18','S-1-5-32-544'])
def test_private_acl_accepts_owner_and_system(sid):
    _validate_windows_acl(acl(sid))


@pytest.mark.parametrize('rights',[1,2,4,0x10000,0x40000,0x80000])
def test_private_acl_rejects_other_reader_writer_or_acl_editor(rights):
    with pytest.raises(ValueError):_validate_windows_acl(acl('other',rights))


def test_inherit_only_and_deny_are_not_access_grants():
    _validate_windows_acl(acl('other',inheritOnly=True))
    _validate_windows_acl(acl('other',allow=False))


def test_owner_and_empty_acl_fail_closed():
    value=acl();value['owner']='other'
    with pytest.raises(ValueError):_validate_windows_acl(value)
    value=acl();value['rules']=[]
    with pytest.raises(ValueError):_validate_windows_acl(value)
