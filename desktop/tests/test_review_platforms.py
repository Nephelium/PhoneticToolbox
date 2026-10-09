from pathlib import Path
import pytest
from ptb_desktop.platform_paths import user_data_root,windows_workbench_storage


@pytest.mark.parametrize('platform,env,suffix',[
    ('win32',{'LOCALAPPDATA':'D:/AppData'},'AppData/PhoneticToolbox/v3'),
    ('darwin',{},'Library/Application Support/PhoneticToolbox/v3'),
    ('linux',{'XDG_DATA_HOME':'D:/xdg'},'xdg/PhoneticToolbox/v3'),
    ('linux',{},'.local/share/PhoneticToolbox/v3'),
])
def test_user_data_location_does_not_require_windows_environment(platform,env,suffix,tmp_path):
    assert user_data_root(platform=platform,environ=env,home=tmp_path).as_posix().endswith(suffix)


def test_relative_xdg_directory_is_ignored(tmp_path):
    assert user_data_root(platform='linux',environ={'XDG_DATA_HOME':'relative'},home=tmp_path)==tmp_path/'.local/share/PhoneticToolbox/v3'


def test_unknown_platform_fails_explicitly(tmp_path):
    with pytest.raises(ValueError):user_data_root(platform='other',environ={},home=tmp_path)


def test_windows_ui_storage_preserves_original_qt_directory(tmp_path):
    location=windows_workbench_storage(environ={'LOCALAPPDATA':str(tmp_path)},home=tmp_path)
    assert location==tmp_path/'PhoneticToolbox-v3/workbench'
    assert windows_workbench_storage(environ={'LOCALAPPDATA':'relative'},home=tmp_path)==tmp_path/'AppData/Local/PhoneticToolbox-v3/workbench'
