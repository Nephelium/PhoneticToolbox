"""Browser handoff keeps the local workbench and rejects non-web popups."""
from types import SimpleNamespace
import pytest
from PyQt6.QtCore import QUrl
from ptb_desktop import host


@pytest.mark.parametrize('target',['https://doi.org/10.1016/j.wocn.2018.07.001','http://www.praat.org/'])
def test_user_new_window_hands_exact_url_to_browser(monkeypatch,target):
    opened=[];monkeypatch.setattr(host.QDesktopServices,'openUrl',lambda url:opened.append(url.toString()))
    request=SimpleNamespace(isUserInitiated=lambda:True,requestedUrl=lambda:QUrl(target))
    host.Page.open_external_window(SimpleNamespace(external_url=host.Page.external_url),request)
    assert opened==[target]


@pytest.mark.parametrize('target,user',[
    ('https://example.com/popup',False),('file:///C:/private.txt',True),
    ('javascript:alert(1)',True),('data:text/html,hi',True),
    ('ptbapp://app/index.html',True),('https:///no-host',True),
])
def test_non_user_or_non_web_windows_are_not_handed_off(monkeypatch,target,user):
    opened=[];monkeypatch.setattr(host.QDesktopServices,'openUrl',lambda url:opened.append(url.toString()))
    request=SimpleNamespace(isUserInitiated=lambda:user,requestedUrl=lambda:QUrl(target))
    host.Page.open_external_window(SimpleNamespace(external_url=host.Page.external_url),request)
    assert opened==[]


def test_external_same_frame_link_does_not_replace_workbench(monkeypatch):
    opened=[];monkeypatch.setattr(host.QDesktopServices,'openUrl',lambda url:opened.append(url.toString()))
    page=SimpleNamespace(external_url=host.Page.external_url)
    link=host.QWebEnginePage.NavigationType.NavigationTypeLinkClicked
    other=host.QWebEnginePage.NavigationType.NavigationTypeOther
    assert not host.Page.acceptNavigationRequest(page,QUrl('http://www.praat.org/'),link,True)
    assert opened==['http://www.praat.org/']
    assert not host.Page.acceptNavigationRequest(page,QUrl('https://example.com/'),other,True)
    assert host.Page.acceptNavigationRequest(page,QUrl('ptbapp://app/index.html'),other,True)
    assert not host.Page.acceptNavigationRequest(page,QUrl('file:///C:/private.txt'),link,True)
    assert opened==['http://www.praat.org/']
