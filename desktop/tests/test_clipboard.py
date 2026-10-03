"""Native clipboard slot boundaries, without accessing the user's clipboard."""
import json
from ptb_desktop import host


class Clipboard:
    def __init__(self):self.value='previous';self.writes=[]
    def setText(self,value):self.value=value;self.writes.append(value)
    def text(self):return self.value


def test_clipboard_returns_only_status_and_preserves_unicode(monkeypatch):
    clipboard=Clipboard()
    monkeypatch.setattr(host.QApplication,'clipboard',lambda:clipboard)
    value='中文 ḁ e\u0301 𝼆 V𐞀\nʔ'
    assert json.loads(host.Bridge.writeClipboard(None,value))=={'ok':True}
    assert clipboard.value==value


def test_oversized_copy_does_not_touch_clipboard(monkeypatch):
    clipboard=Clipboard()
    monkeypatch.setattr(host.QApplication,'clipboard',lambda:clipboard)
    result=json.loads(host.Bridge.writeClipboard(None,'a'*2_000_001))
    assert not result['ok'] and '过长' in result['error']
    assert clipboard.value=='previous' and clipboard.writes==[]


def test_clipboard_readback_failure_is_not_reported_as_success(monkeypatch):
    clipboard=Clipboard()
    monkeypatch.setattr(clipboard,'setText',lambda value:None)
    monkeypatch.setattr(host.QApplication,'clipboard',lambda:clipboard)
    result=json.loads(host.Bridge.writeClipboard(None,'requested'))
    assert result=={'ok':False,'error':'系统剪贴板暂不可用，请重试。'}
    assert 'previous' not in json.dumps(result)


def test_unavailable_clipboard_has_actionable_error(monkeypatch):
    monkeypatch.setattr(host.QApplication,'clipboard',lambda:None)
    assert json.loads(host.Bridge.writeClipboard(None,''))=={'ok':False,'error':'系统剪贴板暂不可用，请重试。'}
