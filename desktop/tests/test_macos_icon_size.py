"""Mac icon footprint, backward compatibility and per-artwork preferences."""
import base64,json,sys
from pathlib import Path
import pytest
from PyQt6.QtCore import QBuffer,QIODevice
from PyQt6.QtGui import QIcon,QImage,QPainter,QPixmap,QColor
from PyQt6.QtWidgets import QApplication
from ptb_desktop.app_icon import builtin_image,icon_images,macos_icon_images,MACOS_BUILTIN_PERCENT
from ptb_desktop.icon_preferences import IconPreferences

DIST=Path(__file__).resolve().parents[2]/'frontend/public'

@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication(['owned-macos-icon-size-test'])

def bounds(image):
    points=[(x,y) for y in range(image.height()) for x in range(image.width()) if image.pixelColor(x,y).alpha()>=32]
    return min(x for x,y in points),min(y for x,y in points),max(x for x,y in points),max(y for x,y in points)

def custom():
    image=QImage(256,256,QImage.Format.Format_ARGB32);image.fill(0)
    painter=QPainter(image);painter.fillRect(48,12,160,232,QColor('#99aacc'));painter.end()
    buffer=QBuffer();buffer.open(QIODevice.OpenModeFlag.WriteOnly);assert image.save(buffer,'PNG')
    return image,'data:image/png;base64,'+base64.b64encode(bytes(buffer.data())).decode()

def test_builtin_mac_gutter_and_unchanged_custom_alpha(app):
    image=builtin_image(DIST)
    old=icon_images(image)[-1];new=macos_icon_images(image)[-1]
    a,b,c,d=bounds(new);assert 207<=c-a+1<=211 and 207<=d-b+1<=211
    assert a>=22 and b>=22 and c<=233 and d<=233
    assert bounds(old)[2]-bounds(old)[0]+1>=250
    image,_=custom();original=QPixmap.fromImage(image).toImage()
    assert bounds(macos_icon_images(image,100,builtin=False)[-1])==bounds(original)
    assert bounds(macos_icon_images(image,60,builtin=False)[-1])[2]-bounds(macos_icon_images(image,60,builtin=False)[-1])[0]<100

@pytest.mark.skipif(sys.platform!='darwin',reason='Actual native Mac preference capability')
def test_old_preferences_stay_byte_exact_until_user_changes_and_each_icon_remembers(app,tmp_path):
    store=IconPreferences(tmp_path,QIcon(),dist=DIST)
    store.path.write_text(json.dumps(dict(version=1,selected='default',custom=[])))
    old=store.path.read_bytes();assert store.request({'action':'status'})['dockSize']['value']==MACOS_BUILTIN_PERCENT
    assert store.path.read_bytes()==old
    store.request({'action':'resize','id':'default','size':93})
    _,url=custom();state=store.request({'action':'import','name':'avatar.png','dataUrl':url});identity=state['selected']
    assert state['dockSize']['value']==100
    data=state['custom'][0]['dataUrl'];before=bounds(store.icon_for(state).pixmap(256,256).toImage())
    state=store.request({'action':'resize','id':identity,'size':70});after=bounds(store.icon_for(state).pixmap(256,256).toImage())
    assert after[2]-after[0]<before[2]-before[0]
    assert state['custom'][0]['dataUrl']==data
    assert store.request({'action':'select','id':'default'})['dockSize']['value']==93
    assert store.request({'action':'select','id':identity})['dockSize']['value']==70
    assert IconPreferences(tmp_path,QIcon(),dist=DIST).read()['dockSize']['value']==70
    saved=json.loads(store.path.read_text());assert saved['version']==1 and 'dockSize' not in saved
    assert saved['macosSizes']=={'default':93,identity:70}
    assert store.request({'action':'resize','id':identity,'size':100})['dockSize']['value']==100

@pytest.mark.skipif(sys.platform!='darwin',reason='Actual native Mac preference capability')
@pytest.mark.parametrize('size',[59,101,True,82.5,None])
def test_invalid_or_stale_resize_keeps_saved_bytes(app,tmp_path,size):
    store=IconPreferences(tmp_path,QIcon(),dist=DIST);store.request({'action':'select','id':'default'});before=store.path.read_bytes()
    with pytest.raises(ValueError):store.request({'action':'resize','id':'default','size':size})
    assert store.path.read_bytes()==before
    with pytest.raises(ValueError):store.request({'action':'resize','id':'builtin-033','size':80})
    assert store.path.read_bytes()==before

def test_windows_path_has_no_size_capability_and_original_pixels(app,tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'platform','win32')
    store=IconPreferences(tmp_path,QIcon(),dist=DIST);state=store.request({'action':'select','id':'default'})
    assert 'dockSize' not in state
    expected=QPixmap.fromImage(icon_images(builtin_image(DIST))[-1]).toImage()
    actual=store.icon_for(state).pixmap(256,256).toImage()
    assert bounds(actual)==bounds(expected)
    before=store.path.read_bytes()
    with pytest.raises(ValueError):store.request({'action':'resize','id':'default','size':82})
    assert store.path.read_bytes()==before


def test_simulated_darwin_preferences_on_any_host(app,tmp_path,monkeypatch):
    """Exercise the platform branch without claiming native Dock integration."""
    monkeypatch.setattr(sys,'platform','darwin')
    test_old_preferences_stay_byte_exact_until_user_changes_and_each_icon_remembers(app,tmp_path)


@pytest.mark.parametrize('size',[59,101,True,82.5,None])
def test_simulated_darwin_rejected_resize_on_any_host(app,tmp_path,monkeypatch,size):
    monkeypatch.setattr(sys,'platform','darwin')
    test_invalid_or_stale_resize_keeps_saved_bytes(app,tmp_path,size)
