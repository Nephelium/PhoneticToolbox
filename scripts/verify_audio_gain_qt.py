"""Actual muted Qt WebEngine rendering of the shared workbench on owned Vite."""
import json
import os
from pathlib import Path
import time
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --mute-audio --autoplay-policy=no-user-gesture-required')
from PyQt6.QtCore import QEventLoop,QTimer,QUrl,QPoint,Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6 import sip

app=QApplication(['Audition gain QA']);view=QWebEngineView();profile=QWebEngineProfile('audition-gain-qa',app);profile.setPersistentStoragePath(str(Path(os.environ['AUDIO_GAIN_SCRATCH'])/'qt-profile'));profile.setCachePath(str(Path(os.environ['AUDIO_GAIN_SCRATCH'])/'qt-cache'));view.setPage(QWebEnginePage(profile,view));view.resize(1440,900);view.page().setAudioMuted(True);view.show()
def pause(ms=100):
    loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
def js(code):
    result=[];loop=QEventLoop();view.page().runJavaScript(code,lambda value:(result.append(value),loop.quit()));QTimer.singleShot(10000,loop.quit);loop.exec();return result[0] if result else None
def until(code):
    deadline=time.monotonic()+40
    while time.monotonic()<deadline:
        if js(code):return
        pause()
    raise AssertionError(code)
def click(selector):
    point=js('(()=>{const e=document.querySelector('+json.dumps(selector)+'),r=e.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2];})()');QTest.mouseClick(view.focusProxy(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(point[0]),round(point[1])));pause()
try:
    view.load(QUrl(os.environ['AUDIO_GAIN_URL']));until('!!document.querySelector("#tool-sidebar")')
    js('[...document.querySelectorAll("#tool-sidebar button")].find(e=>e.textContent.trim()==="设置").click()');until('!!document.querySelector("#settings-tab-audio")');click('#settings-tab-audio');selector='[aria-label="音频音量自动增益"]';assert js('document.querySelector('+json.dumps(selector)+').checked') is True
    js('window.__gains=[];const old=AudioContext.prototype.createGain;AudioContext.prototype.createGain=function(){const g=old.call(this);window.__gains.push(g);return g;};import(new URL("/src/state/audio.ts",location.href).href).then(async audio=>{window.__audio=audio;audio.volume(.5);const pcm=new Float32Array(48000);pcm.fill(.1);await audio.play({name:"quiet",channels:[pcm],sampleRate:48000,frames:48000,duration:1},0,1,0);window.__gainReady=true;}).catch(e=>window.__gainError=String(e));');until('window.__gainReady===true||!!window.__gainError');assert js('window.__gainError||null') is None,js('window.__gainError');assert abs(js('window.__gains.at(-1).gain.value')-4.4562546)<1e-5
    click(selector);assert js('window.__gains.at(-1).gain.value')==.5
    click(selector);assert js('window.__gains.at(-1).gain.value')>4
    for theme in ('light','dark'):
        js('document.documentElement.dataset.theme='+json.dumps(theme));pause();view.grab().save(str(Path(os.environ['AUDIO_GAIN_OUT'])/('qt-settings-'+theme+'.png')))
    js('window.__audio.stop()');click('[aria-label="收起侧栏"]')
    offsets=js('(()=>{const e=document.querySelector("#tool-sidebar"),r=e.getBoundingClientRect(),center=r.x+(r.width-1)/2;return [...e.querySelectorAll(".brand img,.nav-item>svg,.nav-item>.ipa-icon,.utility-grid svg")].map(e=>{const r=e.getBoundingClientRect();return r.x+r.width/2-center;});})()');assert all(abs(offset)<1 for offset in offsets),offsets
    view.grab().save(str(Path(os.environ['AUDIO_GAIN_OUT'])/'qt-sidebar-collapsed.png'));print(json.dumps({'success':True,'checks':['Qt settings default/toggle','Qt actual AudioContext output gain and disable','light/dark render','collapsed icon centers']}))
finally:
    view.close();sip.delete(view);pause();sip.delete(profile);sip.delete(app)
