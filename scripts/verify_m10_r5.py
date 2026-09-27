"""M10 R5 owned Qt check for ordering, clear/undo, and replay state."""
import base64
import json
import sys
import time
import uuid
from pathlib import Path

from PyQt6.QtCore import QEventLoop, QTimer, Qt
from PyQt6.QtWidgets import QApplication
from ptb_desktop.host import Workbench, register_scheme

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'output/validation/m10/r5-ui'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    register_scheme()
    app = QApplication(['m10-r5-qa'])
    app.setApplicationName('M10-R5-owned-QA')
    profile = OUT / ('profile-' + uuid.uuid4().hex[:8])
    w = Workbench(ROOT / 'frontend/dist', test=True, vocal_profile=profile,
                  vocal_resources=ROOT / 'resources/vocal_tract/native', start_module='M10')
    w.setFixedSize(1650, 1000)
    w.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    w.show()

    def wait(ms=150):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def js(code):
        loop = QEventLoop()
        box = []
        w.page.runJavaScript(code, lambda value: (box.append(value), loop.quit()))
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        return box[0] if box else None

    def local(code):
        return js("(()=>{const d=document.querySelector('iframe')?.contentDocument;if(!d)return null;" + code + '})()')

    def until(code, seconds=40):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if local('return ' + code + ';'):
                return
            wait()
        raise AssertionError('UI timeout: ' + code + ' ' + str(local("return d.querySelector('#frameStatus')?.textContent;")))

    def click(selector):
        local('d.querySelector(' + json.dumps(selector) + ').click();')
        wait()

    def change(selector, value):
        local('const e=d.querySelector(' + json.dumps(selector) + ');e.value=' + json.dumps(value) + ";e.dispatchEvent(new Event('change'));")
        wait()

    def saved():
        return json.loads((profile / 'keyframes.json').read_text('utf-8'))

    def until_saved(predicate, seconds=20):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if (profile / 'keyframes.json').exists() and predicate(saved()):
                return
            wait()
        raise AssertionError('profile save timeout')

    export = OUT / 'ordered.ptb-vocal.json'
    video = OUT / 'ordered-current.webm'
    w.bridge.test_vocal_picker = lambda op: str({'document/save': export, 'video/begin': video}[op])
    result = {'platform': 'Windows Qt WebEngine development', 'profile': str(profile)}
    try:
        until("d.querySelector('#engineStatus')?.textContent.includes('已连接')", 55)
        until("d.body.dataset.posePending==='false'")
        click('#tab-motion')
        click('[data-preset=a]'); until("d.body.dataset.posePending==='false'"); click('#captureFrame')
        click('[data-preset=i]'); until("d.body.dataset.posePending==='false'"); click('#captureFrame')
        click('#captureSilence')
        until("d.querySelectorAll('.pose-card').length===3")
        change('[aria-label="姿势 3 静音秒数"]', .05)
        until("d.querySelectorAll('.pose-card .pose-time')[2].textContent.includes('0.45')")
        click('#pitchExpand')
        until("d.querySelector('#pitchDialog').open")
        local("const c=d.querySelector('#pitchCanvas');c.focus();c.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowUp',bubbles:true}));")
        until("d.querySelector('#pitchStatus').textContent.includes('实线')")
        click('#pitchClose')
        until_saved(lambda item: len(item['frames']) == 3 and len(item['pitch_curve']) == 201)
        before = saved()
        assert len(before['pitch_curve']) == 201
        # Exercise the page's native drag/drop handlers with real card geometry.
        drag = local("const cards=[...d.querySelectorAll('.pose-card')],list=d.querySelector('#keyframeList'),data=new DataTransfer(),y=cards[2].getBoundingClientRect().bottom+8;cards[0].dispatchEvent(new DragEvent('dragstart',{bubbles:true,dataTransfer:data,clientY:cards[0].getBoundingClientRect().top+8}));list.dispatchEvent(new DragEvent('dragover',{bubbles:true,cancelable:true,dataTransfer:data,clientY:y}));list.dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:data,clientY:y}));return [...d.querySelectorAll('.pose-card .pose-title')].map(e=>e.textContent);")
        until("d.querySelectorAll('.pose-card')[1]?.querySelector('small')?.textContent.includes('静音')")
        until_saved(lambda item: item['frames'] == [before['frames'][1], before['frames'][2], before['frames'][0]])
        ordered = saved()
        assert ordered['frames'] == [before['frames'][1], before['frames'][2], before['frames'][0]], drag
        assert ordered['pitch_curve'] == before['pitch_curve']
        assert local("return [...d.querySelectorAll('.pose-card .pose-time')].map(e=>e.textContent);") == [
            '⠿ 0.00–0.20 s', '⠿ 0.20–0.25 s', '⠿ 0.25–0.45 s']
        assert local("return [...d.querySelectorAll('.pose-card')].findIndex(e=>e.classList.contains('current'));") == 1
        result['drag_order'] = [f.get('preset', '') for f in ordered['frames']]
        result['drag_labels'] = drag
        click('#exportFrames')
        until("d.querySelector('#frameStatus').textContent.includes('已导出')")
        document = json.loads(export.read_text('utf-8'))
        assert document['frames'] == ordered['frames'] and document['pitch_curve'] == ordered['pitch_curve']
        click('#exportVideo'); click('#videoStart')
        until("d.querySelector('#videoStatus').textContent.startsWith('已保存')||d.querySelector('#videoStatus').textContent.includes('失败')", 120)
        result['ordered_video'] = local("return d.querySelector('#videoStatus').textContent;")
        assert video.is_file(), result['ordered_video']
        video_sequence = saved()
        assert [f.get('name') for f in video_sequence['frames']] == [f.get('name') for f in ordered['frames']]
        assert [f['duration'] for f in video_sequence['frames']] == [f['duration'] for f in ordered['frames']]
        (OUT / 'ordered-sequence.json').write_text(json.dumps(video_sequence, ensure_ascii=False), encoding='utf-8')
        current = w.vocal.invoke('animation/prepare', {'frames': video_sequence['frames'],
            'pitch_curve': video_sequence['pitch_curve'],
            'keep_vowel': bool(local("return d.querySelector('#keepVowel').checked;"))})
        assert current['cached'], 'export did not retain the prepared sequence'
        assert current['frames'] == video_sequence['frames'] and current['duration'] == .45
        assert w.vocal.invoke('animation/picture', {'id': current['id'], 'index': 0})['time'] == 0
        prepared = current
        source_audio = w.vocal.invoke('animation/audio', {'id': current['id']})
        assert source_audio['samples'] == 21600
        (OUT / 'ordered-source.f32').write_bytes(base64.b64decode(source_audio['base64']))
        click('#videoClose')
        w.vocal.invoke('audio/settings', {'volume': 0})
        click('#playFrames')
        until("d.querySelector('#frameStatus').textContent.includes('播放结束')", 80)
        until("d.body.dataset.posePending==='false'")
        click('#playFrames')
        until("d.querySelector('#frameStatus').textContent.includes('已复用')", 80)
        result['replay_cache'] = local("return d.querySelector('#playFrames').dataset.cache;")
        assert result['replay_cache'] == 'hit'
        click('#clearFrames')
        until("d.querySelectorAll('.pose-card').length===0")
        until("d.querySelector('#undoClearFrames').hidden===false")
        until("d.querySelector('#frameStatus').textContent.includes('已清空')")
        until_saved(lambda item: item['frames'] == [] and item['pitch_curve'] == [])
        assert saved()['frames'] == [] and saved()['pitch_curve'] == []
        assert local("return d.querySelector('#playFrames').disabled&&d.querySelector('#exportVideo').disabled;")
        try:
            w.vocal.invoke('animation/audio', {'id': prepared['id']})
        except Exception as exc:
            assert '失效' in str(exc), str(exc)
        else:
            raise AssertionError('old recording survived clear')
        click('#undoClearFrames')
        until("d.querySelectorAll('.pose-card').length===3")
        until_saved(lambda item: item['frames'] == ordered['frames'] and item['pitch_curve'] == ordered['pitch_curve'])
        assert saved()['frames'] == ordered['frames'] and saved()['pitch_curve'] == ordered['pitch_curve']
        assert local("return [...d.querySelectorAll('.pose-card')].findIndex(e=>e.classList.contains('current'));") == 1
        click('#clearFrames')
        until("d.querySelectorAll('.pose-card').length===0")
        click('#captureFrame')
        until("d.querySelectorAll('.pose-card').length===1")
        until_saved(lambda item: len(item['frames']) == 1 and item['pitch_curve'] == [])
        assert local("return d.querySelector('#undoClearFrames').hidden;")
        assert len(saved()['frames']) == 1 and saved()['pitch_curve'] == []
        result['clear_undo_new_edit'] = True
        result['passed'] = True
    except Exception:
        w.view.grab().save(str(OUT / 'failure.png'))
        raise
    finally:
        owned = w.vocal.process
        w.closing = True
        w.close()
        w.page.deleteLater()
        app.processEvents()
        if owned is not None:
            assert owned.poll() is not None, 'owned M10 worker survived window close'
            result['owned_worker_exited'] = True
        if result.get('passed'):
            (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
