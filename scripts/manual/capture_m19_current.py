"""Capture neutral, actual M19 operations in an isolated maximized Qt instance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from verification_artifacts import verification_run
from workbench_source import configure

os.environ['QT_QPA_PLATFORM'] = 'windows'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--mute-audio'
for key in ('QT_ENABLE_HIGHDPI_SCALING', 'QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS'):
    os.environ.pop(key, None)


def main(args):
    configure()
    from PyQt6 import sip
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_workspace import prepare_workspace
    native_directory = args.data_directory.resolve()

    register_scheme()
    app = QApplication(['M19 manual screenshots'])
    recipe = ('Actual Windows Qt, isolated project; light, native DPR1.5, page100, '
              'maximized with collapsed navigation; neutral phonology/solar vocabulary '
              'and explicitly illustrative custom sites; no scanned pages or physical devices')
    with verification_run('m19-manual', 'screenshots-20261024', recipe) as (out, scratch):
        print(out, flush=True)
        if not args.reader_only:
            from package_geolinguistics_data import package_data
            native_directory = scratch / 'inputs' / 'geolinguistics-data'
            data_receipt = package_data(args.data_directory, native_directory)
            original_manifest = args.data_directory / 'data-manifest.json'
            snapshot_manifest = native_directory / 'data-manifest.json'
            original = json.loads(original_manifest.read_text('utf-8'))
            snapshot = json.loads(snapshot_manifest.read_text('utf-8'))
            identity = lambda items: sorted((item['path'], item['bytes'], item['sha256']) for item in items)
            assert identity(original['files']) == identity(snapshot['files']), 'Source manifest changed or is stale'
            shutil.copyfile(original_manifest, snapshot_manifest)
        db, cache = prepare_workspace(scratch / 'state', ROOT / 'backend/migrations')
        window = Workbench(args.frontend_dist.resolve(), test=True, jobs_path=db,
                           local_files_root=cache, vocal_profile=scratch / 'vocal', start_module='M19')
        if args.reader_only:
            window.assets.geolinguistics_root = native_directory
        if not args.reader_only:
            assert not (args.frontend_dist / 'geolinguistics').exists(), 'Use a clean frontend directory without a data sidecar'
            window.assets.geolinguistics_root = None
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        window.resize(window.screen().geometry().width(), window.screen().geometry().height())
        window.showMaximized()
        window.view.setZoomFactor(1)
        window.page.setAudioMuted(True)
        report = dict(success=False, captures=[], checks=[], exports=[], scope=recipe,
                      frontend=str(args.frontend_dist.resolve()), sourceChapter='m19',
                      ownedPid=os.getpid(), physicalCompositorVerified=False)
        picker = QFileDialog.getSaveFileName
        chosen = []

        def destination(*values, **kwargs):
            path = scratch / (str(len(chosen) + 1) + '-' + Path(values[2]).name)
            chosen.append(path)
            return str(path), ''

        QFileDialog.getSaveFileName = destination

        def pause(ms=120):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            wrapped = ('(()=>{try{return {ok:true,value:eval(' + json.dumps(code) +
                       ')??null};}catch(e){return {ok:false,error:String(e)};}})()')
            loop, values = QEventLoop(), []
            window.page.runJavaScript(wrapped, lambda v: (values.append(v), loop.quit()))
            QTimer.singleShot(10000, loop.quit)
            loop.exec()
            if not values or not values[0] or not values[0].get('ok'):
                raise RuntimeError(str(values) + ' ' + code)
            return values[0].get('value')

        def until(code, seconds=45):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-2500)')))

        def click(name):
            expression = ('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&'
                          '(b.getAttribute("aria-label")===' + json.dumps(name) +
                          '||b.textContent.trim()===' + json.dumps(name) + '))')
            until('!!(' + expression + ')&&!(' + expression + ').disabled')
            js(expression + '.click()')
            pause(200)

        def fill(selector, value, event='input'):
            assert js('(()=>{const e=document.querySelector(' + json.dumps(selector) +
                      ');if(!e)return false;e.value=' + json.dumps(value) +
                      ';e.dispatchEvent(new Event(' + json.dumps(event) +
                      ',{bubbles:true}));return true})()'), selector
            pause(160)

        def capture(name):
            until('document.fonts.status==="loaded"')
            until('!document.querySelector(".map-loading")')
            assert not js('!!document.querySelector(".atlas-message.error")')
            js('document.querySelector(".atlas-message:not(.error) button")?.click()')
            pause(150)
            text = js('document.body.innerText')
            assert all(term not in text for term in ('F77_L', 'F73_L', '屄', '阴门', '脧', '𡳞'))
            pause(800)
            window.view.repaint()
            pause(160)
            image = window.view.grab().toImage()
            raw = [image.width(), image.height()]
            dpr = image.devicePixelRatio()
            assert raw in ([2560, 1600], [2561, 1601]) and abs(dpr - 1.5) < .01, (raw, dpr)
            assert window.isMaximized() and window.view.zoomFactor() == 1
            assert js('document.documentElement.dataset.theme') == 'light'
            assert js('document.querySelector(".app-shell").classList.contains("collapsed")')
            if raw != [2560, 1600]:
                image = image.copy(0, 0, 2560, 1600)
            path = out / (name + '.png')
            assert image.save(str(path))
            report['captures'].append(dict(name=name, file=path.name, image=[2560, 1600],
                raw_image=raw, dpr=dpr, pageZoom=1, maximized=True, theme='light',
                navigationCollapsed=True, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                visibleText=text))
            print('Captured ' + name, flush=True)

        def export(name, suffix):
            before = len(chosen)
            click(name)
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                if len(chosen) > before and chosen[-1].is_file() and chosen[-1].stat().st_size:
                    pause(600)
                    path = chosen[-1]
                    assert path.suffix == suffix
                    report['exports'].append(dict(name=path.name, bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    return path
                pause()
            raise AssertionError('Export missing: ' + name)

        def choose_feature(fid):
            fill('[aria-label="检索语言特征"]', '')
            click('语音')
            catalog = json.loads((native_directory / 'catalog.json').read_text('utf-8'))
            title = next(f['title'] for f in catalog['features'] if f['id'] == fid)
            fill('[aria-label="检索语言特征"]', title)
            until('document.querySelectorAll(".feature-list button").length===1')
            js('document.querySelector(".feature-list button").click()')
            pause(300)

        try:
            until('!!document.querySelector(".language-atlas")')
            click('设置')
            click('浅色')
            click('语言地图')
            click('关闭 设置')
            click('收起侧栏')
            if args.reader_only:
                chapter = json.loads((ROOT / 'manual/chapters/m19.json').read_text('utf-8'))
                project = json.loads((ROOT / 'manual/project.json').read_text('utf-8'))
                referenced = {n['attrs']['assetId'] for n in chapter['body']['content'] if n['type'] == 'image'}
                assets = [a for a in project['assets'] if a['id'] in referenced]
                assert len(assets) == 8
                packaged_project = json.loads((args.frontend_dist / 'manual/project.json').read_text('utf-8'))
                packaged_assets = {a['id']: a for a in packaged_project['assets']}
                report['byteChecks'] = []
                for asset in assets:
                    for folder in (ROOT / 'manual', ROOT / 'frontend/public/manual'):
                        assert hashlib.sha256((folder / asset['path']).read_bytes()).hexdigest() == asset['sha256']
                    packed = packaged_assets[asset['id']]
                    assert packed.get('originalSha256', packed['sha256']) == asset['sha256']
                    packaged_path = args.frontend_dist / 'manual' / packed['path']
                    assert hashlib.sha256(packaged_path.read_bytes()).hexdigest() == packed['sha256']
                    original_image = QImage(str(ROOT / 'manual' / asset['path'])).convertToFormat(QImage.Format.Format_RGBA8888)
                    packaged_image = QImage(str(packaged_path)).convertToFormat(QImage.Format.Format_RGBA8888)
                    assert not original_image.isNull() and not packaged_image.isNull()
                    assert original_image.size() == packaged_image.size()
                    pixel_sha = lambda image: hashlib.sha256(bytes(image.constBits().asstring(image.sizeInBytes()))).hexdigest()
                    assert pixel_sha(original_image) == pixel_sha(packaged_image), 'Packaged screenshot must remain pixel-identical'
                    report['byteChecks'].append(dict(id=asset['id'], sourceSha256=asset['sha256'],
                        packageSha256=packed['sha256'], packagePath=packed['path'], pixelsIdentical=True))
                click('使用方法')
                click('打开完整使用说明')
                until('document.querySelector(".manual-document")?.dataset.manualChapter==="m19"')
                assert js('document.querySelectorAll(".manual-media-image").length') == 8
                report['themes'] = []
                for theme, label in (('light', '浅色'), ('dark', '深色')):
                    if theme == 'dark':
                        click('设置')
                        click(label)
                        # Activate the existing reader tab. The sidebar help
                        # action from Settings intentionally opens its default
                        # chapter rather than preserving the M19 target.
                        assert js('!!document.querySelector("#tab-help")')
                        js('document.querySelector("#tab-help").click()')
                        until('document.querySelector(".manual-document")?.dataset.manualChapter==="m19"')
                    until('document.documentElement.dataset.theme===' + json.dumps(theme))
                    for asset in assets:
                        expression = 'document.querySelector(' + json.dumps('[data-asset-id="' + asset['id'] + '"]') + ')'
                        js(expression + '.scrollIntoView({block:"center"})')
                        until('(()=>{const i=' + expression + '.querySelector("img");return i?.complete&&i.naturalWidth===2560&&i.naturalHeight===1600})()')
                        js('window.__m19ImageDecoded=false;' + expression + '.querySelector("img").decode().then(()=>window.__m19ImageDecoded=true)')
                        until('window.__m19ImageDecoded')
                        js(expression + '.querySelector("button").click()')
                        until('document.querySelector(".manual-image-overlay img")?.naturalWidth===2560')
                        click('关闭图片')
                    assert not js('!!document.querySelector(".manual-media-error,.manual-load-error")')
                    assert js('document.documentElement.scrollWidth<=innerWidth+1')
                    window.view.grab().save(str(out / ('reader-' + theme + '.png')))
                    report['themes'].append(dict(theme=theme, decoded=8, zoomed=8, noMediaErrors=True))
                report['checks'].append('M19 help opens its stable chapter; eight source/public hashes, packaged identities and exact decoded pixels, plus light/dark image decode/zoom pass')
                report['success'] = True
                return
            until('document.querySelector(".atlas-empty")?.textContent.includes("下载地图资料")')
            if js('!!document.querySelector(".atlas-message.error")'):
                js('document.querySelector(".atlas-message.error button:last-child").click()')
            assert js('document.querySelectorAll(".atlas-empty button").length===1&&document.querySelector(".atlas-empty button").textContent.trim()==="下载资料"')
            capture('data-download')
            # Supply real File bytes to the existing directory-input handler.
            # Hidden Qt cannot activate a physical native chooser. This step
            # claims the application import/validation path, not native dialogs.
            window.assets.geolinguistics_root = native_directory
            manifest = json.loads((native_directory / 'data-manifest.json').read_text('utf-8'))
            names = ['data-manifest.json', *[entry['path'] for entry in manifest['files']]]
            js('window.__manualFolderImport={phase:"reading"};(async()=>{const dt=new DataTransfer();for(const name of ' + json.dumps(names) + '){const response=await fetch("ptbapp://app/geolinguistics/"+name);if(!response.ok)throw Error("Local file read failed: "+name);const file=new File([await response.blob()],name.split("/").at(-1));Object.defineProperty(file,"webkitRelativePath",{value:"geolinguistics-data/"+name});dt.items.add(file);}const input=document.querySelector("[aria-label=选择语言地图资料夹]");input.files=dt.files;input.dispatchEvent(new Event("change",{bubbles:true}));window.__manualFolderImport={phase:"selected",files:dt.files.length};})().catch(error=>{window.__manualFolderImport={phase:"failed",error:String(error)}})')
            until('window.__manualFolderImport.phase!=="reading"', 45)
            assert js('window.__manualFolderImport.phase') == 'selected', js('window.__manualFolderImport')
            until('!!document.querySelector(".atlas-workbench")&&document.querySelectorAll(".ol-layer canvas").length>0', 90)
            report['directoryImport'] = dict(directory=str(native_directory), files=len(names),
                method='Real bytes from task-local Qt scheme -> File input change -> original folder import and SHA/structure validation; physical native chooser not claimed')
            report['dataSnapshot'] = data_receipt
            report['dataSource'] = str(args.data_directory)
            report['dataManifestSha256'] = hashlib.sha256((native_directory / 'data-manifest.json').read_bytes()).hexdigest()
            if args.preflight_only:
                report['checks'].append('Genuine no-data empty view and real-byte file-input import of all 16 verified runtime files; no public download or physical chooser claim')
                report['success'] = True
                return

            click('方言分布')
            click('中国')
            fill('[aria-label="省级地区"]', 'province:440000', 'change')
            until('!!document.querySelector(".table-scroll tbody tr")')
            js('document.querySelector(".table-scroll tbody tr").click()')
            pause(400)
            capture('dialect-distribution')
            report['checks'].append('Genuine no-data empty view, real-byte folder import and Guangdong classification filtering with original row details; no public download, progress or physical chooser claim')

            fill('[aria-label="省级地区"]', '', 'change')
            click('语言现象')
            choose_feature('F120_P')
            click('中国')
            until('document.querySelectorAll(".legend button").length===6')
            js('document.querySelector(".legend button").click()')
            pause(400)
            assert js('document.querySelector(".legend button.chosen").textContent.includes("[-m]")')
            capture('phonology-broad-filter')
            click('全部类别')
            choose_feature('F1_P')
            fill('.map-heading label select', '2', 'change')
            fill('[aria-label="对照图 2"]', 'F120_P', 'change')
            until('document.querySelectorAll(".map-cell").length===2')
            click('中国')
            capture('two-maps')
            fill('.map-heading label select', '4', 'change')
            for number, fid in ((2, 'F117_P'), (3, 'F120_P'), (4, 'F1_L')):
                fill('[aria-label="对照图 ' + str(number) + '"]', fid, 'change')
            until('document.querySelectorAll(".map-cell").length===4')
            click('中国')
            capture('four-maps')
            report['checks'].append('Neutral nasal-tail broad filter and two/four-map views: tone counts, rounded vowels, nasal tails, sun')

            click('方言相似性')
            for site in ('S49', 'S151'):
                fill('[aria-label="检索调查点"]', site)
                until('document.querySelectorAll(".table-scroll tbody tr").length===1')
                js('document.querySelector(".table-scroll tbody tr input[type=checkbox]").click()')
            fill('[aria-label="检索调查点"]', '')
            click('比较 2–8 个地点')
            until('!!document.querySelector(".comparison-controls")')
            fill('[aria-label="比较项目范围"]', 'P', 'change')
            until('document.querySelectorAll(".comparison-scroll tbody tr").length>0')
            capture('two-place-comparison')
            comparison = export('导出对照表', '.csv')
            assert '北京' in comparison.read_text('utf-8-sig') and '广州' in comparison.read_text('utf-8-sig')
            click('关闭对话框')
            report['checks'].append('Actual Beijing/Guangzhou phonology comparison and current-scope CSV export')

            click('我的研究地图')
            fill('.atlas-catalog label input', '珠三角语音调查示例')
            js('document.querySelector(".atlas-catalog label input").dispatchEvent(new Event("change",{bubbles:true}))')
            click('输入地点坐标')
            fields = ('珠海示例点', 113.56, 22.27, 'aː˧', '手册演示，请替换为实测数据')
            for index, value in enumerate(fields, 1):
                fill('dialog .edit-grid label:nth-child(' + str(index) + ') input', value)
            capture('custom-point-editor')
            click('保存地点')
            until('document.querySelector(".table-scroll").textContent.includes("珠海示例点")')
            click('输入地点坐标')
            for index, value in enumerate(('广州示例点', 113.2643, 23.1291, 'aː˨', '手册演示，请替换为实测数据'), 1):
                fill('dialog .edit-grid label:nth-child(' + str(index) + ') input', value)
            click('保存地点')
            fill('[aria-label="省级地区"]', 'province:440000', 'change')
            click('保存到本机')
            pause(700)
            capture('custom-map-export')
            project_path = export('导出工程', '.json')
            saved = json.loads(project_path.read_text('utf-8'))
            assert len(saved['customSites']) == 2 and saved['customSites'][0]['value'] == 'aː˧'
            for label, suffix in (('导出表格', '.csv'), ('SVG', '.svg'), ('PNG', '.png')):
                path = export(label, suffix)
                if suffix == '.png':
                    assert not QImage(str(path)).isNull()
                elif suffix == '.svg':
                    assert '<svg' in path.read_text('utf-8')
                else:
                    assert '珠海示例点' in path.read_text('utf-8-sig')
            report['checks'].append('Actual illustrative IPA custom-point editing, local save and project/CSV/SVG/PNG exports')
            assert len(report['captures']) == 8
            report['success'] = True
        except Exception:
            report['error'] = traceback.format_exc()
            try:
                report['page'] = js('document.body.innerText')
                window.view.grab().save(str(out / 'failure.png'))
            except Exception:
                pass
            raise
        finally:
            source_paths = {
                'captureScript': Path(__file__),
                'frontendIndex': args.frontend_dist / 'index.html',
                'geolinguisticsPage': ROOT / 'frontend/src/modules/geolinguistics/GeolinguisticsPage.vue',
                'desktopHost': ROOT / 'desktop/src/ptb_desktop/host.py',
                'desktopPresentation': ROOT / 'desktop/src/ptb_desktop/presentation.py',
                'dataManifest': args.data_directory / 'data-manifest.json',
            }
            report['sources'] = {role: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                 for role, path in source_paths.items()}
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            QFileDialog.getSaveFileName = picker
            window.closing = True
            window.close()
            app.processEvents()
            sip.delete(window)
            app.processEvents()
            sip.delete(app)
        print(json.dumps(dict(success=True, out=str(out), captures=len(report['captures']))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frontend-dist', type=Path, default=ROOT / 'frontend/dist')
    parser.add_argument('--data-directory', type=Path, default=ROOT / 'local-data/geolinguistics')
    parser.add_argument('--reader-only', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    args.frontend_dist = args.frontend_dist.resolve()
    args.data_directory = args.data_directory.resolve()
    if args.child:
        main(args)
    else:
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8', __file__, *sys.argv[1:], '--child'],
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        print(result.stdout + result.stderr, flush=True)
        prefix = str(ROOT / 'output/validation/m19-manual/screenshots-20261024-')
        runs = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip().startswith(prefix)]
        if runs:
            # verification_run has already attempted cleanup once in the child.
            # Retain a refusal receipt without retrying the same directory here.
            receipt = runs[0] / 'cleanup.json'
            status = json.loads(receipt.read_text('utf-8-sig')) if receipt.is_file() else {'status': 'missing'}
            (runs[0] / 'console.log').write_text(result.stdout + result.stderr, encoding='utf-8')
            (runs[0] / 'child-exit.json').write_text(json.dumps({'returncode': result.returncode,
                'cleanupStatus': status['status']}, indent=2) + '\n', encoding='utf-8')
            if status['status'] != 'deleted':
                raise RuntimeError('Owned M19 screenshot scratch retained; see cleanup receipt')
        raise SystemExit(result.returncode)
