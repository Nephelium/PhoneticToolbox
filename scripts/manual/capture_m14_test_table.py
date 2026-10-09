"""Capture the M14 manual using an untouched author-provided XLSX.

Owns a disposable Qt profile and a copy of the existing task schema. Uses the
user-authorized 2560 x 1440 fallback when the physical maximized viewport is
smaller. No product behavior, source table, existing profile or schema changes.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
DIRS = ('desktop/src', 'backend/src', 'packages/phonetic_core/src', 'scripts')
for directory in DIRS:
    sys.path.insert(0, str(ROOT / directory))
os.environ['PYTHONPATH'] = os.pathsep.join(str(ROOT / p) for p in DIRS)
os.environ.setdefault('QT_QPA_PLATFORM', 'windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --disable-gpu-compositing --mute-audio')

TONES = {'1': '阴平（暂拟）', '2': '阳平（暂拟）', '5': '阴去（暂拟）',
         '6': '阳去（暂拟）', '7': '阴入（暂拟）', '8': '阳入（暂拟）'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from openpyxl import load_workbook
    from docx import Document
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.m14_table import load_v2
    from phonetic_core.transcription.phonology.rules import PhonologyRules
    from verify_m14_wiring import setup

    source = args.source.resolve(strict=True)
    original_hash = sha(source)
    out = ROOT / 'output/manual-work/m14-test-table' / uuid4().hex
    out.mkdir(parents=True)
    inputs, screenshots, saved, retry = [out / n for n in ('inputs', 'screenshots', 'saved', 'retry')]
    for directory in (inputs, screenshots, saved, retry):
        directory.mkdir()
    table = inputs / source.name
    shutil.copyfile(source, table)
    raw_rows = list(load_workbook(table, read_only=True, data_only=True).active.values)
    options = dict(start_row=1, character_column=1, ipa_column=2, note_column=3, table_index=0)
    rows, diagnostics = load_v2(table.read_bytes(), table.name, options)
    analysis = PhonologyRules('m14/2').analyze(rows)
    assert len(rows) == 5453 and not diagnostics['skipped'] and not diagnostics['warnings']
    assert analysis.unique_tones == list(TONES)
    reordered = inputs / '测试_列映射演示.csv'
    with reordered.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['由测试.xlsx派生的列映射操作副本', '', '', ''])
        writer.writerow(['编号', 'IPA', '备注', '字头'])
        for i, (character, ipa, note) in enumerate(raw_rows, 1):
            writer.writerow([i, ipa, note or '', character])
    text_file = inputs / '测试_分号摘录.txt'
    with text_file.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter=';')
        writer.writerow(['字头', 'IPA', '备注'])
        writer.writerows(raw_rows[:3])
    input_hashes = {p.name: sha(p) for p in inputs.iterdir()}
    baseline = {'chapters': {p.name: sha(p) for p in (ROOT / 'manual/chapters').glob('*.json')},
                'project': sha(ROOT / 'manual/project.json'),
                'product': {p.relative_to(ROOT).as_posix(): sha(p)
                            for folder in ('frontend/src', 'desktop/src', 'backend/src', 'packages/phonetic_core/src')
                            for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.ts', '.vue')}}
    (out / 'baseline.json').write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding='utf-8')
    task_out, db, cache = setup()
    register_scheme()
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.Floor)
    app = QApplication(['PTB-owned-M14-test-table-manual'])
    window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache,
                       vocal_profile=out / 'vocal-profile')
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.showMaximized()
    window.page.setAudioMuted(True)
    destination = {'path': saved}
    QFileDialog.getExistingDirectory = lambda *a, **k: str(destination['path'])
    report = dict(success=False, out=str(out), source=str(source), originalSHA256=original_hash,
                  sourceCopy=str(table), inputs=input_hashes, toneMap=TONES,
                  sourceSummary=dict(rows=len(rows), uniqueIPA=len(analysis.unique_ipa),
                                     initials=len(analysis.unique_initials), finals=len(analysis.unique_finals),
                                     toneCounts=dict(Counter(r.tone_value for r in analysis.rows)),
                                     duplicates=diagnostics['duplicate_rows'], skipped=diagnostics['skipped']),
                  ownedTaskState=str(task_out), captures=[], checks=[], errors=[], exports=[], terminations=[],
                  scope='Actual hidden Windows Qt workbench; complete application client window; author-authorized 2560x1440 fallback; demonstration tone names, no linguistic validation')
    window.page.renderProcessTerminated.connect(lambda s, c: report['terminations'].append([s.name, c]))

    def pause(ms=100):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def js(code):
        loop, values = QEventLoop(), []
        window.page.runJavaScript(code, lambda v: (values.append(v), loop.quit()))
        QTimer.singleShot(8000, loop.quit)
        loop.exec()
        if not values:
            raise RuntimeError('JavaScript timeout')
        return values[0]

    def until(code, seconds=90):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js(code):
                return
            pause()
        raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-4000)')))

    def click(label):
        expr = '[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")||b.textContent.trim())===' + json.dumps(label) + ')'
        until('!!(' + expr + ')&&!(' + expr + ').disabled')
        js(expr + '.click()')
        pause(120)

    def select(label, value):
        assert js('(()=>{const e=[...document.querySelectorAll("select")].find(e=>e.offsetParent&&(e.getAttribute("aria-label")===' + json.dumps(label) + '||e.closest("label")?.textContent.trim().startsWith(' + json.dumps(label) + ')));if(!e||![...e.options].some(o=>o.value===' + json.dumps(str(value)) + '))return false;e.value=' + json.dumps(str(value)) + ';e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), label
        pause(150)

    def fill(label, value):
        assert js('(()=>{const e=document.querySelector(' + json.dumps('input[aria-label="' + label + '"]') + ');if(!e)return false;e.value=' + json.dumps(str(value)) + ';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), label
        pause(150)

    def idle():
        until('document.querySelector(".phonology-page")?.getAttribute("aria-busy")==="false"')

    def upload(path):
        raw = base64.b64encode(path.read_bytes()).decode()
        js('(()=>{const d=new DataTransfer();d.items.add(new File([Uint8Array.from(atob(' + json.dumps(raw) + '),c=>c.charCodeAt(0))],' + json.dumps(path.name) + '));const e=document.querySelector(".phonology-page input[type=file]");e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        idle()
        until('!!document.querySelector("table[aria-label=原始表格样例]")')

    def symbol(value, ctrl=False):
        assert js('(()=>{const e=[...document.querySelectorAll("[aria-label=声母列表] [role=option]")].find(e=>e.textContent===' + json.dumps(value) + ');if(!e)return false;e.dispatchEvent(new MouseEvent("click",{bubbles:true,ctrlKey:' + str(ctrl).lower() + '}));return true})()'), value
        pause()

    def snapshot(key, figure, caption):
        assert [window.width(), window.height()] == [2560, 1440]
        pause(450)
        window.repaint()
        window.grab()
        pause(450)
        pix = window.grab()
        assert [pix.width(), pix.height()] == [2560, 1440], [pix.width(), pix.height()]
        file = screenshots / ('m14-test-' + key + '-20261006-light.png')
        assert pix.save(str(file))
        report['captures'].append(dict(id=file.stem, chapterId='m14', figureId=figure, file=str(file),
                                       caption=caption, maximized=window.isMaximized(), captureMode='user-authorized-2560x1440-window',
                                       window=[window.width(), window.height()], image=[pix.width(), pix.height()],
                                       viewport=js('[innerWidth,innerHeight]'), devicePixelRatio=pix.devicePixelRatio(),
                                       theme=js('document.documentElement.dataset.theme'), sha256=sha(file)))
        print('Captured ' + key, flush=True)

    try:
        until('!!document.querySelector("nav")&&document.fonts.status==="loaded"')
        report['physicalMaximizedWindow'] = [window.width(), window.height()]
        window.showNormal()
        window.setFixedSize(2560, 1440)
        pause(500)
        click('设置')
        click('浅色')
        until('document.documentElement.dataset.theme==="light"')
        click('音系归纳')
        until('!!document.querySelector(".phonology-page")')
        snapshot('overview', 'm14-figure-m14-overview-r2-light', '2560×1440 完整应用中的音系归纳空态，保留导航、标签栏与四步工作区。')
        upload(table)
        fill('数据开始行', 1)
        idle()
        snapshot('import', 'm14-figure-m14-import-detail-r4-light', '测试.xlsx 的 Sheet1 有 5,453 行、3 列，无表头，从第 1 行读取；字头、IPA、备注分别映射第 1、2、3 列。')
        select('字头列', 2)
        until('document.querySelector(".import-card [role=alert]")?.textContent.includes("不同")')
        snapshot('column-conflict', 'm14-figure-m14-column-conflict-r5-light', '测试.xlsx 的字头列与 IPA 列均选第 2 列时，页面提示须选择不同的有效列。')
        select('字头列', 1)
        click('确认导入并继续')
        idle()
        until('document.body.innerText.includes("已读取 5453 条记录")')
        snapshot('tone-original', 'm14-figure-m14-tone-detail-r4-light', '全表导入后显示六个原始数字编码 1、2、5、6、7、8，记录数与例字来自测试.xlsx。')
        for code, name in TONES.items():
            fill('调类 ' + code, name)
        snapshot('tone-edited', 'm14-figure-m14-tone-edited-r5-light', '六个编码暂拟阴平、阳平、阴去、阳去、阴入、阳入，各名称标注暂拟；本例仅展示操作，不代表真实调类。')
        click('保存调类设置并继续')
        snapshot('symbols', 'm14-figure-m14-symbol-detail-r4-light', '测试.xlsx 解析出 34 个声母与 44 个韵母候选，完整应用窗口中并列核对。')
        symbol('ph')
        symbol('pʰ', True)
        snapshot('multiselect', 'm14-figure-m14-symbol-multiselect-r5-light', '实际表中 ph 与 pʰ 两个声母候选已多选；排序可多选，归并前须单选源项。')
        symbol('ph')
        click('归并声母')
        select('归并目标', 'pʰ')
        until('document.querySelector("[aria-label=归并确认]")?.textContent.includes("影响 1 条记录")')
        snapshot('merge-confirm', 'm14-figure-m14-merge-confirm-r5-light', '仅为展示，把 ph 暂指向 pʰ，确认区显示影响 1 条记录：捧 phoŋ1。归并依据尚未核实。')
        click('确认归并')
        snapshot('merge-applied', 'm14-figure-m14-merge-applied-r5-light', '操作演示确认后 ph → pʰ 显示在下方，声母暂为 33 项；本例随后撤销，最终结果保留原分类。')
        click('撤销归并 ph')
        click('保存声韵设置并继续')
        click('生成三份结果')
        idle()
        until('document.body.innerText.includes("三份结果已完整生成")')
        snapshot('result', 'm14-figure-m14-result-detail-r4-light', '测试.xlsx 全部 5,453 条记录生成两份同音字表 DOCX 与一份二维 XLSX，右栏可整组保存。')
        js('[...document.querySelectorAll("details")].filter(e=>e.querySelector("summary")?.textContent.includes("声母、韵母与声调统计")).forEach(e=>e.open=true)')
        snapshot('initial', 'm14-figure-m14-initial-preview-r5-light', '声母到韵母预览展开统计，34 声母、44 韵母与六个暂拟调类均来自同次全表生成；正文按每页 100 条浏览。')
        click('韵母 → 声母DOCX')
        snapshot('final', 'm14-figure-m14-final-preview-r5-light', '同次生成的韵母到声母预览，完整大窗口中按韵母查看声母与例字，调类名称仅作展示。')
        click('二维声韵表XLSX')
        snapshot('matrix', 'm14-figure-m14-matrix-preview-r5-light', '二维表以韵母为行、声母为列，在完整 2560×1440 应用中查看首组 12 行×10 列；其余类别通过分组按钮与滚动查看。')
        fill('搜索字头或 IPA', 'phoŋ1')
        js('[...document.querySelectorAll("details")].filter(e=>e.querySelector("summary")?.textContent.includes("导入与归并审阅")).forEach(e=>e.open=true)')
        until('document.querySelector(".review")?.textContent.includes("phoŋ1")')
        snapshot('review', 'm14-figure-m14-merged-review-r5-light', '搜索原始 phoŋ1 定位捧并展开审阅，原始字音、来源行与 ph 分类保留；此前演示的归并已撤销。')
        fill('搜索字头或 IPA', '')
        js('[...document.querySelectorAll("details.review")].forEach(e=>e.open=false)')
        click('选择目录保存三份结果')
        idle()
        until('document.body.innerText.includes("三份结果已保存")')
        assert len(list(saved.iterdir())) == 3
        export_hashes = {p.name: sha(p) for p in saved.iterdir()}
        for path in saved.iterdir():
            if path.suffix == '.docx':
                content = '\n'.join(p.text for p in Document(path).paragraphs)
            else:
                sheet = load_workbook(path, rich_text=True).active
                assert sheet.title == '二维同音字表' and sheet.freeze_panes == 'B2'
                content = '\n'.join(str(c.value) for row in sheet for c in row if c.value is not None)
            assert all(r.character in content and (not r.note or r.note in content) for r in rows), path
            assert all(name in content for name in TONES.values()), path
            report['exports'].append(dict(file=str(path), size=path.stat().st_size, sha256=sha(path)))
        click('选择目录保存三份结果')
        idle()
        until('document.body.innerText.includes("同名结果")')
        assert export_hashes == {p.name: sha(p) for p in saved.iterdir()}
        snapshot('save-conflict', 'm14-figure-m14-save-conflict-r5-light', '再次保存到已有同名成果的目录时拒绝覆盖，三份原文件保持不变，可重新选择目录。')
        destination['path'] = retry
        click('选择目录保存三份结果')
        idle()
        until('document.body.innerText.includes("三份结果已保存")')
        assert export_hashes == {p.name: sha(p) for p in retry.iterdir()}
        click('1 · 导入')
        select('文本编码', 'utf-8-sig')
        select('分隔符', 'comma')
        upload(reordered)
        select('字头列', 4)
        select('IPA 列', 2)
        select('备注列', 3)
        fill('数据开始行', 3)
        idle()
        snapshot('selected-columns', 'm14-figure-m14-selected-columns-r5-light', '由测试.xlsx 全量派生的 UTF-8 列映射副本有 5,455 行、4 列；开始行 3，字头、IPA、备注分别映射第 4、2、3 列。')
        click('取消此次选择')
        click('1 · 导入')
        select('文本编码', 'utf-8-sig')
        select('分隔符', 'semicolon')
        upload(text_file)
        select('字头列', 1)
        select('IPA 列', 2)
        select('备注列', 3)
        fill('数据开始行', 2)
        idle()
        snapshot('semicolon', 'm14-figure-m14-gbk-delimiter-r5-light', '测试.xlsx 前三行的 UTF-8 分号摘录保留知、蜘、䵹及原 IPA，新增表头后从第 2 行读取。')
        click('取消此次选择')
        assert original_hash == sha(source) == sha(table)
        assert input_hashes == {p.name: sha(p) for p in inputs.iterdir()}
        assert not report['terminations'] and len(report['captures']) == 17
        report['checks'] = ['5453 accepted, 11 duplicates retained, no skipped rows or warnings',
                            'All 17 chapter states captured as complete 2560x1440 application client windows',
                            'Demonstration ph->pʰ merge affects one source record and is undone before final generation',
                            'All source characters and notes present in each DOCX/DOCX/XLSX readback; six provisional names verified',
                            'Save collision preserves SHA; retry files identical; source and copies unchanged']
        report['success'] = True
    except Exception:
        report['errors'].append(traceback.format_exc())
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        window.close()
        pause(300)
        app.quit()
        print(json.dumps({'success': report['success'], 'report': str(out / 'report.json'), 'captures': len(report['captures']), 'errors': report['errors']}, ensure_ascii=False), flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
