"""Capture unobscured M01 role choices and a clean native REAPER dialog."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from manual.capture_a_support import Capture
from verification_artifacts import verification_run


def run(args, out, scratch):
    capture = Capture(args, out, scratch, 'm01', 'parameter-estimation',
                      reaper_binary=ROOT / 'phonetic_toolbox/core/acoustic/reaper.exe')
    from PyQt6.QtWidgets import QFileDialog
    config = json.loads((ROOT / 'local-data/manual-authoring/sources.json').read_text(encoding='utf-8'))
    source = Path(config['egg'])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    audio = scratch / 'audio'
    audio.mkdir()
    shutil.copyfile(source, audio / 'egg-stereo.wav')
    QFileDialog.getExistingDirectory = lambda *a, **k: str(audio)

    def aria(chinese):
        return '[aria-label=' + json.dumps(capture.ui.label(chinese), ensure_ascii=False) + ']'

    try:
        capture.until('!!document.querySelector(".m01-page")')
        capture.prepare()
        capture.click('编辑14项设置')
        assert capture.js('document.querySelector("dialog[open] .parameter-toolbar button:nth-child(2)").click();true')
        capture.pause()
        assert capture.js('document.querySelectorAll(".m01-settings input[type=checkbox]").length') == 2
        assert capture.js('document.querySelector("dialog[open] [role=alert]")===null')
        capture.shot('m01-reaper-clean', {
            'settings': '.m01-settings',
            'hilbert': '.m01-settings .setting-row:first-child',
            'highpass': '.m01-settings .setting-row:last-child',
        })
        capture.click('取消')
        capture.click('打开音频目录')
        capture.until('document.querySelectorAll(".m01-file-entry").length===1')
        assert capture.js('document.querySelector(".m01-file-list button").click();true')
        capture.until('!!document.querySelector(".m01-page .wave-track svg")&&!document.querySelector(".m01-page .audio-loading")')
        label = capture.ui.label('联合计算 EGG')
        assert capture.js('(()=>{const e=[...document.querySelectorAll("label")].find(e=>e.textContent.includes(' + json.dumps(label, ensure_ascii=False) + '))?.querySelector("input[type=checkbox]");if(!e)return false;if(!e.checked)e.click();return true})()')
        capture.fill('select' + aria('默认分析声道'), '1', 'change')
        capture.fill('select' + aria('EGG参数保存方式'), 'cycles', 'change')
        capture.js('document.querySelector(".m01-joint").scrollIntoView({block:"start"})')
        capture.pause()
        capture.shot('m01-egg-roles-detail-parent', {
            'channel': '.m01-joint>label:first-of-type',
            'storage': '.m01-joint>label:has(select' + aria('EGG参数保存方式') + ')',
        })
        capture.report['captures'][-1]['auxiliary'] = True
        capture.report.update(success=True, originalSourceHashes={str(source): digest},
                              originalSourcesUnchanged=hashlib.sha256(source.read_bytes()).hexdigest() == digest,
                              checks=['Fresh real REAPER dialog has both options and no residual validation error',
                                      'Authorized stereo sample loaded; real default audio role and independent raw-cycle storage choices fully visible; no scientific job submitted'])
        assert capture.report['originalSourcesUnchanged']
    finally:
        capture.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--locale', choices=('en', 'ja'), required=True)
    parser.add_argument('--frontend-dist', type=Path, default=ROOT / 'output/validation/app-i18n-root/m01-capture-build')
    args = parser.parse_args()
    args.frontend_dist = args.frontend_dist.resolve(strict=True)
    with verification_run('m01-native-details', args.locale, 'Real native Windows/M01 controls; authorized EGG copy; no original/profile writes') as (out, scratch):
        run(args, out, scratch)
