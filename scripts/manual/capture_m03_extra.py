"""Supplementary real M03 states; called within the owned R8 capture host."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(*, ui, js, until, pause, click, idle, capture, report, out, scratch,
        source_sha, folder_choice, database, window, selected=()):
    """Exercise real public controls, native tasks and save operations."""
    def key(name, values=None):
        return ui.key('modules.egg-analysis.' + name, values)

    def button(name):
        text = key(name)
        assert js("(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.offsetParent&&e.textContent.trim()===" + json.dumps(text) + ");if(!e||e.disabled)return false;e.click();return true})()"), text
        pause()

    def shot(suffix, *, ready=True):
        if selected and not any(suffix.startswith(scene + '-') for scene in selected):
            return
        capture('m03-' + suffix, ready=ready)
        report.setdefault('visibleStates', {})[suffix] = js("(document.querySelector('dialog[open]')||document.querySelector('.egg-page'))?.innerText")

    def range_(start, end):
        assert js("(()=>{const a=document.querySelectorAll('.egg-bottom-bar .selection-controls input');if(a.length!==2)return false;for(const[i,v]of [[1," + str(end) + "],[0," + str(start) + "]]){a[i].value=v;a[i].dispatchEvent(new Event('input',{bubbles:true}));a[i].dispatchEvent(new Event('change',{bubbles:true}))}return true})()")
        idle()

    def close_result():
        button('back_to_analysis')
        until("!document.querySelector('dialog[open]')")

    def save_result(kind):
        saved = scratch / ('saved-' + kind)
        saved.mkdir()
        folder_choice['path'] = saved
        button('select_folder_to_save_complete_results')
        until("[...document.querySelectorAll('dialog[open] [role=status]')].some(e=>e.textContent.trim().length>0)", 90)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not list(saved.glob('*.ptb.json')):
            pause(150)
        assert list(saved.glob('*.ptb.json')), 'Native result save did not finish'
        until("![...document.querySelectorAll('dialog[open] button')].find(e=>e.textContent.trim()===" + json.dumps(key('select_folder_to_save_complete_results')) + ")?.disabled")
        report.setdefault('savedFiles', {})[kind] = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(saved.iterdir()) if p.is_file()]

    button('swap_channels')
    idle()
    shot('channel-swap-light-scaled-r20261007')
    button('swap_channels')
    idle()
    assert js("(()=>{const e=document.querySelector('.egg-overview input[type=range]');if(!e)return false;e.value=e.max;e.dispatchEvent(new Event('input',{bubbles:true}));return true})()")
    shot('overview-tail-light-scaled-r20261007')
    assert js("(()=>{const e=document.querySelector('.egg-overview input[type=range]');e.value=0;e.dispatchEvent(new Event('input',{bubbles:true}));return true})()")

    button('batch_analysis')
    until("!!document.querySelector('dialog[open] .batch-settings')")
    shot('batch-parameters-light-scaled-r20261007-v2-57bdee21')
    assert js("(()=>{const e=document.querySelector('.batch-values input');e.value=2500;e.dispatchEvent(new Event('input',{bubbles:true}));return true})()")
    button('submit_selected_files')
    until("!!document.querySelector('dialog[open] [role=alert]')")
    shot('batch-filter-error-light-scaled-r20261007', ready=False)
    assert js("(()=>{const e=document.querySelector('.batch-values input');e.value=25;e.dispatchEvent(new Event('input',{bubbles:true}));return true})()")
    button('back_to_workbench')
    until("!document.querySelector('dialog[open]')")

    range_(40, 51)
    button('inverse_filtered_if')
    until("!!document.querySelector('.egg-page [role=alert]')")
    shot('inverse-budget-error-light-scaled-r20261007', ready=False)
    range_(40, 40.5)
    button('inverse_filtered_if')
    until("document.querySelectorAll('.inverse-grid .scientific-plot>svg').length===4", 180)
    until("document.querySelectorAll('dialog[open] .audio-transport').length===2", 60)
    shot('inverse-result-light-scaled-r20261007-v2-d5faeb4b')
    save_result('inverse')
    shot('inverse-saved-light-scaled-r20261007-v2-89edec96')
    if not selected:
        scientific_export(ui, js, until, pause, report, out, scratch, window)
    for index, suffix, count in [(2, 'inverse-egg-four-light-scaled-r20261007', 4), (3, 'inverse-two-light-scaled-r20261007', 2)]:
        assert js("(()=>{const e=document.querySelectorAll('.inverse-layouts button')[" + str(index) + "];if(!e)return false;e.click();return true})()")
        until("document.querySelectorAll('.inverse-grid .scientific-plot>svg').length===" + str(count))
        shot(suffix)
    assert js("(()=>{const e=document.querySelector('.egg-wave-options input');if(!e)return false;if(!e.checked)e.click();return true})()")
    until("document.querySelectorAll('dialog[open] section .wave-track').length>=3")
    assert js("(()=>{const e=document.querySelector('dialog[open] .dialog-body>section:last-of-type');if(!e)return false;e.scrollIntoView({block:'end'});return true})()")
    until("(()=>{const e=document.querySelector('dialog[open] .dialog-body>section:last-of-type'),r=e?.getBoundingClientRect(),d=document.querySelector('dialog[open]')?.getBoundingClientRect();return !!r&&!!d&&r.top>=d.top&&r.bottom<=d.bottom})()")
    shot('inverse-full-egg-light-scaled-r20261007')
    close_result()

    button('save_csv_three_plots')
    until("[...document.querySelectorAll('.egg-result-links button')].length>=2", 180)
    assert js("(()=>{const e=document.querySelector('.egg-result-links button');if(!e)return false;e.click();return true})()")
    until("document.querySelectorAll('dialog[open] .egg-export-image').length===3", 90)
    save_result('single')
    shot('single-saved-light-scaled-r20261007')
    close_result()
    button('batch_analysis')
    until("!!document.querySelector('dialog[open] .batch-settings')")
    button('submit_selected_files')
    until("!document.querySelector('dialog[open]')", 90)
    batch_prefix = key('saveBatchCount', {'count': 1})
    until("[...document.querySelectorAll('.egg-actions button')].some(e=>e.textContent.startsWith(" + json.dumps(batch_prefix) + ")&&!e.disabled)", 240)
    saved = scratch / 'saved-batch'
    saved.mkdir()
    folder_choice['path'] = saved
    assert js("(()=>{const e=[...document.querySelectorAll('.egg-actions button')].find(e=>e.textContent.startsWith(" + json.dumps(batch_prefix) + "));if(!e||e.disabled)return false;e.click();return true})()")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and not list(saved.rglob('*.ptb.json')):
        pause(150)
    assert list(saved.rglob('*.ptb.json')), 'Batch result save did not finish'
    until("![...document.querySelectorAll('.egg-actions button')].find(e=>e.textContent.startsWith(" + json.dumps(batch_prefix) + "))?.disabled")
    shot('batch-saved-light-scaled-r20261007')
    report['savedFiles']['batch'] = [{'name': p.relative_to(saved).as_posix(), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(saved.rglob('*')) if p.is_file()]
    with sqlite3.connect(database) as connection:
        report['scientificTaskSnapshots'] = [json.loads(row[0]) for row in connection.execute("SELECT snapshot FROM jobs WHERE state='succeeded' ORDER BY created_at")]
    connection.close()
    report['sameAuthorisedRecording'] = source_sha


def scientific_export(ui, js, until, pause, report, out, scratch, window):
    """Re-export the same legacy numeric result, verifying the original metadata."""
    from PyQt6.QtGui import QImage
    from manual.localization import fingerprint
    baseline_path = ROOT / 'output/manual-work/m03-captures/27216cb27c2c46f780b2711b0910c6e1/saved/3_40_00s_40_50s.ptb.json'
    baseline = json.loads(baseline_path.read_text('utf-8'))
    current_path, = (scratch / 'saved-inverse').glob('*.ptb.json')
    current = json.loads(current_path.read_text('utf-8'))
    scientific_fields = ['mode', 'micro_center', 'micro_width_ms', 'flip_channels', 'signal_mode', 'roi_start', 'roi_end',
                         'gci_method', 'goi_method', 'peak_prominence', 'valley_prominence', 'auto_prominence',
                         'highpass_cutoff', 'lowpass_cutoff', 'lp_order', 'export_policy']

    def inputs(data):
        return {key: data[key] for key in ['input_sha256', 'sample_rate_hz', 'sample_count', 'selection', 'channel_roles', 'normalized_peak']}

    def parameters(data):
        return {'layout': 'separate', 'resolutionDpi': 300, 'centerWindowMs': 50,
                'config': {key: data['config'][key] for key in scientific_fields},
                'inverseViewSha256': fingerprint(data['inverse_view'])}

    actual_input, original_input = inputs(current), inputs(baseline)
    actual_parameters, original_parameters = parameters(current), parameters(baseline)
    assert actual_input == original_input, 'Scientific input/selection differs from the original export'
    assert actual_parameters == original_parameters, 'Scientific parameters or any inverse-view numeric array differs from original'
    assert js("(()=>{const e=document.querySelector('.inverse-layouts button');if(!e)return false;e.click();return true})()")
    until("document.querySelectorAll('.inverse-grid .scientific-plot>svg').length===6")
    target = out / 'inverse-six-export.png'
    downloads = []
    window.profile.downloadRequested.disconnect()

    def download(item):
        item.setDownloadDirectory(str(out))
        item.setDownloadFileName(target.name)
        item.accept()
        downloads.append(item)

    window.profile.downloadRequested.connect(download)
    assert js("(()=>{const e=document.querySelector('.inverse-actions>button');if(!e||e.disabled)return false;e.click();return true})()")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and not (downloads and downloads[0].isFinished()):
        pause(150)
    assert downloads and downloads[0].isFinished() and target.is_file(), 'Actual six-panel PNG download did not finish'
    image = QImage(str(target))
    assert not image.isNull()
    asset_id = 'm03-inverse-six-export-light'
    asset = next(item for item in json.loads((ROOT / 'manual/project.json').read_text('utf-8'))['assets'] if item['id'] == asset_id)
    source_sha = hashlib.sha256((ROOT / 'manual' / asset['path']).read_bytes()).hexdigest()
    metrics = ui.metrics(js)
    entry = {key: metrics[key] for key in ['locale', 'fontLocale', 'fontFamily', 'dpr', 'pageZoom', 'theme']}
    entry.update(name='inverse-six-export', assetId=asset_id, file=target.name, image=[image.width(), image.height()],
                 sha256=hashlib.sha256(target.read_bytes()).hexdigest(), maximized=window.isMaximized(), captureKind='scientific-export',
                 scientificExport={'schemaVersion': 'ptb-manual-scientific-export/1', 'sourceAssetId': asset_id,
                                   'sourceImageSha256': source_sha, 'input': actual_input, 'parameters': actual_parameters,
                                   'inputSha256': fingerprint(actual_input), 'parametersSha256': fingerprint(actual_parameters),
                                   'baseline': {'input': original_input, 'parameters': original_parameters,
                                                'evidence': baseline_path.relative_to(ROOT).as_posix() + '#sha256=' + hashlib.sha256(baseline_path.read_bytes()).hexdigest()},
                                   'action': '.inverse-layouts button:first-child then .inverse-actions>button: first actual PNG download'})
    report['captures'].append(entry)
    print('Exported actual six-panel scientific PNG', flush=True)
