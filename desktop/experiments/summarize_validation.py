"""Summarize final P01 artifact evidence and update only its task ledger entry."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'output/validation/p01'

if __name__ == '__main__':
    build = json.loads((OUTPUT / 'build-report.json').read_text('utf-8'))
    artifact = ROOT / 'output/p01-probe/PhoneticToolbox-P01.exe'
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert digest == build['exe_sha256']
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == value for p, value in build['input_sha256'].items())
    cases = {}
    for name in ['final-unicode', 'final-125', 'final-dual-200']:
        runtime = json.loads((OUTPUT / name / 'runtime-report.json').read_text('utf-8'))
        assert runtime['success'], name
        measured = []
        for instance in range(runtime['instances']):
            host = json.loads((OUTPUT / name / f'instance-{instance}/host-report.json').read_text('utf-8'))
            assert host['success'] and host['frozen']
            canvas = host['layouts'][0]['snapshot']['canvases'][0]
            measured.append(round(canvas['pixels'] / canvas['width'], 4))
        cases[name] = {'success': True, 'instances': runtime['instances'],
                       'observed_pixel_ratios': measured, 'wall_seconds_including_test': runtime['wall_seconds'],
                       'tracked_processes': runtime['tracked_process_count'],
                       'owned_processes_remaining': runtime['owned_processes_remaining'],
                       'onefile_temp_cleaned': runtime['onefile_temp_cleaned'],
                       'digital_loopback_verified': runtime.get('loopback', {}).get('test_tones_detected')}
    preservation = json.loads((OUTPUT / 'preservation.json').read_text('utf-8'))
    assert all(preservation.values())
    summary = {'task': 'P01', 'task_status': 'in_progress', 'windows_probe_status': 'verified',
        'artifact': str(artifact.relative_to(ROOT)), 'bytes': artifact.stat().st_size,
        'sha256': digest, 'cases': cases, 'preservation': preservation,
        'remaining': ['User hands-on trial', 'Complete redistribution strategy and historical source permissions',
                      'Mac/Linux native build and device evidence; no full-stack freeze', 'P03 scientific baseline is not part of this prototype']}
    (OUTPUT / 'final-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    ledger_path = ROOT / 'docs/plans/task-ledger.json'
    ledger = json.loads(ledger_path.read_text('utf-8'))
    task = next(t for t in ledger['tasks'] if t['id'] == 'P01')
    task.update(status='in_progress', note='Windows 原型已验证；整体冻结保留来源/发行策略与跨平台原生未决项。',
                execution_evidence=['docs/testing/p01-host-probe-report.md', 'desktop/experiments/README.md',
                                    'output/validation/p01/final-summary.json'],
                acceptance_status={'windows_host_and_onefile': 'verified', 'digital_audio_output': 'verified',
                                   'source_and_distribution_gate': 'blocked', 'mac_linux_native_validation': 'planned'})
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'output/p01-probe/SHA256.txt').write_text(digest + '  PhoneticToolbox-P01.exe\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
