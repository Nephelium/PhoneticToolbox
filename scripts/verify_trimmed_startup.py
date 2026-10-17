"""Alternate three real cold launches per build, then warm/reuse launches.

Offscreen/software rendering measures readiness only. Visible default-GPU
acceptance is a separate final-package gate. OS caches are not cleared.
"""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from verification_artifacts import verification_run, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    args = parser.parse_args()
    from source_snapshot import identity
    recipe = dict(command=sys.argv, before=identity(args.before), after=identity(args.after),
                  scope=__doc__, osCacheCleared=False)
    with verification_run('package-trim', 'startup', recipe) as (out, scratch):
        print(json.dumps({'evidence': str(out)}), flush=True)
        rows = []
        node = shutil.which('node')

        def run(label, exe, count=1, profile=None):
            target = scratch / label
            command = [node, str(ROOT / 'scripts/measure_frozen_startup.mjs'), str(exe.resolve()), str(target), str(count)]
            if profile:
                command.append(str(profile))
            print('Starting ' + label, flush=True)
            with (out / (label + '.log')).open('wb') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=600, creationflags=subprocess.CREATE_NO_WINDOW)
            if (target / 'results.json').exists():
                data = json.loads((target / 'results.json').read_text('utf8'))
                write_json(out / (label + '.json'), data)
            if result.returncode:
                raise RuntimeError(label + ' failed; see retained log')
            print(json.dumps({'phase': label, 'readyMs': [r['readyMs'] for r in data['rows']]}), flush=True)
            return data

        for iteration in range(1, 4):
            for variant in (('before', 'after') if iteration % 2 else ('after', 'before')):
                data = run(variant + '-' + str(iteration), getattr(args, variant))
                rows.append(dict(variant=variant, iteration=iteration, result=data['rows'][0]))
        warm = run('after-warm', args.after, 3, scratch / 'after-3/run-1/profile')
        cross = run('before-to-after', args.after, 1, scratch / 'before-3/run-1/profile')
        for row in warm['rows']:
            if set(row['persistentCache']['reused']) != {'science', 'host', 'apps'}:
                raise RuntimeError('Warm launch did not reuse all verified components')
        if 'host' not in cross['rows'][0]['persistentCache']['reused']:
            raise RuntimeError('Cross-build launch did not reuse unchanged host bytes')
        medians = {v: statistics.median(r['result']['readyMs'] for r in rows if r['variant'] == v) for v in ('before', 'after')}
        threshold = max(1000, medians['before'] * .05)
        passed = medians['after'] - medians['before'] <= threshold
        report = dict(success=passed, recipe=recipe, cold=rows, mediansMs=medians,
                      allowedIncreaseMs=threshold, warm=warm, crossBuild=cross)
        write_json(out / 'report.json', report)
        print(json.dumps({'success': passed, 'mediansMs': medians, 'evidence': str(out)}), flush=True)
        if not passed:
            raise RuntimeError('Startup median regression exceeds the project threshold')


if __name__ == '__main__':
    main()
