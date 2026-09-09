"""Read-only comparison of the user's executable payload and reference source."""
import hashlib
import json
import marshal
import types
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader

from baseline_support import sha

ROOT = Path(__file__).resolve().parents[1]


def normalized(code):
    def constant(value):
        if isinstance(value, types.CodeType):
            return normalized(value)
        if isinstance(value, (tuple, frozenset)):
            values = [constant(v) for v in value]
            return sorted(values, key=repr) if isinstance(value, frozenset) else values
        return (type(value).__name__, repr(value))
    return {'bytecode': code.co_code.hex(), 'constants': [constant(v) for v in code.co_consts],
            'names': code.co_names, 'varnames': code.co_varnames, 'freevars': code.co_freevars,
            'cellvars': code.co_cellvars, 'argcount': code.co_argcount, 'posonly': code.co_posonlyargcount,
            'kwonly': code.co_kwonlyargcount, 'flags': code.co_flags, 'exceptiontable': code.co_exceptiontable.hex()}


def code_hash(code):
    return hashlib.sha256(json.dumps(normalized(code), sort_keys=True).encode()).hexdigest()


def main():
    evidence = json.loads((ROOT / 'docs/baseline/local-evidence.json').read_text('utf-8'))
    source = Path(evidence['baseline']['root'])
    executable = source / 'dist/PhoneticToolbox_v2.2.0.exe'
    output = ROOT / 'output/validation/p03/executable'
    output.mkdir(parents=True, exist_ok=True)
    archive = CArchiveReader(str(executable))
    pyz = archive.open_embedded_archive('PYZ.pyz')
    names = [name for name in pyz.toc if name.startswith((
        'phonetic_toolbox.core.acoustic', 'phonetic_toolbox.core.egg', 'phonetic_toolbox.services.io'))
        or name in {'phonetic_toolbox.models.config', 'phonetic_toolbox.models.egg_models',
                    'phonetic_toolbox.services.acoustic_service', 'phonetic_toolbox.services.egg_service'}]
    comparisons = []
    for name in sorted(names):
        path = source / (name.replace('.', '/') + '.py')
        if not path.exists():
            path = source / name.replace('.', '/') / '__init__.py'
        embedded = pyz.extract(name)
        original = compile(path.read_bytes(), str(path), 'exec', optimize=0)
        left, right = code_hash(embedded), code_hash(original)
        comparisons.append({'module': name, 'source_sha256': sha(path), 'embedded_code_sha256': left,
                            'source_code_sha256': right, 'same_normalized_code': left == right})
        # Local evidence only, never imported by application packages or shipped.
        (output / (name + '.marshal')).write_bytes(marshal.dumps(embedded))
    native_key = 'phonetic_toolbox\\core\\acoustic\\reaper.exe'
    native = archive.extract(native_key)
    native_hash = hashlib.sha256(native).hexdigest()
    report = {'executable_name': executable.name, 'executable_sha256': sha(executable),
        'baseline_declared_sha256': '40c807cd9a58da11d1e87e805f9ea84cdb515b8998f99588e9ae788dde218cdd',
        'comparison': comparisons, 'reaper_sha256': native_hash,
        'reaper_matches_reference': native_hash == sha(source / native_key.replace('\\', '/')),
        'scope': 'Embedded normalized code comparison; excludes filename/line tables, retains constants and exception behavior',
        'full_executable_gui_execution': 'not_performed'}
    assert report['executable_sha256'] == report['baseline_declared_sha256'], 'Executable changed since D0.3'
    (output / 'module-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    assert comparisons and all(c['same_normalized_code'] for c in comparisons), 'Embedded/reference code differs; inspect report'
    assert report['reaper_matches_reference'], 'Embedded/reference REAPER differs'
    print(json.dumps({'modules': len(comparisons), 'different': [c['module'] for c in comparisons if not c['same_normalized_code']],
                      'reaper_matches': report['reaper_matches_reference'], 'exe_hash': report['executable_sha256']}))


if __name__ == '__main__':
    main()
