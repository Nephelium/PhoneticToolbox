"""Explicit uninstall of app-managed MFA copies; external environments survive."""
import json
import re
from pathlib import Path

from .platform_paths import user_data_root
from .startup_cache import FileLock, active_leases, atomic_json, cache_root, plain, remove_owned


def remove_managed_mfa():
    # Never use PTB_M11_COMPONENT_ROOT here: it can name a user's external tree.
    base = user_data_root()
    root = plain(base / 'components/mfa', base)
    report = dict(complete=False, removed=[], retained=[])
    try:
        with FileLock(base / '.startup-cache.lock'):
            cache = plain(cache_root(), base)
            if cache.exists() and active_leases(cache):
                raise ValueError('请先关闭全部 PhoneticToolbox 窗口。')
            if not root.exists():
                report['complete'] = True
                return report
            versions = plain(root / 'versions', root)
            targets = []
            if versions.is_dir():
                for folder in versions.iterdir():
                    plain(folder, root)
                    if not folder.is_dir() or not re.fullmatch(r'mfa338-[0-9a-f]{16}', folder.name):
                        report['retained'].append(str(folder))
                        continue
                    marker = plain(folder / 'ptb-component.json', root)
                    if not marker.is_file():
                        report['retained'].append(str(folder))
                        continue
                    value = json.loads(marker.read_text('utf8'))
                    manifest = value.get('manifest', {})
                    if manifest.get('schema') != 'm11-component/1' or manifest.get('mfa_version') != '3.3.8':
                        report['retained'].append(str(folder))
                        continue
                    targets.append(folder)
            registry_path = plain(root / 'registry.json', root)
            registry = json.loads(registry_path.read_text('utf8')) if registry_path.exists() else None
            removed_ids = set()
            remaining_models = []
            model_files = set()
            receipts = set()
            if registry is not None:
                if registry.get('schema') != 'm11-registry/1':
                    raise ValueError('MFA 登记格式无法确认，组件已保留。')
                kept = []
                for row in registry['runtimes']:
                    runtime = Path(row['path']).absolute()
                    owned = runtime.name == 'env' and runtime.parent.parent == versions and re.fullmatch(r'mfa338-[0-9a-f]{16}', runtime.parent.name)
                    if owned and (runtime.parent in targets or not runtime.parent.exists()):
                        removed_ids.add(row['id'])
                        receipt = Path(row.get('receipt', '')).absolute()
                        if receipt.name == 'receipt.json' and receipt.parent.parent == root / 'checks' and re.fullmatch(r'[0-9a-f]{32}', receipt.parent.name):
                            receipts.add(plain(receipt.parent, root))
                    else:
                        kept.append(row)
                for row in registry['models']:
                    if row.get('validated_runtime') not in removed_ids:
                        remaining_models.append(row)
                        continue
                    for role, suffix in (('model', '.zip'), ('dictionary', '.dict')):
                        sha = row.get(role + '_sha256', '')
                        candidate = Path(row[role]).absolute()
                        if re.fullmatch(r'[0-9a-f]{64}', sha) and candidate == root / 'resources' / role / (sha + suffix):
                            model_files.add(plain(candidate, root))
                for row in remaining_models:
                    for role in ('model', 'dictionary'):
                        model_files.discard(Path(row[role]).absolute())
                registry.update(runtimes=kept, models=remaining_models)
            current_path = plain(root / 'current.json', root)
            current = json.loads(current_path.read_text('utf8')) if current_path.exists() else None
            for folder in targets:
                remove_owned(folder, root)
                report['removed'].append(str(folder))
            if registry is not None:
                atomic_json(registry_path, registry)
            if current and Path(current.get('path', '')).absolute() in targets:
                current_path.unlink()
            for folder in receipts:
                if folder.is_dir():
                    remove_owned(folder, root)
                    report['removed'].append(str(folder))
            downloads = plain(root / 'downloads', root)
            if downloads.is_dir():
                model_files.update(plain(p, root) for p in downloads.iterdir()
                                   if re.fullmatch(r'[0-9a-f]{64}\.part(?:\.rejected-[0-9a-f]{8})?', p.name))
            for file in model_files:
                if file.is_file():
                    plain(file, root).unlink()
                    report['removed'].append(str(file))
            report['complete'] = not report['retained']
    except (OSError, ValueError, KeyError, TypeError) as error:
        report['error'] = str(error)
    finally:
        base.mkdir(parents=True, exist_ok=True)
        atomic_json(plain(base / 'mfa-uninstall-report.json', base), report)
    return report
