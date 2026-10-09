"""Only explicitly registered, completed, local MFA diagnostic attempts expire."""
import json
import os
from pathlib import Path
import re
import stat
from uuid import UUID
from .io.scratch import no_links
from .store import canonical, JobError

ATTEMPT = re.compile(r'^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})-(0|[1-9][0-9]{0,9})-([0-9a-f]{32})$')
MARKER = '.ptb-mfa-attempt.json'
DAYS = 7
TERMINAL = {'succeeded','failed','cancelled','interrupted'}
ACTIVE = {'queued','running','cancel_requested'}


def directory_id(path):
    no_links(path)
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_ino <= 0:
        raise JobError('mfa_diagnostic_identity_rejected',409)
    return [value.st_dev,value.st_ino]


def tree(path,*,include_items=False):
    """Validate before recursing, so a junction's target is never enumerated."""
    pending=[Path(path)];items=[];size=0
    while pending:
        item=pending.pop();no_links(item)
        value=item.lstat()
        if stat.S_ISDIR(value.st_mode):
            pending.extend(item.iterdir())
        elif stat.S_ISREG(value.st_mode) and value.st_nlink==1:
            size+=value.st_size
        else:
            raise JobError('mfa_diagnostic_tree_rejected',409)
        items.append(item)
        if len(items)>200000:
            raise JobError('mfa_diagnostic_tree_rejected',409)
    return (size,items) if include_items else size


def remove_owned_tree(path):
    """Keep the ownership receipt until all other children have been removed."""
    _,items=tree(path,include_items=True)
    marker=path/MARKER;receipt=marker.read_bytes();identity=directory_id(path)
    for item in sorted(items,key=lambda item:len(item.parts),reverse=True):
        if item in (path,marker):continue
        no_links(item);value=item.lstat()
        if stat.S_ISDIR(value.st_mode):item.rmdir()
        elif stat.S_ISREG(value.st_mode) and value.st_nlink==1:item.unlink()
        else:raise JobError('mfa_diagnostic_tree_rejected',409)
    no_links(marker)
    if marker.lstat().st_nlink!=1:
        raise JobError('mfa_diagnostic_tree_rejected',409)
    marker.unlink()
    try:path.rmdir()
    except OSError:
        # A new child or denied rmdir leaves a partial attempt. Restore only the
        # exact receipt of the same directory, so a later sweep can retry safely.
        if path.exists() and directory_id(path)==identity and not marker.exists():
            with marker.open('xb') as output:output.write(receipt)
        raise


def current_path(name, record):
    from .mfa.runtime import registry_root
    registry=registry_root().absolute();no_links(registry)
    if record.get('registry')!=str(registry):
        raise JobError('mfa_diagnostic_registry_changed',409)
    match=ATTEMPT.fullmatch(name)
    if not match or match[1]!=record['job_id'] or int(match[2])!=record['generation']:
        raise JobError('mfa_diagnostic_identity_rejected',409)
    path=registry/'attempts'/name;no_links(path)
    if path.parent!=registry/'attempts' or not path.resolve().is_relative_to(registry.resolve()):
        raise JobError('mfa_diagnostic_path_rejected',409)
    return path


def validate_owned(name,record,instance_id):
    path=current_path(name,record)
    if directory_id(path)!=record['directory_id']:
        raise JobError('mfa_diagnostic_identity_rejected',409)
    marker=path/MARKER;no_links(marker)
    info=marker.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size>4096:
        raise JobError('mfa_diagnostic_identity_rejected',409)
    value=json.loads(marker.read_text('utf-8'))
    expected=dict(schema='ptb-mfa-attempt/1',instance_id=instance_id,job_id=record['job_id'],
                  generation=record['generation'],attempt=name,directory_id=record['directory_id'])
    if value!=expected:
        raise JobError('mfa_diagnostic_identity_rejected',409)
    return path


def register(path,identity,state,policy,now):
    from .mfa.runtime import registry_root
    registry=registry_root().absolute();path=Path(path).absolute()
    match=ATTEMPT.fullmatch(path.name)
    if (not match or path.parent!=registry/'attempts' or match[1]!=str(UUID(identity[0])) or
            int(match[2])!=identity[2] or type(identity[2]) is not int):
        raise JobError('mfa_diagnostic_path_rejected',409)
    record=dict(registry=str(registry),job_id=identity[0],generation=identity[2],
                directory_id=directory_id(path),first_seen=now,created_at=now,
                completed_at=None,size_bytes=0,removed=False)
    entries=policy.setdefault('mfa_attempts',{})
    if path.name in entries:
        raise JobError('mfa_diagnostic_duplicate',409)
    marker=dict(schema='ptb-mfa-attempt/1',instance_id=state['instance_id'],job_id=record['job_id'],
                generation=record['generation'],attempt=path.name,directory_id=record['directory_id'])
    no_links(path/MARKER)
    with (path/MARKER).open('x',encoding='utf-8') as stream:
        stream.write(canonical(marker));stream.flush();os.fsync(stream.fileno())
    entries[path.name]=record


def completed(path,identity,state,policy,now):
    record=policy.get('mfa_attempts',{}).get(Path(path).name)
    if not record or record['job_id']!=identity[0] or record['generation']!=identity[2]:
        raise JobError('mfa_diagnostic_unregistered',409)
    root=validate_owned(Path(path).name,record,state['instance_id'])
    if root!=Path(path).absolute():
        raise JobError('mfa_diagnostic_path_rejected',409)
    record['size_bytes']=tree(root)
    # The first complete diagnostic receipt starts a full seven-day grace.
    # No mtime, old task timestamps or a later status read can shorten it.
    if record['completed_at'] is None:
        record['completed_at']=now


def sweep(state,policy,jobs,now,*,remove=remove_owned_tree):
    removed=freed=failed=protected=0
    for name,record in policy.get('mfa_attempts',{}).items():
        if record.get('removed'):continue
        job=jobs.get(record['job_id'])
        completed_at=record.get('completed_at')
        if (completed_at is None or not job or job['state'] in ACTIVE or job['state'] not in TERMINAL or
                now-max(record['first_seen'],record['created_at'],completed_at)<DAYS*86400):
            protected+=1;continue
        try:
            before=None
            path=validate_owned(name,record,state['instance_id'])
            before=tree(path)
            # Recheck the root after complete-tree validation and immediately
            # before deleting. The local asset/SQLite lock remains held.
            if directory_id(path)!=record['directory_id']:
                raise JobError('mfa_diagnostic_identity_rejected',409)
            remove(path)
            if path.exists():raise OSError('diagnostic still present')
        except Exception as error:
            failed+=1
            record['last_failure']=getattr(error,'code','mfa_diagnostic_remove_failed')
            if before is not None:
                try:
                    remaining=tree(validate_owned(name,record,state['instance_id']))
                    record['size_bytes']=remaining;freed+=max(0,before-remaining)
                except Exception:pass
            # A partial deletion must remain registered and retryable. Its root
            # and ownership marker are retained by the deletion routine below.
            continue
        record.update(removed=True,removed_at=now,size_bytes=0)
        record.pop('last_failure',None)
        removed+=1;freed+=before
    return dict(diagnostic_count=removed,diagnostic_bytes=freed,
                diagnostic_failed_count=failed,diagnostic_protected_count=protected,
                diagnostic_days=DAYS)
