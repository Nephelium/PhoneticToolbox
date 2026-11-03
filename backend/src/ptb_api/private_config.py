"""Read-only verification of private launcher configuration permissions."""
import json
import os
import stat
import subprocess


def validate_private_config(path):
    from .storage import no_links
    no_links(path)
    info=path.stat()
    if not stat.S_ISREG(info.st_mode):raise ValueError('Private configuration must be a regular file')
    if os.name=='posix':
        if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:
            raise ValueError('Private configuration must be owned and mode 0600')
        return
    if os.name!='nt':raise ValueError('Private configuration permission checks unavailable')
    # Inspect the ACL only. Do not rewrite permissions or print configuration.
    script=r'''
$ErrorActionPreference='Stop'
$a=Get-Acl -LiteralPath $env:PTB_PRIVATE_CONFIG_PATH
$u=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$o=$a.GetOwner([System.Security.Principal.SecurityIdentifier]).Value
$r=@($a.GetAccessRules($true,$true,[System.Security.Principal.SecurityIdentifier]) | ForEach-Object {
  @{sid=$_.IdentityReference.Value; allow=($_.AccessControlType -eq 'Allow'); rights=[int]$_.FileSystemRights; inheritOnly=(($_.PropagationFlags -band 2) -ne 0)}
})
@{user=$u;owner=$o;rules=$r} | ConvertTo-Json -Depth 4 -Compress
'''
    env=dict(os.environ,PTB_PRIVATE_CONFIG_PATH=str(path))
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],
        env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=15,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise ValueError('Private configuration ACL inspection failed')
    _validate_windows_acl(json.loads(result.stdout))


def _validate_windows_acl(value):
    trusted={value['user'],'S-1-5-18','S-1-5-32-544'}
    if value['owner']!=value['user'] or not value['rules']:
        raise ValueError('Private configuration must have an owner-scoped ACL')
    for rule in value['rules']:
        if rule['allow'] and not rule['inheritOnly'] and rule['sid'] not in trusted and rule['rights'] & 0xD0117:
            raise ValueError('Private configuration grants another principal read/write access')
