param([ValidateSet('Web','Manager','Tunnel')][string]$View='Web')
$ErrorActionPreference='Stop'
$testRoot=Split-Path -Parent $PSScriptRoot
if($View -eq 'Tunnel') {
    # Interactive password prompt; no password is stored by this launcher.
    & ssh.exe -C -N -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18765:127.0.0.1:18765 -L 127.0.0.1:18768:127.0.0.1:18768 admin@8.134.196.183
    exit $LASTEXITCODE
}
try {
    $testHealth=Invoke-RestMethod -Uri 'http://127.0.0.1:18765/api/v1/health' -TimeoutSec 5
} catch {
    throw '隔离测试隧道尚未连接。先运行本脚本 -View Tunnel，并在SSH窗口输入服务器密码。'
}
if($View -eq 'Web') {
    Start-Process 'http://127.0.0.1:18765/app/'
} else {
    $testPython=Join-Path $testRoot '.venv/m14/Scripts/pythonw.exe'
    $testEntry=Join-Path $testRoot 'tools/feedback-manager/launch_isolated.py'
    Start-Process -FilePath $testPython -ArgumentList @('-B',('"'+$testEntry+'"')) -WorkingDirectory $testRoot
}
