; Current-user Preview installer. Product data remains outside {app}.
; Compile with /DPackageDir=... /DReleaseDir=... /DVersion=3.0.0-preview.1
#ifndef PackageDir
  #error PackageDir is required
#endif
#ifndef ReleaseDir
  #error ReleaseDir is required
#endif
#ifndef Version
  #define Version "3.0.0-preview.1"
#endif
#ifndef AppIdentity
  #define AppIdentity "{{54E566B2-E6C5-46FA-B16C-3BEF91DD8769}"
#endif

[Setup]
AppId={#AppIdentity}
AppName=PhoneticToolbox
AppVersion={#Version}
AppPublisher=PhoneticToolbox
AppPublisherURL=https://www.phonetictoolbox.com/
AppSupportURL=https://www.phonetictoolbox.com/
AppUpdatesURL=https://www.phonetictoolbox.com/
DefaultDirName={localappdata}\Programs\PhoneticToolbox
DefaultGroupName=PhoneticToolbox
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#ReleaseDir}
OutputBaseFilename=PhoneticToolbox-{#Version}-windows-x64-setup
Compression=lzma2/fast
SolidCompression=yes
DiskSpanning=no
SetupLogging=yes
UninstallDisplayIcon={app}\PhoneticToolbox.exe
#ifdef OwnedQA
; Isolated installation/update tests must not register a second user product.
CreateUninstallRegKey=no
#endif
CloseApplications=no
RestartApplications=no
WizardStyle=modern
UsePreviousAppDir=yes
DisableProgramGroupPage=yes
DisableWelcomePage=no
ChangesAssociations=no
ChangesEnvironment=no
VersionInfoVersion=3.0.0.1
VersionInfoProductName=PhoneticToolbox

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："; Flags: unchecked

[Files]
#ifdef CompactOnefile
Source: "{#PackageDir}\PhoneticToolbox.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#PackageDir}\application.json"; DestDir: "{app}"; Flags: ignoreversion
#else
Source: "{#PackageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif

[Icons]
#ifndef OwnedQA
Name: "{userprograms}\PhoneticToolbox\PhoneticToolbox"; Filename: "{app}\PhoneticToolbox.exe"; WorkingDir: "{app}"
Name: "{userdesktop}\PhoneticToolbox"; Filename: "{app}\PhoneticToolbox.exe"; WorkingDir: "{app}"; Tasks: desktopicon
#endif

[Run]
Filename: "{app}\PhoneticToolbox.exe"; Description: "启动 PhoneticToolbox"; Flags: nowait postinstall skipifsilent unchecked

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
#ifdef PersistentCache
    WizardForm.StatusLabel.Caption := '正在准备运行文件，首次安装需要一些时间，请耐心等待…';
    if not Exec(ExpandConstant('{app}\PhoneticToolbox.exe'), '--ptb-prepare-cache', ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
      RaiseException('运行文件准备失败，请检查磁盘空间或目录权限后重新安装。');
#endif
    if not SaveStringToFile(ExpandConstant('{app}\.ptb-installed.json'),
      '{"schema":"ptb-install/1","kind":"installer","version":"{#Version}"}', False) then
      RaiseException('无法保存本机安装信息，请检查安装目录权限。');
  end;
end;

#ifdef PersistentCache
function InitializeUninstall(): Boolean;
var ResultCode: Integer;
begin
  Result := Exec(ExpandConstant('{app}\PhoneticToolbox.exe'), '--ptb-clear-all-caches', ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
  if not Result and not UninstallSilent then
    MsgBox('缓存暂时无法清理。请先关闭全部 PhoneticToolbox 窗口，再重试卸载。应用和研究资料均已保留。', mbError, MB_OK);
end;
#endif

[UninstallDelete]
Type: files; Name: "{app}\.ptb-installed.json"

// No uninstall-delete entry is permitted for settings, recordings or projects.
