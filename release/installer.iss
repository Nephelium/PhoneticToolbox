; Current-user Preview installer. Product data remains outside {app}.
; Compile with /DPackageDir=... /DReleaseDir=... /DVersion=3.0.0-preview.1
#ifndef PackageDir
  #error PackageDir is required
#endif
#ifndef ReleaseDir
  #error ReleaseDir is required
#endif
#ifndef Version
  #define Version "3.0.0"
#endif
#ifndef AppIdentity
  #define AppIdentity "{{54E566B2-E6C5-46FA-B16C-3BEF91DD8769}"
#endif
#ifndef AppExecutable
  #define AppExecutable "PhoneticToolbox.exe"
#endif

[Setup]
AppId={#AppIdentity}
AppName=PhoneticToolbox
AppVerName=PhoneticToolbox {#Version} (Windows x64)
AppVersion={#Version}
AppPublisher=PhoneticToolbox
AppPublisherURL=https://www.phonetictoolbox.com/
AppSupportURL=https://www.phonetictoolbox.com/
AppUpdatesURL=https://www.phonetictoolbox.com/
DefaultDirName={localappdata}\Programs\PhoneticToolbox
DefaultGroupName=PhoneticToolbox
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0.17763
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
VersionInfoVersion=3.0.0.0
VersionInfoProductName=PhoneticToolbox

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Messages]
english.FinishedLabel=PhoneticToolbox has been installed. Open it from the Start menu or, if selected, the desktop shortcut.
english.FinishedLabelNoIcons=PhoneticToolbox has been installed. Open it from the Start menu or, if selected, the desktop shortcut.
chinesesimplified.FinishedLabel=PhoneticToolbox 安装完成。请从开始菜单打开，若选择了桌面快捷方式，也可以从桌面打开。
chinesesimplified.FinishedLabelNoIcons=PhoneticToolbox 安装完成。请从开始菜单打开，若选择了桌面快捷方式，也可以从桌面打开。

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："; Flags: unchecked

[Files]
#ifdef CompactOnefile
Source: "{#PackageDir}\{#AppExecutable}"; DestDir: "{app}"; DestName: "PhoneticToolbox.exe"; Flags: ignoreversion
Source: "{#PackageDir}\application.json"; DestDir: "{app}"; Flags: ignoreversion
#else
Source: "{#PackageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif

[Icons]
#ifndef OwnedQA
Name: "{userprograms}\PhoneticToolbox\PhoneticToolbox"; Filename: "{app}\PhoneticToolbox.exe"; WorkingDir: "{app}"
Name: "{userdesktop}\PhoneticToolbox"; Filename: "{app}\PhoneticToolbox.exe"; WorkingDir: "{app}"; Tasks: desktopicon
#endif

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

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var ResultCode: Integer;
begin
  { Ask only after the ordinary uninstall confirmation. Silent uninstall keeps MFA. }
  if (CurUninstallStep = usUninstall) and not UninstallSilent and
     DirExists(ExpandConstant('{localappdata}\PhoneticToolbox\v3\components\mfa')) then
  begin
    if MsgBox('是否同时删除本应用下载或从 ZIP 安装的 MFA 组件？' #13#10#13#10 +
      '包括组件内的模型、词典及下载缓存。删除后再次使用需要重新安装。' #13#10 +
      '自行选择的外部环境、录音、工程和导出文件会保留。' #13#10#13#10 +
      '选择“是”删除组件，选择“否”保留组件（默认）。',
      mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
    begin
      if not Exec(ExpandConstant('{app}\PhoneticToolbox.exe'), '--ptb-remove-managed-mfa',
        ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
        MsgBox('部分 MFA 组件因占用、权限或归属无法确认而保留。主程序将继续卸载。' #13#10 +
          '清理记录：%LOCALAPPDATA%\PhoneticToolbox\v3\mfa-uninstall-report.json', mbInformation, MB_OK);
    end;
  end;
end;
#endif

[UninstallDelete]
Type: files; Name: "{app}\.ptb-installed.json"

// No uninstall-delete entry is permitted for settings, recordings or projects.
