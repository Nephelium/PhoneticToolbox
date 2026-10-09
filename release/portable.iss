; Single downloadable EXE: extract a complete portable folder, then run it.
; No installed-product registration, shortcuts, uninstaller or data deletion.
#ifndef PackageDir
  #error PackageDir is required
#endif
#ifndef ReleaseDir
  #error ReleaseDir is required
#endif
#ifndef Version
  #define Version "3.0.0-preview.1"
#endif

[Setup]
AppId=PhoneticToolbox-Portable-{#Version}
AppName=PhoneticToolbox 免安装版
AppVersion={#Version}
AppPublisher=PhoneticToolbox
DefaultDirName={src}\PhoneticToolbox-{#Version}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#ReleaseDir}
OutputBaseFilename=PhoneticToolbox-{#Version}-windows-x64-portable
Compression=lzma2/fast
SolidCompression=yes
DiskSpanning=no
Uninstallable=no
CreateUninstallRegKey=no
CloseApplications=no
RestartApplications=no
WizardStyle=modern
UsePreviousAppDir=no
DisableProgramGroupPage=yes
DisableDirPage=no
DisableWelcomePage=yes
ChangesAssociations=no
ChangesEnvironment=no
VersionInfoVersion=3.0.0.1
VersionInfoProductName=PhoneticToolbox Portable

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Files]
Source: "{#PackageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Run]
Filename: "{app}\PhoneticToolbox.exe"; Description: "启动 PhoneticToolbox"; Flags: nowait postinstall skipifsilent

[CustomMessages]
english.WelcomeLabel2=Extract the complete portable application to your selected folder. This does not register an installed product. Settings and research data use the existing user profile.
chinesesimplified.WelcomeLabel2=将完整免安装应用解压到所选目录。此操作不登记安装项。设置与研究数据继续使用原用户目录。

[Code]
function DirectoryContainsFiles(Directory: String): Boolean;
var
  Entry: TFindRec;
begin
  Result := False;
  if FindFirst(AddBackslash(Directory) + '*', Entry) then begin
    try
      repeat
        if (Entry.Name <> '.') and (Entry.Name <> '..') then begin
          Result := True;
          Break;
        end;
      until not FindNext(Entry);
    finally
      FindClose(Entry);
    end;
  end;
end;

function TargetContainsFiles(): Boolean;
begin
  Result := DirectoryContainsFiles(ExpandConstant('{app}'));
end;

function InitializeSetup(): Boolean;
var
  Directory: String;
begin
  Result := True;
  if WizardSilent then begin
    Directory := ExpandConstant('{param:DIR|}');
    if Directory = '' then
      Directory := ExpandConstant('{src}\PhoneticToolbox-{#Version}');
    if DirectoryContainsFiles(Directory) then begin
      Log('Portable target is not empty; preserving every existing file.');
      Result := False;
    end;
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = wpSelectDir then
    if TargetContainsFiles() then begin
      SuppressibleMsgBox('此目录已有文件。请选择新的空目录，或使用应用内检查更新。现有文件将完整保留。', mbError, MB_OK, IDOK);
      Result := False;
    end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
    if TargetContainsFiles() then
      RaiseException('目标目录已有文件，已阻止覆盖。请改用新的空目录。');
end;
