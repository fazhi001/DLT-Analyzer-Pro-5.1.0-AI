#define MyAppName "DLT Analyzer Pro 5.2 三种彩票可信分析版"
#define MyAppVersion "5.2.0"
#define MyAppPublisher "DLT Analyzer Pro"
#define MyAppExeName "DLTAnalyzerPro.exe"

[Setup]
AppId={{0C726E3B-11B5-4D92-98DB-B47E7F8D8B42}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\DLT Analyzer Pro AI
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
OutputDir=..\release
OutputBaseFilename=DLT_Analyzer_Pro_5.2.0_3Games_Setup_x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\resources\app.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes

[Files]
Source: "..\dist\DLTAnalyzerPro\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加选项："; Flags: checkedonce

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent
