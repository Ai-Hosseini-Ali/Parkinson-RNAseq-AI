; Inno Setup script — builds a real Windows installer (setup.exe) for
; Parkinson AI Predictor, with a Start Menu entry, optional Desktop icon,
; and a proper uninstaller listed in "Add or Remove Programs".
;
; Requirements:
;   1. Download & install Inno Setup (free): https://jrsoftware.org/isdl.php
;   2. Build the .exe FIRST with PyInstaller (see build_exe.spec) —
;      this script packages the *already built* dist\ParkinsonAIPredictor folder.
;   3. Open this file in Inno Setup (or right-click > Compile), or run:
;        iscc installer.iss
;   4. The final installer appears in the "installer_output" folder.

#define MyAppName "Parkinson AI Predictor"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Ali Hoseini"
#define MyAppExeName "ParkinsonAIPredictor.exe"

[Setup]
AppId={{B2E1F1F0-9C3A-4F2E-8B1A-PARKINSONAI01}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=ParkinsonAIPredictor-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
SetupIconFile=gui\assets\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; PyInstaller's onedir build output — everything inside dist\ParkinsonAIPredictor\
Source: "dist\ParkinsonAIPredictor\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
