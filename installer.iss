; Inno Setup 配置文件 - 自动助手安装程序
; 使用 Inno Setup 6.0 或更高版本编译

#define MyAppName "自动助手"
#define MyAppVersion "2.0.6"
#define MyAppPublisher "自动助手开发团队"
#define MyAppExeName "XiMengHelper.exe"

[Setup]
; 应用程序基本信息
AppId={{A5B6C7D8-E9F0-1234-5678-9ABCDEF01234}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; 输出配置
OutputDir=installer_output
OutputBaseFilename=自动助手_v{#MyAppVersion}_安装包
SetupIconFile=
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; 权限和兼容性
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; 界面配置
DisableWelcomePage=no
LicenseFile=
InfoBeforeFile=
InfoAfterFile=
; 卸载配置
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加图标:"; Flags: unchecked
Name: "quicklaunchicon"; Description: "创建快速启动栏快捷方式"; GroupDescription: "附加图标:"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; 主程序文件
Source: "D:\zdqd\XiMengHelper.exe"; DestDir: "{app}"; Flags: ignoreversion
; _internal 目录（所有依赖库）
Source: "D:\zdqd\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
; 配置文件
Source: "D:\zdqd\config\*"; DestDir: "{app}\config"; Flags: ignoreversion recursesubdirs createallsubdirs
; 数据文件（只包含模板文件）
Source: "D:\zdqd\data\账号详情.xlsx"; DestDir: "{app}\data"; Flags: ignoreversion
; 根目录配置文件
Source: "D:\zdqd\config.yaml"; DestDir: "{app}"; Flags: ignoreversion
Source: "D:\zdqd\.env"; DestDir: "{app}"; Flags: ignoreversion
; 模型文件（可选，如果目录为空则跳过）
Source: "D:\zdqd\models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

[Icons]
; 开始菜单快捷方式
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
; 桌面快捷方式
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
; 快速启动栏快捷方式
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
; 安装完成后运行程序（可选）
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
// [2026-02-24] 修复原因：检测 VC++ 运行库，解决 torch DLL 依赖问题
function IsVCRedistInstalled(): Boolean;
var
  Major: Cardinal;
  Minor: Cardinal;
  Bld: Cardinal;
  Rbld: Cardinal;
begin
  // 检查 VC++ 2015-2022 Redistributable (x64) 是否已安装
  // 注册表路径：HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64
  Result := RegQueryDWordValue(HKEY_LOCAL_MACHINE, 
    'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 
    'Installed', Major) and (Major = 1);
    
  // 如果上面的路径不存在，尝试检查另一个路径
  if not Result then
  begin
    Result := RegQueryDWordValue(HKEY_LOCAL_MACHINE,
      'SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
      'Installed', Major) and (Major = 1);
  end;
end;

function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
  ErrorCode: Integer;
begin
  Result := True;
  
  // [2026-02-24] 检查 VC++ 运行库
  if not IsVCRedistInstalled() then
  begin
    if MsgBox('检测到系统缺少必需的运行库：' + #13#10 + #13#10 +
              'Microsoft Visual C++ 2015-2022 Redistributable (x64)' + #13#10 + #13#10 +
              '程序运行需要此运行库支持。' + #13#10 + #13#10 +
              '是否立即下载并安装？' + #13#10 +
              '（选择"否"将继续安装，但程序可能无法正常运行）', 
              mbConfirmation, MB_YESNO) = IDYES then
    begin
      // 打开微软官方下载页面
      ShellExec('open', 
        'https://aka.ms/vs/17/release/vc_redist.x64.exe',
        '', '', SW_SHOW, ewNoWait, ErrorCode);
      
      MsgBox('请在浏览器中下载并安装 VC++ 运行库。' + #13#10 + #13#10 +
             '安装完成后，请重新运行本安装程序。', 
             mbInformation, MB_OK);
      
      Result := False;
      Exit;
    end;
  end;
  
  // 检查是否已安装，如果已安装则提示卸载
  if RegKeyExists(HKEY_LOCAL_MACHINE, 'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{A5B6C7D8-E9F0-1234-5678-9ABCDEF01234}_is1') or
     RegKeyExists(HKEY_CURRENT_USER, 'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{A5B6C7D8-E9F0-1234-5678-9ABCDEF01234}_is1') then
  begin
    if MsgBox('检测到已安装旧版本的自动助手。' + #13#10 + #13#10 +
              '建议先卸载旧版本再继续安装。' + #13#10 + #13#10 +
              '是否继续安装？', mbConfirmation, MB_YESNO) = IDNO then
    begin
      Result := False;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // 安装完成后的操作
    if IsVCRedistInstalled() then
    begin
      MsgBox('安装完成！' + #13#10 + #13#10 +
             '提示：' + #13#10 +
             '1. 首次运行需要配置模拟器路径' + #13#10 +
             '2. 模型文件已自动安装' + #13#10 +
             '3. 如有问题请查看程序日志', mbInformation, MB_OK);
    end
    else
    begin
      MsgBox('安装完成！' + #13#10 + #13#10 +
             '⚠️ 警告：检测到系统缺少 VC++ 运行库' + #13#10 + #13#10 +
             '程序可能无法正常运行。建议安装：' + #13#10 +
             'Microsoft Visual C++ 2015-2022 Redistributable (x64)' + #13#10 + #13#10 +
             '下载地址：https://aka.ms/vs/17/release/vc_redist.x64.exe' + #13#10 + #13#10 +
             '其他提示：' + #13#10 +
             '1. 首次运行需要配置模拟器路径' + #13#10 +
             '2. 模型文件已自动安装' + #13#10 +
             '3. 如有问题请查看程序日志', mbInformation, MB_OK);
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ResultCode: Integer;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    // 询问是否删除用户数据
    if MsgBox('是否删除所有用户数据和配置文件？' + #13#10 + #13#10 +
              '选择"是"将删除所有数据（包括账号信息、历史记录等）' + #13#10 +
              '选择"否"将保留数据以便重新安装后继续使用', mbConfirmation, MB_YESNO) = IDYES then
    begin
      // 删除用户数据
      DelTree(ExpandConstant('{app}\data'), True, True, True);
      DelTree(ExpandConstant('{app}\config'), True, True, True);
      DelTree(ExpandConstant('{app}\models'), True, True, True);
      DeleteFile(ExpandConstant('{app}\config.yaml'));
      DeleteFile(ExpandConstant('{app}\.env'));
    end;
  end;
end;
