@echo off
chcp 65001 >nul
echo ========================================
echo 自动备份模板文件（静默模式）
echo ========================================
echo.

REM 检查模板文件夹是否存在
if not exist "dist\JT" (
    echo ❌ 错误: dist\JT 模板文件夹不存在
    exit /b 1
)

REM 创建备份目录
set backup_dir=template_backups
if not exist "%backup_dir%" mkdir "%backup_dir%"

REM 生成备份文件名（带时间戳）
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set timestamp=%datetime:~0,8%_%datetime:~8,6%
set backup_file=%backup_dir%\templates_backup_%timestamp%.zip

echo [%date% %time%] 开始备份...

REM 使用 PowerShell 压缩文件
powershell -Command "Compress-Archive -Path 'dist\JT\*' -DestinationPath '%backup_file%' -Force" >nul 2>&1

if %errorlevel% equ 0 (
    echo [%date% %time%] ✅ 压缩成功
    
    REM 使用 Python 加密备份文件
    python -c "import sys; sys.path.insert(0, 'src'); from crypto_utils import crypto; data = open('%backup_file%', 'rb').read(); encrypted = crypto.encrypt_file_content(data); open('%backup_file%.encrypted', 'wb').write(encrypted)" >nul 2>&1
    
    if %errorlevel% equ 0 (
        echo [%date% %time%] ✅ 加密成功
        
        REM 删除未加密的备份文件
        del /q "%backup_file%" >nul 2>&1
        
        echo [%date% %time%] ✅ 备份完成: %backup_file%.encrypted
    ) else (
        echo [%date% %time%] ❌ 加密失败
        exit /b 1
    )
) else (
    echo [%date% %time%] ❌ 压缩失败
    exit /b 1
)

REM 清理旧备份（保留最近 5 个）
echo [%date% %time%] 清理旧备份...
set count=0
for /f "skip=5 delims=" %%f in ('dir /b /o-d "%backup_dir%\*.encrypted" 2^>nul') do (
    del /q "%backup_dir%\%%f" >nul 2>&1
    set /a count+=1
)

if %count% gtr 0 (
    echo [%date% %time%] ✅ 已清理 %count% 个旧备份
)

echo [%date% %time%] 备份任务完成
echo.

REM 如果是手动运行，暂停查看结果
if "%1"=="" pause
