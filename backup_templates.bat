@echo off
chcp 65001 >nul
echo ========================================
echo 模板文件备份和加密工具
echo ========================================
echo.

REM 检查模板文件夹是否存在
if not exist "dist\JT" (
    echo ❌ 错误: dist\JT 模板文件夹不存在
    echo 请先运行构建脚本创建模板文件
    pause
    exit /b 1
)

REM 统计模板文件数量
set count=0
for %%f in (dist\JT\*) do set /a count+=1
echo 找到 %count% 个模板文件
echo.

REM 创建备份目录
set backup_dir=template_backups
if not exist "%backup_dir%" mkdir "%backup_dir%"

REM 生成备份文件名（带时间戳）
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set timestamp=%datetime:~0,8%_%datetime:~8,6%
set backup_file=%backup_dir%\templates_backup_%timestamp%.zip

echo 正在备份模板文件...
echo.

REM 使用 PowerShell 压缩文件（Windows 自带）
powershell -Command "Compress-Archive -Path 'dist\JT\*' -DestinationPath '%backup_file%' -Force"

if %errorlevel% equ 0 (
    echo ✅ 备份成功: %backup_file%
    
    REM 显示备份文件大小
    for %%A in ("%backup_file%") do (
        set size=%%~zA
        set /a size_kb=!size! / 1024
        echo 备份大小: !size_kb! KB
    )
) else (
    echo ❌ 备份失败
    pause
    exit /b 1
)

echo.
echo 正在加密备份文件...
echo.

REM 使用 Python 加密备份文件
python -c "import sys; sys.path.insert(0, 'src'); from crypto_utils import crypto; import shutil; data = open('%backup_file%', 'rb').read(); encrypted = crypto.encrypt_file_content(data); open('%backup_file%.encrypted', 'wb').write(encrypted); print('✅ 加密成功: %backup_file%.encrypted')"

if %errorlevel% equ 0 (
    echo.
    echo 是否删除未加密的备份文件? (Y/N)
    set /p delete_plain="请选择: "
    
    if /i "!delete_plain!"=="Y" (
        del /q "%backup_file%"
        echo ✅ 已删除未加密的备份文件
    ) else (
        echo ⚠️  保留了未加密的备份文件
    )
) else (
    echo ❌ 加密失败
)

echo.
echo ========================================
echo 备份完成！
echo ========================================
echo.
echo 备份位置: %backup_dir%
echo.

REM 列出所有备份文件
echo 现有备份文件:
dir /b "%backup_dir%"

echo.
echo 💡 提示:
echo   - 加密的备份文件 (.encrypted) 只能用解密脚本恢复
echo   - 建议定期备份到云盘或其他安全位置
echo   - 保留最近 5 个备份即可
echo.

pause
