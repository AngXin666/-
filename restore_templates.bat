@echo off
chcp 65001 >nul
echo ========================================
echo 模板文件恢复工具
echo ========================================
echo.

REM 检查备份目录是否存在
if not exist "template_backups" (
    echo ❌ 错误: 备份目录不存在
    echo 请先运行 backup_templates.bat 创建备份
    pause
    exit /b 1
)

REM 列出所有备份文件
echo 可用的备份文件:
echo.
set index=0
for %%f in (template_backups\*.encrypted) do (
    set /a index+=1
    echo   !index!. %%~nxf
    set "backup_!index!=%%f"
)

if %index% equ 0 (
    echo ❌ 没有找到加密的备份文件
    echo.
    echo 查找未加密的备份文件:
    for %%f in (template_backups\*.zip) do (
        set /a index+=1
        echo   !index!. %%~nxf
        set "backup_!index!=%%f"
    )
    
    if !index! equ 0 (
        echo ❌ 没有找到任何备份文件
        pause
        exit /b 1
    )
)

echo.
set /p choice="请选择要恢复的备份 (输入编号): "

REM 验证输入
if not defined backup_%choice% (
    echo ❌ 无效的选择
    pause
    exit /b 1
)

call set selected_backup=%%backup_%choice%%%
echo.
echo 选择的备份: %selected_backup%
echo.

REM 确认恢复
echo ⚠️  警告: 恢复将覆盖当前的模板文件！
echo.
set /p confirm="确认恢复? (Y/N): "

if /i not "%confirm%"=="Y" (
    echo 已取消恢复
    pause
    exit /b 0
)

echo.
echo 正在恢复模板文件...
echo.

REM 检查是否是加密文件
echo %selected_backup% | findstr /i ".encrypted" >nul
if %errorlevel% equ 0 (
    echo 正在解密备份文件...
    
    REM 使用 Python 解密
    python -c "import sys; sys.path.insert(0, 'src'); from crypto_utils import crypto; encrypted = open('%selected_backup%', 'rb').read(); decrypted = crypto.decrypt_file_content(encrypted); open('%selected_backup:~0,-10%', 'wb').write(decrypted); print('✅ 解密成功')"
    
    if %errorlevel% neq 0 (
        echo ❌ 解密失败
        pause
        exit /b 1
    )
    
    set "zip_file=%selected_backup:~0,-10%"
) else (
    set "zip_file=%selected_backup%"
)

REM 备份当前模板文件
if exist "dist\JT" (
    echo 正在备份当前模板文件...
    if exist "dist\JT_old" rmdir /s /q "dist\JT_old"
    move "dist\JT" "dist\JT_old" >nul
    echo ✅ 当前模板已备份到 dist\JT_old
)

REM 创建目标目录
if not exist "dist" mkdir "dist"

REM 解压备份文件
echo 正在解压备份文件...
powershell -Command "Expand-Archive -Path '%zip_file%' -DestinationPath 'dist\JT' -Force"

if %errorlevel% equ 0 (
    echo ✅ 恢复成功！
    
    REM 统计恢复的文件数量
    set count=0
    for %%f in (dist\JT\*) do set /a count+=1
    echo 恢复了 %count% 个模板文件
    
    REM 删除临时解密文件
    if exist "%zip_file%" (
        if not "%zip_file%"=="%selected_backup%" (
            del /q "%zip_file%"
            echo ✅ 已删除临时解密文件
        )
    )
) else (
    echo ❌ 恢复失败
    
    REM 恢复旧的模板文件
    if exist "dist\JT_old" (
        if exist "dist\JT" rmdir /s /q "dist\JT"
        move "dist\JT_old" "dist\JT" >nul
        echo ✅ 已恢复原来的模板文件
    )
    
    pause
    exit /b 1
)

echo.
echo ========================================
echo 恢复完成！
echo ========================================
echo.
echo 模板文件位置: dist\JT
echo 旧模板备份: dist\JT_old (可手动删除)
echo.

pause
