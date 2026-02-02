@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
echo ========================================
echo 自动签到助手 v2.0.6 构建脚本
echo ========================================
echo.

echo [1/4] 备份模板文件...
if exist dist\JT (
    REM 创建备份目录
    if not exist template_backups mkdir template_backups
    
    REM 固定备份文件名（只保留一个最新的）
    set backup_file=template_backups\templates_latest.zip
    
    REM 压缩模板文件
    powershell -Command "Compress-Archive -Path 'dist\JT\*' -DestinationPath '!backup_file!' -Force" >nul 2>&1
    
    if exist "!backup_file!" (
        REM 加密备份文件
        python -c "import sys; sys.path.insert(0, 'src'); from crypto_utils import crypto; data = open('!backup_file!', 'rb').read(); encrypted = crypto.encrypt_file_content(data); open('!backup_file!.encrypted', 'wb').write(encrypted)" >nul 2>&1
        
        if exist "!backup_file!.encrypted" (
            del /q "!backup_file!" >nul 2>&1
            echo ✅ 模板文件已加密备份
        ) else (
            echo ✅ 模板文件已备份
        )
    ) else (
        echo ⚠️  模板文件备份失败
    )
) else (
    echo ⚠️  警告: dist\JT 模板文件夹不存在
)
echo.

echo [2/4] 清理旧的构建文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist "自动签到助手_v2.0.6.spec" del /q "自动签到助手_v2.0.6.spec"
echo 清理完成
echo.

echo [3/4] 恢复模板文件...
REM 从加密备份恢复
if exist "template_backups\templates_latest.zip.encrypted" (
    echo 正在从加密备份恢复...
    
    REM 解密备份文件
    python -c "import sys; sys.path.insert(0, 'src'); from crypto_utils import crypto; encrypted = open('template_backups\templates_latest.zip.encrypted', 'rb').read(); decrypted = crypto.decrypt_file_content(encrypted); open('template_backups\templates_latest.zip', 'wb').write(decrypted)" >nul 2>&1
    
    if exist "template_backups\templates_latest.zip" (
        REM 解压到 dist\JT
        if not exist dist mkdir dist
        powershell -Command "Expand-Archive -Path 'template_backups\templates_latest.zip' -DestinationPath 'dist\JT' -Force" >nul 2>&1
        
        if exist "dist\JT" (
            echo ✅ 模板文件已从加密备份恢复
            del /q "template_backups\templates_latest.zip" >nul 2>&1
            
            REM 加密模板文件（在打包前）
            echo 正在加密模板文件...
            python encrypt_templates.py <<EOF >nul 2>&1
1
EOF
            if %errorlevel% equ 0 (
                echo ✅ 模板文件已加密
            ) else (
                echo ⚠️  模板加密失败
            )
        ) else (
            echo ❌ 解压失败
        )
    ) else (
        echo ❌ 解密失败
    )
) else (
    echo ⚠️  警告: 没有找到备份文件
)
echo.

echo [4/9] 开始构建 EXE...
echo 注意: 模型文件不打包到EXE中，将在构建后单独复制
echo.

pyinstaller --onefile --windowed ^
  --name="自动签到助手_v2.0.6" ^
  --add-data "C:\Program Files\Python311\Lib\site-packages\rapidocr;rapidocr" ^
  --add-data "dist/JT;dist/JT" ^
  run.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ 构建失败！
    pause
    exit /b 1
)

echo.
echo [5/7] 复制配置文件...
if exist ".env" (
    copy /Y ".env" "dist\.env" >nul
    echo ✅ 配置文件已复制到 dist 目录
) else (
    echo ⚠️  警告: .env 文件不存在，请手动复制
)
echo.

echo [6/7] 创建 runtime_data 目录...
if not exist "dist\runtime_data" mkdir "dist\runtime_data"
echo ✅ runtime_data 目录已创建
echo.

echo [6.5/7] 处理数据文件...
echo.
echo 选择数据处理方式:
echo   1 - 复制现有数据 (用于自己测试)
echo   2 - 创建空数据库 (用于发布给新用户)
echo.
set /p data_choice="请选择 (1 或 2): "

if "%data_choice%"=="1" (
    echo.
    echo 正在复制现有数据...
    if exist "runtime_data\license.db" (
        copy /Y "runtime_data\license.db" "dist\runtime_data\license.db" >nul
        echo ✅ 已复制数据库文件
    )
    if exist ".account_cache.json" (
        copy /Y ".account_cache.json" "dist\.account_cache.json" >nul
        echo ✅ 已复制账号缓存
    )
    if exist "login_cache" (
        xcopy /E /I /Y /Q "login_cache" "dist\login_cache" >nul
        echo ✅ 已复制登录缓存
    )
) else (
    echo.
    echo 正在创建新用户环境...
    
    REM 创建空的数据库
    python -c "import sys; sys.path.insert(0, 'src'); import os; os.chdir('dist'); from local_db import LocalDatabase; db = LocalDatabase(); print('✅ 已创建空数据库')"
    
    REM 创建示例账号文件
    echo 13800138000----password123 > "dist\账号.txt"
    echo 13900139000----password456 >> "dist\账号.txt"
    echo ✅ 已创建示例账号文件 (dist\账号.txt)
    
    REM 创建登录缓存目录
    if not exist "dist\login_cache" mkdir "dist\login_cache"
    echo ✅ 已创建登录缓存目录
    
    echo ⚠️  注意: 新用户需要自己激活许可证
)
echo.

echo [7/9] 复制models目录...
if exist "models" (
    xcopy /E /I /Y /Q "models" "dist\models" >nul
    echo ✅ models 目录已复制到 dist
    echo    模型文件不打包到EXE中，作为外部文件夹分发
) else (
    echo ❌ 警告: models 目录不存在
)
echo.

echo [8/9] 创建发布包...
if not exist "release" mkdir "release"

REM 创建发布包目录
set release_dir=release\自动签到助手_v2.0.6
if exist "%release_dir%" rmdir /s /q "%release_dir%"
mkdir "%release_dir%"

REM 复制文件
copy /Y "dist\自动签到助手_v2.0.6.exe" "%release_dir%\" >nul
xcopy /E /I /Y /Q "dist\models" "%release_dir%\models" >nul
if exist "dist\.env" copy /Y "dist\.env" "%release_dir%\.env.example" >nul
if exist "dist\账号.txt" copy /Y "dist\账号.txt" "%release_dir%\账号.txt.example" >nul

REM 创建README
echo 自动签到助手 v2.0.6 > "%release_dir%\README.txt"
echo. >> "%release_dir%\README.txt"
echo 使用说明: >> "%release_dir%\README.txt"
echo 1. 将 .env.example 重命名为 .env >> "%release_dir%\README.txt"
echo 2. 将 账号.txt.example 重命名为 账号.txt 并填写账号信息 >> "%release_dir%\README.txt"
echo 3. 运行 自动签到助手_v2.0.6.exe >> "%release_dir%\README.txt"
echo. >> "%release_dir%\README.txt"
echo 注意: >> "%release_dir%\README.txt"
echo - models 文件夹包含所有模型文件，请勿删除 >> "%release_dir%\README.txt"
echo - 更新模型时，只需替换 models 文件夹即可 >> "%release_dir%\README.txt"
echo - 模型版本信息见 models\model_version.json >> "%release_dir%\README.txt"

echo ✅ 发布包已创建: %release_dir%
echo.

echo [9/9] 构建完成！
echo.
echo ✅ EXE 文件位置: dist\自动签到助手_v2.0.6.exe
echo ✅ 配置文件位置: dist\.env
echo ✅ 模板文件位置: dist\JT (共 30 个模板文件)
echo ✅ 模型文件位置: dist\models (外置，可单独更新)
echo ✅ 运行数据目录: dist\runtime_data
echo ✅ 发布包位置: %release_dir%
echo.
if "%data_choice%"=="1" (
    echo ⚠️  当前版本包含你的数据 (仅供自己使用)
) else (
    echo ✅ 当前版本为干净版本 (可发布给新用户)
    echo.
    echo 📦 新用户版本包含:
    echo   ✅ 空的数据库 (需要激活许可证)
    echo   ✅ 示例账号文件 (账号.txt.example)
    echo   ✅ 空的登录缓存目录
    echo   ✅ 完整的models文件夹 (可单独更新)
    echo   ✅ 完整的目录结构
)
echo.
echo ========================================
echo 重要更新:
echo - 模型文件已外置到 models 文件夹
echo - 更新模型时无需重新打包EXE
echo - 只需替换 models 文件夹即可
echo - 模型版本: 见 models\model_version.json
echo ========================================
echo.

echo 构建完成！模型文件已外置，支持独立更新。
echo.
pause
