@echo off
chcp 65001 >nul
echo ========================================
echo 溪盟商城自动化助手 - 构建脚本
echo 目标目录: D:\zdqd
echo ========================================
echo.

echo [提示] 此脚本将执行以下操作:
echo   1. 清理旧的构建文件
echo   2. 使用 PyInstaller 构建 EXE
echo   3. 自动整理并复制到 D:\zdqd
echo   4. 只复制必要文件，保持目录整洁
echo.
pause

echo.
echo [1/4] 清理旧的构建文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo [OK] 清理完成
echo.

echo [2/4] 构建 EXE（这可能需要几分钟）...
echo 正在打包代码到 EXE...
echo [提示] 构建过程中会显示详细进度，请耐心等待...
echo.

REM 执行构建并实时显示输出
pyinstaller --clean build_exe.spec 2>&1

if errorlevel 1 (
    echo.
    echo ========================================
    echo [ERROR] EXE 构建失败！
    echo ========================================
    echo.
    echo 可能的原因:
    echo   1. 缺少 pyinstaller: pip install pyinstaller
    echo   2. 缺少依赖库: pip install -r requirements.txt
    echo   3. 代码有语法错误
    echo   4. 内存不足
    echo.
    echo 请查看上方的错误信息，修复后重新运行
    pause
    exit /b 1
)

echo.
echo [OK] EXE 构建完成
echo.

echo [3/4] 验证构建结果...
if not exist "dist\XiMengHelper\XiMengHelper.exe" (
    echo [ERROR] EXE 文件不存在！
    pause
    exit /b 1
)
echo [OK] EXE 文件已生成
echo.

echo [4/4] 整理并复制到 D:\zdqd...

REM 清理目标目录（保留 models 文件夹）
if exist "D:\zdqd\XiMengHelper.exe" del /q "D:\zdqd\XiMengHelper.exe"
if exist "D:\zdqd\_internal" rmdir /s /q "D:\zdqd\_internal"
if exist "D:\zdqd\config" rmdir /s /q "D:\zdqd\config"
if exist "D:\zdqd\data" rmdir /s /q "D:\zdqd\data"

REM 复制 EXE 和 _internal 文件夹
echo 复制 EXE 和依赖库...
xcopy "dist\XiMengHelper\XiMengHelper.exe" "D:\zdqd\" /Y /Q
xcopy "dist\XiMengHelper\_internal" "D:\zdqd\_internal\" /E /I /Y /Q

REM 复制配置文件夹
echo 复制配置文件...
xcopy "config" "D:\zdqd\config\" /E /I /Y /Q

REM 复制数据文件夹
echo 复制数据文件...
xcopy "data" "D:\zdqd\data\" /E /I /Y /Q

REM 复制根目录配置文件
if exist "config.yaml" copy "config.yaml" "D:\zdqd\" /Y >nul
if exist ".env" copy ".env" "D:\zdqd\" /Y >nul

REM [2026-02-24] 修复原因：自动复制 models 文件夹，解决安装包缺失模型文件的问题
echo 复制模型文件...
if exist "models" (
    xcopy "models" "D:\zdqd\models\" /E /I /Y /Q
    echo [OK] 模型文件复制完成
) else (
    echo [警告] models 文件夹不存在！
    mkdir "D:\zdqd\models"
    echo [提示] 请手动复制 YOLO 模型文件到 D:\zdqd\models\
)

echo [OK] 复制完成
echo.

echo ========================================
echo 构建完成！
echo ========================================
echo.
echo 输出目录: D:\zdqd
echo.
echo 目录结构:
echo   XiMengHelper.exe        ^<-- 主程序
echo   _internal\              ^<-- 依赖库（numpy、cv2等）
echo   config\                 ^<-- 配置文件
echo   models\                 ^<-- YOLO 模型
echo   data\                   ^<-- 数据文件
echo   config.yaml             ^<-- 主配置
echo   .env                    ^<-- 环境变量
echo.

REM 询问是否自动运行测试
echo [测试] 是否自动运行程序测试？
echo   按任意键 = 自动运行测试
echo   Ctrl+C = 跳过测试
pause >nul

echo.
echo [测试] 正在启动程序...
echo [提示] 程序将在新窗口中打开
echo [提示] 如果程序正常打开GUI界面，说明构建成功
echo [提示] 如果出现错误，请查看控制台输出
echo.

REM 启动程序（在新窗口中运行，这样可以看到输出）
start "溪盟商城自动化助手" "D:\zdqd\XiMengHelper.exe"

echo [OK] 程序已启动
echo.
echo 请检查程序窗口：
echo   ✓ GUI界面正常显示 = 构建成功
echo   ✗ 出现错误或闪退 = 构建失败，请查看错误信息
echo.

pause
