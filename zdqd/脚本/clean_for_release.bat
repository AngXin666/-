@echo off
chcp 65001 >nul
echo ========================================
echo 清理敏感数据 - 准备发布给新用户
echo ========================================
echo.

echo 此脚本将删除以下文件/目录:
echo   - dist\runtime_data\license.db (许可证数据库)
echo   - dist\.account_cache.json (账号缓存)
echo   - dist\login_cache (登录缓存)
echo   - dist\.env (配置文件，包含 Supabase 密钥)
echo.
echo ⚠️  警告: 此操作不可恢复！
echo.
set /p confirm="确认清理? (Y/N): "

if /i not "%confirm%"=="Y" (
    echo.
    echo 已取消清理
    pause
    exit /b 0
)

echo.
echo 开始清理...
echo.

REM 删除数据库文件
if exist "dist\runtime_data\license.db" (
    del /q "dist\runtime_data\license.db"
    echo ✅ 已删除数据库文件
) else (
    echo ⚠️  数据库文件不存在
)

REM 删除账号缓存
if exist "dist\.account_cache.json" (
    del /q "dist\.account_cache.json"
    echo ✅ 已删除账号缓存
) else (
    echo ⚠️  账号缓存不存在
)

REM 删除登录缓存目录
if exist "dist\login_cache" (
    rmdir /s /q "dist\login_cache"
    echo ✅ 已删除登录缓存目录
) else (
    echo ⚠️  登录缓存目录不存在
)

REM 删除配置文件
if exist "dist\.env" (
    del /q "dist\.env"
    echo ✅ 已删除配置文件
) else (
    echo ⚠️  配置文件不存在
)

REM 删除备份目录
if exist "dist\runtime_data\backups" (
    rmdir /s /q "dist\runtime_data\backups"
    echo ✅ 已删除备份目录
)

echo.
echo 正在创建新用户环境...
echo.

REM 创建空的数据库
python -c "import sys; sys.path.insert(0, 'src'); import os; os.chdir('dist'); from local_db import LocalDatabase; db = LocalDatabase(); print('✅ 已创建空数据库')"

REM 创建示例账号文件
echo 13800138000----password123 > "dist\账号.txt"
echo 13900139000----password456 >> "dist\账号.txt"
echo ✅ 已创建示例账号文件 (dist\账号.txt)

REM 创建登录缓存目录
if not exist "dist\login_cache" mkdir "dist\login_cache"
echo ✅ 已创建登录缓存目录

echo.
echo ========================================
echo ✅ 清理完成！
echo ========================================
echo.
echo 当前 dist 目录包含:
echo   ✅ 自动签到助手_v2.0.6.exe (主程序)
echo   ✅ JT 目录 (模板文件)
echo   ✅ runtime_data 目录 (空数据库)
echo   ✅ 账号.txt (示例账号文件)
echo   ✅ login_cache 目录 (空目录)
echo.
echo 📦 可以安全地发布给新用户了！
echo.
echo 新用户首次运行时:
echo   1. 需要自己激活许可证
echo   2. 需要配置模拟器路径
echo   3. 可以修改 账号.txt 添加自己的账号
echo   4. 所有数据将保存在他们自己的 runtime_data 目录
echo   5. 目录结构已经准备好，无需手动创建
echo.

pause
