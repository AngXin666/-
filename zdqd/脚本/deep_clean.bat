@echo off
chcp 65001 >nul
echo ========================================
echo 深度清理缓存
echo ========================================
echo.
echo ⚠️  警告: 此操作将清理所有缓存，包括:
echo   - Python字节码缓存 (__pycache__)
echo   - 登录缓存 (login_cache)
echo   - 运行时数据 (runtime_data)
echo   - 模型缓存 (如果存在)
echo   - 日志文件 (可选)
echo.
echo 是否继续？ [Y/N]
choice /C YN /N /M "请选择: "
if errorlevel 2 goto cancel
if errorlevel 1 goto start_clean

:cancel
echo.
echo 操作已取消
pause
exit /b 0

:start_clean
echo.
echo ========================================
echo 开始清理...
echo ========================================

REM 1. 终止所有Python进程
echo.
echo [1/6] 终止Python进程...
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    taskkill /F /IM python.exe >nul 2>&1
    timeout /t 2 /nobreak >nul
    echo ✓ Python进程已终止
) else (
    echo ✓ 没有运行中的Python进程
)

REM 2. 清理Python缓存
echo.
echo [2/6] 清理Python字节码缓存...
set count=0
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" (
        rd /s /q "%%d" 2>nul
        set /a count+=1
    )
)
echo ✓ 清理了 %count% 个 __pycache__ 目录

REM 3. 清理.pyc文件
echo.
echo [3/6] 清理.pyc文件...
del /s /q "*.pyc" >nul 2>&1
echo ✓ .pyc文件已清理

REM 4. 清理登录缓存
echo.
echo [4/6] 清理登录缓存...
if exist "login_cache" (
    del /q "login_cache\*.json" 2>nul
    echo ✓ 登录缓存已清理
) else (
    echo ✓ 没有登录缓存
)

REM 5. 清理运行时数据
echo.
echo [5/6] 清理运行时数据...
if exist "runtime_data" (
    del /q "runtime_data\*.db-shm" 2>nul
    del /q "runtime_data\*.db-wal" 2>nul
    echo ✓ 运行时数据已清理
) else (
    echo ✓ 没有运行时数据
)

REM 6. 询问是否清理日志
echo.
echo [6/6] 是否清理日志文件？ [Y/N]
choice /C YN /N /M "请选择: "
if errorlevel 2 goto skip_logs
if errorlevel 1 (
    if exist "logs" (
        echo 清理日志文件...
        del /q "logs\*.log" 2>nul
        echo ✓ 日志文件已清理
    ) else (
        echo ✓ 没有日志文件
    )
)
:skip_logs

REM 完成
echo.
echo ========================================
echo ✓ 清理完成！
echo ========================================
echo.
echo 是否立即启动程序？ [Y/N]
choice /C YN /N /M "请选择: "
if errorlevel 2 goto end
if errorlevel 1 (
    echo.
    echo 正在启动程序...
    python run.py
)

:end
echo.
pause
