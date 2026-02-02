@echo off
chcp 65001 >nul
echo ========================================
echo 清理Python缓存并启动程序
echo ========================================
echo.

echo [1/3] 清理 __pycache__ 文件夹（保留账号缓存）...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul

echo.
echo [2/3] 清理 .pyc 和 .pyo 文件（保留账号缓存）...
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
echo   ✓ 已清理所有代码缓存文件

echo.
echo [3/3] 启动程序...
echo ========================================
echo.

python run.py
pause
