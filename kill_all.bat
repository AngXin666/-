@echo off
chcp 65001 >nul
echo ========================================
echo 强制结束所有相关进程
echo ========================================
echo.

echo 正在结束进程...
echo.

REM 结束主程序
taskkill /F /IM "自动签到助手_v1.9.3.exe" 2>nul
taskkill /F /IM "自动签到助手_v1.9.2.exe" 2>nul
taskkill /F /IM "自动签到助手_v1.9.1.exe" 2>nul
taskkill /F /IM "AutoSignHelper*.exe" 2>nul

REM 结束 wmic.exe
taskkill /F /IM "wmic.exe" 2>nul

REM 结束 mshta.exe (HTA 对话框)
taskkill /F /IM "mshta.exe" 2>nul

REM 结束 Python 进程
taskkill /F /IM "python.exe" 2>nul
taskkill /F /IM "pythonw.exe" 2>nul

echo.
echo ========================================
echo 所有进程已结束
echo ========================================
echo.

pause
