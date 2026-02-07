@echo off
chcp 65001 >nul
echo ========================================
echo    图像标注工具
echo ========================================
echo.
echo 正在启动标注工具...
echo.

python standalone_annotation_tool.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo 启动失败!
    echo ========================================
    echo.
    echo 可能的原因:
    echo 1. Python未安装或未添加到PATH
    echo 2. 缺少必要的库 (pillow)
    echo.
    echo 解决方法:
    echo 1. 安装Python 3.7+
    echo 2. 运行: pip install pillow
    echo.
    pause
) else (
    echo.
    echo 程序已退出
)
