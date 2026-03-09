@echo off
chcp 65001 >nul
echo ========================================
echo 清理并整理 D:\zdqd 目录
echo ========================================
echo.

echo [1/3] 清理根目录的依赖库文件...
del /q "D:\zdqd\*.dll" >nul 2>&1
del /q "D:\zdqd\*.pyd" >nul 2>&1
del /q "D:\zdqd\base_library.zip" >nul 2>&1

echo [2/3] 删除根目录的依赖库文件夹...
for /d %%d in ("D:\zdqd\*") do (
    if /i not "%%~nxd"=="models" (
        if /i not "%%~nxd"=="runtime_data" (
            if /i not "%%~nxd"=="logs" (
                if /i not "%%~nxd"=="login_cache" (
                    if /i not "%%~nxd"==".kiro" (
                        if /i not "%%~nxd"=="_internal" (
                            if /i not "%%~nxd"=="config" (
                                if /i not "%%~nxd"=="data" (
                                    echo 删除: %%~nxd
                                    rmdir /s /q "%%d" >nul 2>&1
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)

echo [3/3] 验证目录结构...
echo.
echo 当前 D:\zdqd 目录结构:
dir /b "D:\zdqd"
echo.
echo [OK] 清理完成！
echo.
echo 保留的文件和文件夹:
echo   XiMengHelper.exe
echo   _internal\
echo   config\
echo   data\
echo   models\
echo   config.yaml
echo   .env
echo.
pause
