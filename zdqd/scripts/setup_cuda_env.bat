@echo off
REM 设置CUDA环境变量

set CUDA_PATH=C:\Program Files\Python311\Lib\site-packages\nvidia\cuda_runtime
set CUDA_HOME=%CUDA_PATH%

REM 添加CUDA库到PATH
set PATH=C:\Program Files\Python311\Lib\site-packages\nvidia\cublas\bin;%PATH%
set PATH=C:\Program Files\Python311\Lib\site-packages\nvidia\cuda_cupti\bin;%PATH%
set PATH=C:\Program Files\Python311\Lib\site-packages\nvidia\cuda_nvrtc\bin;%PATH%
set PATH=C:\Program Files\Python311\Lib\site-packages\nvidia\cuda_runtime\bin;%PATH%
set PATH=C:\Program Files\Python311\Lib\site-packages\nvidia\cudnn\bin;%PATH%

echo CUDA环境变量已设置
echo CUDA_PATH=%CUDA_PATH%
echo.
echo 现在可以运行训练脚本了
echo.

REM 运行传入的命令
%*
