@echo off
echo ========================================
echo 安装TensorFlow GPU支持
echo ========================================
echo.

echo [1/3] 卸载现有TensorFlow...
pip uninstall -y tensorflow tensorflow-intel keras tensorboard

echo.
echo [2/3] 安装TensorFlow 2.15（最后一个稳定的GPU版本）...
pip install tensorflow==2.15.0

echo.
echo [3/3] 验证GPU支持...
python -c "import tensorflow as tf; print('TensorFlow版本:', tf.__version__); print('GPU可用:', tf.config.list_physical_devices('GPU'))"

echo.
echo ========================================
echo 安装完成！
echo ========================================
pause
