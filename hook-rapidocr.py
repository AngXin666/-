# PyInstaller hook for rapidocr
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

# 收集rapidocr的所有数据文件（包括yaml配置文件和ONNX模型）
datas = collect_data_files('rapidocr', include_py_files=False)

# 明确添加必需的文件
try:
    import rapidocr
    rapidocr_path = os.path.dirname(rapidocr.__file__)
    
    # 必需的文件列表
    required_files = [
        ('config.yaml', 'rapidocr'),
        ('default_models.yaml', 'rapidocr'),  # 添加default_models.yaml
        ('models/ch_PP-OCRv4_det_infer.onnx', 'rapidocr/models'),
        ('models/ch_PP-OCRv4_rec_infer.onnx', 'rapidocr/models'),
        ('models/ch_ppocr_mobile_v2.0_cls_infer.onnx', 'rapidocr/models'),
        ('inference_engine/pytorch/networks/arch_config.yaml', 'rapidocr/inference_engine/pytorch/networks'),
    ]
    
    for src_rel, dst in required_files:
        src = os.path.join(rapidocr_path, src_rel)
        if os.path.exists(src):
            datas.append((src, dst))
            print(f"[hook-rapidocr] 添加文件: {src_rel}")
        else:
            print(f"[hook-rapidocr] 警告: 找不到文件: {src_rel}")
            
except Exception as e:
    print(f"[hook-rapidocr] 警告: 无法添加rapidocr文件: {e}")

# 收集所有子模块
hiddenimports = collect_submodules('rapidocr')

