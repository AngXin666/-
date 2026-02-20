"""最小化打包脚本 - 用于诊断"""
import subprocess
import sys
import os
import shutil

print("=" * 60)
print("最小化打包 - 诊断版本")
print("=" * 60)

# 清理旧文件
if os.path.exists("build"):
    shutil.rmtree("build")
if os.path.exists("dist"):
    shutil.rmtree("dist")

# 打包命令
cmd = [
    "pyinstaller",
    "--name", "test_minimal",
    "--windowed",
    "--onedir",
    "--clean",
    "--noconfirm",
    "--paths", "src",
    "--add-data", "src;_internal/src",
    "--add-data", "config;config",
    "--add-data", "models;models",
    "--add-data", "config.yaml;.",
    "--hidden-import", "torch",
    "--hidden-import", "PIL",
    "--hidden-import", "src.adb_bridge",
    "--hidden-import", "src.page_detector_integrated",
    "--hidden-import", "src.model_manager",
    "--hidden-import", "src.page_state_dynamic",
    "--hidden-import", "src.ocr_thread_pool",
    "--collect-submodules", "src",
    "run_minimal.py"
]

print("\n执行打包...")
result = subprocess.run(cmd, capture_output=False)

if result.returncode == 0:
    print("\n✓ 打包完成")
    
    # 复制到测试目录
    output_dir = "D:/test_minimal"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    shutil.copytree("dist/test_minimal", output_dir)
    
    # 复制config和models
    if os.path.exists("config"):
        shutil.copytree("config", os.path.join(output_dir, "config"), dirs_exist_ok=True)
    if os.path.exists("models"):
        shutil.copytree("models", os.path.join(output_dir, "models"), dirs_exist_ok=True)
    if os.path.exists("config.yaml"):
        shutil.copy("config.yaml", output_dir)
    
    print(f"\n输出目录: {output_dir}")
    print(f"可执行文件: {output_dir}/test_minimal.exe")
    
else:
    print("\n✗ 打包失败")
    sys.exit(1)
