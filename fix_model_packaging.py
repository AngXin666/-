#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复模型打包问题
确保所有必需的文件都被正确打包
"""

import os
import sys
import json
import shutil
from pathlib import Path

def check_and_fix_config_files():
    """检查并修复配置文件"""
    print("=" * 70)
    print("【1】检查配置文件")
    print("=" * 70)
    
    # 检查必需的配置文件
    required_files = {
        'config/yolo_model_registry.json': 'YOLO模型注册表',
        'models/page_yolo_mapping.json': '页面-YOLO映射',
        'config/page_state_mapping.json': '页面状态映射',
        'config/page_classes.json': '页面类别列表',
        'models/page_classifier_pytorch_best.pth': '页面分类器模型',
    }
    
    missing_files = []
    for filepath, description in required_files.items():
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"✓ {description}: {filepath} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {description}: {filepath} (缺失)")
            missing_files.append(filepath)
    
    if missing_files:
        print(f"\n❌ 缺失 {len(missing_files)} 个文件，无法继续打包")
        return False
    
    print("\n✓ 所有配置文件都存在")
    return True

def check_yolo_models():
    """检查YOLO模型文件"""
    print("\n" + "=" * 70)
    print("【2】检查YOLO模型文件")
    print("=" * 70)
    
    # 读取注册表
    registry_path = 'config/yolo_model_registry.json'
    try:
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取注册表: {e}")
        return False
    
    models = registry.get('models', {})
    print(f"注册表中有 {len(models)} 个模型")
    
    missing_models = []
    for model_name, model_info in models.items():
        model_path = model_info.get('model_path', '')
        if not model_path:
            continue
        
        # 构建完整路径
        full_path = Path('models') / model_path
        
        if full_path.exists():
            size_mb = full_path.stat().st_size / 1024 / 1024
            print(f"✓ {model_name}: {model_path} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {model_name}: {model_path} (缺失)")
            missing_models.append((model_name, model_path))
    
    if missing_models:
        print(f"\n❌ 缺失 {len(missing_models)} 个YOLO模型")
        for name, path in missing_models:
            print(f"  - {name}: {path}")
        return False
    
    print(f"\n✓ 所有YOLO模型文件都存在")
    return True

def update_build_script():
    """更新打包脚本，确保包含所有必需文件"""
    print("\n" + "=" * 70)
    print("【3】检查打包脚本")
    print("=" * 70)
    
    build_script = 'build_exe_optimized.py'
    
    if not os.path.exists(build_script):
        print(f"❌ 打包脚本不存在: {build_script}")
        return False
    
    with open(build_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查关键配置
    checks = {
        "('config', 'config')": "config目录",
        "('models', 'models')": "models目录",
        "folders_to_fix = ['config', 'models']": "文件结构修复",
    }
    
    all_ok = True
    for check_str, description in checks.items():
        if check_str in content:
            print(f"✓ {description}: 已配置")
        else:
            print(f"❌ {description}: 未配置")
            all_ok = False
    
    if all_ok:
        print("\n✓ 打包脚本配置正确")
    else:
        print("\n❌ 打包脚本需要更新")
    
    return all_ok

def create_test_script():
    """创建打包后测试脚本"""
    print("\n" + "=" * 70)
    print("【4】创建测试脚本")
    print("=" * 70)
    
    test_script = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
打包后模型加载测试脚本
将此脚本复制到打包后的目录中运行
\"\"\"

import os
import sys
import json
from pathlib import Path

def test_model_loading():
    print("测试模型加载...")
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    print(f"是否打包: {getattr(sys, 'frozen', False)}")
    
    if getattr(sys, 'frozen', False):
        base_dir = Path(sys.executable).parent
    else:
        base_dir = Path(__file__).parent
    
    print(f"基础目录: {base_dir}")
    
    # 检查目录
    print("\\n检查目录:")
    for dir_name in ['config', 'models', '_internal']:
        dir_path = base_dir / dir_name
        exists = dir_path.exists()
        status = "✓" if exists else "❌"
        print(f"  {status} {dir_name}/: {exists}")
    
    # 检查关键文件
    print("\\n检查关键文件:")
    files_to_check = [
        'config/yolo_model_registry.json',
        'models/page_yolo_mapping.json',
        'config/page_state_mapping.json',
        'models/page_classifier_pytorch_best.pth',
    ]
    
    for filepath in files_to_check:
        full_path = base_dir / filepath
        exists = full_path.exists()
        status = "✓" if exists else "❌"
        
        if exists:
            size_mb = full_path.stat().st_size / 1024 / 1024
            print(f"  {status} {filepath} ({size_mb:.2f} MB)")
        else:
            print(f"  {status} {filepath} (缺失)")
    
    # 尝试加载模型管理器
    print("\\n尝试加载模型管理器:")
    try:
        sys.path.insert(0, str(base_dir))
        from src.model_manager import ModelManager
        
        manager = ModelManager.get_instance()
        print("  ✓ ModelManager初始化成功")
        print(f"  - 基础目录: {manager.base_dir}")
        print(f"  - 模型目录: {manager.models_dir}")
        print(f"  - 模型目录存在: {manager.models_dir.exists()}")
        
    except Exception as e:
        print(f"  ❌ ModelManager初始化失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n测试完成")
    input("按回车键退出...")

if __name__ == '__main__':
    test_model_loading()
"""
    
    test_script_path = 'test_packed_models.py'
    with open(test_script_path, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print(f"✓ 已创建测试脚本: {test_script_path}")
    print("  打包后将此脚本复制到EXE目录中运行")
    
    return True

def generate_packaging_checklist():
    """生成打包检查清单"""
    print("\n" + "=" * 70)
    print("【5】打包检查清单")
    print("=" * 70)
    
    checklist = """
# 打包前检查清单

## 1. 文件完整性检查
- [ ] config/yolo_model_registry.json 存在
- [ ] models/page_yolo_mapping.json 存在
- [ ] config/page_state_mapping.json 存在
- [ ] config/page_classes.json 存在
- [ ] models/page_classifier_pytorch_best.pth 存在
- [ ] 所有YOLO模型文件存在（检查注册表中的路径）

## 2. 打包脚本检查
- [ ] build_exe_optimized.py 包含 ('config', 'config')
- [ ] build_exe_optimized.py 包含 ('models', 'models')
- [ ] build_exe_optimized.py 包含文件结构修复逻辑

## 3. 打包步骤
1. 运行: python build_exe_optimized.py
2. 等待打包完成
3. 检查 dist/溪盟商城自动化助手/ 目录

## 4. 打包后检查
- [ ] dist/溪盟商城自动化助手/config/ 目录存在
- [ ] dist/溪盟商城自动化助手/models/ 目录存在
- [ ] config/yolo_model_registry.json 在根目录的config下
- [ ] models/page_yolo_mapping.json 在根目录的models下
- [ ] 所有YOLO模型文件在根目录的models下

## 5. 功能测试
1. 复制 test_packed_models.py 到打包目录
2. 运行测试脚本
3. 检查所有文件是否能正确找到
4. 启动主程序测试模型加载

## 6. 常见问题
- 如果models在_internal下：检查copy_additional_files()函数
- 如果config在_internal下：检查copy_additional_files()函数
- 如果模型加载失败：运行diagnose_model_loading.py诊断
"""
    
    checklist_path = 'PACKAGING_CHECKLIST_MODELS.md'
    with open(checklist_path, 'w', encoding='utf-8') as f:
        f.write(checklist.strip())
    
    print(f"✓ 已生成检查清单: {checklist_path}")
    
    return True

def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 22 + "模型打包修复工具" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        # 1. 检查配置文件
        if not check_and_fix_config_files():
            print("\n❌ 配置文件检查失败，请先解决缺失文件问题")
            input("\n按回车键退出...")
            return
        
        # 2. 检查YOLO模型
        if not check_yolo_models():
            print("\n❌ YOLO模型检查失败，请先解决缺失模型问题")
            input("\n按回车键退出...")
            return
        
        # 3. 检查打包脚本
        if not update_build_script():
            print("\n⚠ 打包脚本可能需要更新")
        
        # 4. 创建测试脚本
        create_test_script()
        
        # 5. 生成检查清单
        generate_packaging_checklist()
        
        print("\n" + "=" * 70)
        print("✓ 所有检查完成")
        print("=" * 70)
        print("\n下一步:")
        print("  1. 运行: python build_exe_optimized.py")
        print("  2. 打包完成后，复制 test_packed_models.py 到打包目录")
        print("  3. 在打包目录中运行 test_packed_models.py 测试")
        print("  4. 如果有问题，运行 diagnose_model_loading.py 诊断")
        
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n按回车键退出...")
    input()

if __name__ == '__main__':
    main()
