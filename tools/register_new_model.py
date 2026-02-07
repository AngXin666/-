"""
自动注册新模型到registry

用法：
    python tools/register_new_model.py <model_key> <model_name> <page_type> <model_path> <classes>
    
示例：
    python tools/register_new_model.py coupon_detector "优惠券页检测模型" "优惠券页" "yolo_runs/coupon_detector/weights/best.pt" "返回按钮,优惠券卡片"
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def register_model(model_key, model_name, page_type, model_path, classes, notes=""):
    """注册新模型到yolo_model_registry.json
    
    Args:
        model_key: 模型唯一标识符
        model_name: 模型中文名称
        page_type: 页面类型
        model_path: 模型文件路径（相对于models目录）
        classes: 类别列表
        notes: 备注说明
    """
    registry_path = Path("models/yolo_model_registry.json")
    
    if not registry_path.exists():
        print(f"❌ 注册表文件不存在: {registry_path}")
        return False
    
    # 读取现有注册表
    try:
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
    except Exception as e:
        print(f"❌ 读取注册表失败: {e}")
        return False
    
    # 检查模型是否已存在
    if model_key in registry['models']:
        print(f"⚠️  模型 '{model_key}' 已存在")
        response = input("是否覆盖? (y/n): ")
        if response.lower() != 'y':
            print("取消注册")
            return False
    
    # 检查模型文件是否存在
    full_model_path = Path("models") / model_path
    if not full_model_path.exists():
        print(f"⚠️  警告: 模型文件不存在: {full_model_path}")
        response = input("是否继续注册? (y/n): ")
        if response.lower() != 'y':
            print("取消注册")
            return False
    
    # 添加新模型
    registry['models'][model_key] = {
        "name": model_name,
        "page_type": page_type,
        "model_path": model_path,
        "classes": classes,
        "num_classes": len(classes),
        "performance": {
            "mAP50": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mAP50-95": 0.0
        },
        "training_date": datetime.now().strftime("%Y-%m-%d"),
        "dataset_size": {
            "original": 0,
            "augmented": 0,
            "train": 0,
            "val": 0
        },
        "notes": notes or "新添加的模型，请更新性能指标和数据集信息"
    }
    
    # 保存
    try:
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        print(f"✅ 模型已注册: {model_key}")
        print(f"   名称: {model_name}")
        print(f"   页面类型: {page_type}")
        print(f"   类别数: {len(classes)}")
        print(f"   类别: {', '.join(classes)}")
        print()
        print("⚠️  请手动更新以下信息:")
        print("   1. 性能指标 (mAP50, precision, recall)")
        print("   2. 数据集大小 (original, augmented, train, val)")
        print("   3. 备注说明")
        print()
        print("下一步:")
        print(f"   1. 编辑 models/page_yolo_mapping.json 添加页面映射")
        print(f"   2. 运行: python tools/generate_model_version.py --version 1.0.x")
        return True
    except Exception as e:
        print(f"❌ 保存注册表失败: {e}")
        return False


def add_page_mapping(page_name, page_state, model_key, purpose, priority=1):
    """添加页面映射到page_yolo_mapping.json
    
    Args:
        page_name: 页面名称
        page_state: 页面状态枚举
        model_key: 模型标识符
        purpose: 模型用途说明
        priority: 优先级
    """
    mapping_path = Path("models/page_yolo_mapping.json")
    
    if not mapping_path.exists():
        print(f"❌ 映射文件不存在: {mapping_path}")
        return False
    
    # 读取现有映射
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取映射文件失败: {e}")
        return False
    
    # 检查页面是否已存在
    if page_name in mapping_data['mapping']:
        print(f"⚠️  页面 '{page_name}' 已存在")
        # 添加到现有页面的模型列表
        existing_models = mapping_data['mapping'][page_name]['yolo_models']
        existing_keys = [m['model_key'] for m in existing_models]
        
        if model_key in existing_keys:
            print(f"⚠️  模型 '{model_key}' 已在该页面中")
            return False
        
        # 添加新模型
        existing_models.append({
            "model_key": model_key,
            "purpose": purpose,
            "priority": priority
        })
        print(f"✅ 已将模型添加到现有页面: {page_name}")
    else:
        # 创建新页面映射
        mapping_data['mapping'][page_name] = {
            "page_state": page_state,
            "yolo_models": [
                {
                    "model_key": model_key,
                    "purpose": purpose,
                    "priority": priority
                }
            ]
        }
        print(f"✅ 已创建新页面映射: {page_name}")
    
    # 保存
    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"❌ 保存映射文件失败: {e}")
        return False


def main():
    """主函数"""
    if len(sys.argv) < 6:
        print("=" * 60)
        print("自动注册新模型工具")
        print("=" * 60)
        print()
        print("用法:")
        print("  python tools/register_new_model.py <model_key> <model_name> <page_type> <model_path> <classes> [notes]")
        print()
        print("参数:")
        print("  model_key   : 模型唯一标识符 (如: coupon_detector)")
        print("  model_name  : 模型中文名称 (如: 优惠券页检测模型)")
        print("  page_type   : 页面类型 (如: 优惠券页)")
        print("  model_path  : 模型路径 (如: yolo_runs/coupon_detector/weights/best.pt)")
        print("  classes     : 类别列表，逗号分隔 (如: 返回按钮,优惠券卡片)")
        print("  notes       : 备注说明 (可选)")
        print()
        print("示例:")
        print('  python tools/register_new_model.py coupon_detector "优惠券页检测模型" "优惠券页" "yolo_runs/coupon_detector/weights/best.pt" "返回按钮,优惠券卡片"')
        print()
        sys.exit(1)
    
    model_key = sys.argv[1]
    model_name = sys.argv[2]
    page_type = sys.argv[3]
    model_path = sys.argv[4]
    classes = [c.strip() for c in sys.argv[5].split(',')]
    notes = sys.argv[6] if len(sys.argv) > 6 else ""
    
    print("=" * 60)
    print("注册新模型")
    print("=" * 60)
    print(f"模型标识: {model_key}")
    print(f"模型名称: {model_name}")
    print(f"页面类型: {page_type}")
    print(f"模型路径: {model_path}")
    print(f"类别列表: {', '.join(classes)}")
    if notes:
        print(f"备注: {notes}")
    print()
    
    # 注册模型
    if register_model(model_key, model_name, page_type, model_path, classes, notes):
        print()
        print("是否同时添加页面映射? (y/n): ", end='')
        response = input()
        
        if response.lower() == 'y':
            print()
            page_state = input(f"页面状态枚举 (如: COUPON_PAGE): ").strip()
            purpose = input(f"模型用途说明 (如: 检测返回按钮和优惠券卡片): ").strip()
            priority = input(f"优先级 (默认: 1): ").strip()
            priority = int(priority) if priority else 1
            
            if add_page_mapping(page_type, page_state, model_key, purpose, priority):
                print()
                print("✅ 页面映射已添加")
        
        print()
        print("=" * 60)
        print("注册完成！")
        print("=" * 60)
        return 0
    else:
        print()
        print("❌ 注册失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())
