"""
自动注册新页面类型工具

功能：
1. 扫描 page_classes.json 中的所有类别
2. 检查 page_state_mapping.json 中是否已映射
3. 自动为未映射的类别生成配置
4. 提示用户确认并更新配置文件
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def load_page_classes(models_dir: Path):
    """加载页面类别列表"""
    classes_path = models_dir / "page_classes.json"
    if not classes_path.exists():
        print(f"❌ 错误: 找不到 {classes_path}")
        return []
    
    with open(classes_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_state_mapping(config_dir: Path):
    """加载页面状态映射配置"""
    mapping_path = config_dir / "page_state_mapping.json"
    if not mapping_path.exists():
        print(f"❌ 错误: 找不到 {mapping_path}")
        return None
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_state_mapping(config_dir: Path, mapping_config: dict):
    """保存页面状态映射配置"""
    mapping_path = config_dir / "page_state_mapping.json"
    
    # 备份原文件
    if mapping_path.exists():
        backup_path = config_dir / f"page_state_mapping.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        import shutil
        shutil.copy(mapping_path, backup_path)
        print(f"✓ 已备份原配置: {backup_path.name}")
    
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_config, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已保存配置: {mapping_path}")


def generate_state_name(class_name: str):
    """根据类别名称生成状态名称
    
    Args:
        class_name: 类别名称（中文）
        
    Returns:
        (STATE, state_value) 元组
    """
    # 简单的映射规则
    name_map = {
        '页': 'PAGE',
        '弹窗': 'POPUP',
        '广告': 'AD',
        '流水': 'HISTORY',
        '桌面': 'LAUNCHER',
        '提示': 'TIP',
    }
    
    # 生成英文状态名
    state = class_name.upper()
    for cn, en in name_map.items():
        if cn in class_name:
            state = state.replace(cn.upper(), f'_{en}')
    
    # 清理状态名
    state = state.replace('_', '_').strip('_')
    if not state:
        state = 'UNKNOWN'
    
    # 生成state_value（小写+下划线）
    state_value = state.lower()
    
    return state, state_value


def find_unmapped_classes(page_classes: list, mapping_config: dict):
    """查找未映射的类别
    
    Args:
        page_classes: 页面类别列表
        mapping_config: 映射配置
        
    Returns:
        未映射的类别列表
    """
    mapped_classes = set(mapping_config.get('mappings', {}).keys())
    unmapped = [cls for cls in page_classes if cls not in mapped_classes]
    return unmapped


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🔍 自动注册新页面类型")
    print("=" * 80)
    
    # 确定项目根目录
    script_dir = Path(__file__).parent.parent
    models_dir = script_dir / "models"
    config_dir = script_dir / "config"
    
    print(f"\n📁 项目目录: {script_dir}")
    print(f"📁 模型目录: {models_dir}")
    print(f"📁 配置目录: {config_dir}")
    
    # 加载页面类别
    print(f"\n📦 加载页面类别...")
    page_classes = load_page_classes(models_dir)
    if not page_classes:
        return
    
    print(f"  ✓ 找到 {len(page_classes)} 个页面类别")
    
    # 加载状态映射
    print(f"\n📦 加载状态映射配置...")
    mapping_config = load_state_mapping(config_dir)
    if not mapping_config:
        return
    
    mapped_count = len(mapping_config.get('mappings', {}))
    print(f"  ✓ 已映射 {mapped_count} 个类别")
    
    # 查找未映射的类别
    print(f"\n🔍 检查未映射的类别...")
    unmapped = find_unmapped_classes(page_classes, mapping_config)
    
    if not unmapped:
        print(f"  ✓ 所有类别都已映射！")
        return
    
    print(f"  ⚠️  发现 {len(unmapped)} 个未映射的类别:")
    for cls in unmapped:
        print(f"    • {cls}")
    
    # 生成新映射
    print(f"\n🔧 生成新映射配置...")
    new_mappings = {}
    for class_name in unmapped:
        state, state_value = generate_state_name(class_name)
        new_mappings[class_name] = {
            "state": state,
            "state_value": state_value,
            "chinese_name": class_name,
            "description": f"{class_name}（自动生成）"
        }
        print(f"  • {class_name}")
        print(f"    - STATE: {state}")
        print(f"    - state_value: {state_value}")
    
    # 询问用户是否保存
    print(f"\n" + "=" * 80)
    print(f"⚠️  注意事项:")
    print(f"  1. 自动生成的状态名称可能不准确，建议手动检查")
    print(f"  2. 需要在 src/page_detector.py 中添加对应的 PageState 枚举")
    print(f"  3. 原配置文件会自动备份")
    print(f"=" * 80)
    
    response = input(f"\n是否将新映射添加到配置文件？(y/n): ").strip().lower()
    
    if response != 'y':
        print(f"\n❌ 已取消")
        return
    
    # 合并映射
    mapping_config['mappings'].update(new_mappings)
    
    # 保存配置
    print(f"\n💾 保存配置...")
    save_state_mapping(config_dir, mapping_config)
    
    print(f"\n" + "=" * 80)
    print(f"✅ 完成！")
    print(f"=" * 80)
    
    print(f"\n📝 后续步骤:")
    print(f"  1. 检查 config/page_state_mapping.json 中的新映射")
    print(f"  2. 在 src/page_detector.py 中添加对应的 PageState 枚举:")
    print(f"")
    for class_name, config in new_mappings.items():
        state = config['state']
        state_value = config['state_value']
        chinese_name = config['chinese_name']
        print(f"     {state} = \"{state_value}\"  # {chinese_name}")
    print(f"")
    print(f"  3. 在 PageState.chinese_name 属性中添加中文名称映射:")
    print(f"")
    for class_name, config in new_mappings.items():
        state_value = config['state_value']
        chinese_name = config['chinese_name']
        print(f"     \"{state_value}\": \"{chinese_name}\",")
    print(f"")
    print(f"  4. 重启程序测试")
    print(f"")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户取消")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
