"""
批量删除已完成训练的training_data文件夹
只删除已保存原始图且已注册模型的文件夹
"""
import shutil
from pathlib import Path


def main():
    # 可以安全删除的文件夹列表
    folders_to_delete = [
        "个人页_未登录",
        "交易流水",
        "分类页",
        "分类页_temp_augmented",
        "我的优惠劵",
        "转账确认弹窗"
    ]
    
    print(f"{'='*60}")
    print(f"批量删除已完成训练的training_data文件夹")
    print(f"{'='*60}\n")
    
    print(f"将要删除以下文件夹：\n")
    for folder in folders_to_delete:
        print(f"  - training_data/{folder}")
    
    # 确认
    print(f"\n⚠️  警告：此操作不可恢复！")
    print(f"这些文件夹的原始标注图已保存到 原始标注图/ 目录")
    print(f"模型已注册到 yolo_model_registry.json")
    
    confirm = input(f"\n确认删除？(输入 yes 继续): ")
    
    if confirm.lower() != 'yes':
        print(f"\n❌ 已取消删除")
        return
    
    # 开始删除
    print(f"\n🗑️  开始删除...\n")
    
    deleted_count = 0
    failed_count = 0
    
    for folder in folders_to_delete:
        folder_path = Path(f"training_data/{folder}")
        
        if not folder_path.exists():
            print(f"  ⚠ 跳过（不存在）: {folder}")
            continue
        
        try:
            shutil.rmtree(folder_path)
            print(f"  ✓ 已删除: {folder}")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ 删除失败: {folder}")
            print(f"    错误: {e}")
            failed_count += 1
    
    # 总结
    print(f"\n{'='*60}")
    print(f"删除完成")
    print(f"{'='*60}")
    print(f"成功删除: {deleted_count}个文件夹")
    if failed_count > 0:
        print(f"删除失败: {failed_count}个文件夹")
    print(f"\n✅ 操作完成！")


if __name__ == "__main__":
    main()
