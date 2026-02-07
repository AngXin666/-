"""
移动已完成模型的训练数据到归档文件夹
"""
import shutil
from pathlib import Path

def move_completed_data():
    """移动已完成模型的数据"""
    
    # 已完成训练的数据文件夹
    completed_folders = [
        "首页",  # 首页检测模型已完成
        "登录页",  # 登录页检测模型已完成
        "签到页",  # 签到页检测模型已完成
        "签到弹窗",  # 签到弹窗检测模型已完成
        "温馨提示",  # 温馨提示检测模型已完成
        "用户名或密码错误弹窗",  # 登录异常模型已完成（合并数据集之一）
        "手机号码不存在",  # 登录异常模型已完成（合并数据集之一）
        "个人页_已登录_余额积分",  # 余额积分检测模型已完成
    ]
    
    # 创建归档目录
    archive_dir = Path("training_data_completed")
    archive_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("移动已完成模型的训练数据")
    print("=" * 80)
    
    moved_count = 0
    for folder_name in completed_folders:
        source = Path("training_data") / folder_name
        
        if source.exists():
            dest = archive_dir / folder_name
            
            # 如果目标已存在，先删除
            if dest.exists():
                shutil.rmtree(dest)
            
            # 移动文件夹
            shutil.move(str(source), str(dest))
            print(f"✓ 已移动: {folder_name}")
            moved_count += 1
        else:
            print(f"⚠ 不存在: {folder_name}")
    
    print("\n" + "=" * 80)
    print(f"移动完成！共移动 {moved_count} 个文件夹")
    print(f"归档位置: {archive_dir}")
    print("=" * 80)
    
    # 显示剩余的训练数据
    print("\n剩余的训练数据文件夹:")
    remaining = list(Path("training_data").glob("*"))
    remaining_dirs = [d for d in remaining if d.is_dir()]
    
    if remaining_dirs:
        for d in sorted(remaining_dirs):
            print(f"  - {d.name}")
    else:
        print("  (无)")

if __name__ == "__main__":
    move_completed_data()
