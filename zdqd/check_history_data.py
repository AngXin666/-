"""
检查历史数据中的昵称和用户ID
"""

from src.local_db import LocalDatabase

def check_history_data():
    """检查历史数据"""
    db = LocalDatabase()
    
    # 获取所有历史记录
    records = db.get_all_history_records()
    
    print(f"共 {len(records)} 条历史记录")
    print("=" * 80)
    
    # 检查前10条
    for i, record in enumerate(records[:10], 1):
        print(f"\n记录 {i}:")
        print(f"  手机号: {record.get('手机号')}")
        print(f"  昵称: '{record.get('昵称')}'")
        print(f"  用户ID: '{record.get('用户ID')}'")
        print(f"  状态: {record.get('状态')}")
        print(f"  运行日期: {record.get('运行日期')}")
        
        # 检查是否有问题
        if record.get('昵称') == '待处理':
            print(f"  ⚠️ 昵称为'待处理'")
        if record.get('用户ID') == '待处理':
            print(f"  ⚠️ 用户ID为'待处理'")


if __name__ == '__main__':
    check_history_data()
