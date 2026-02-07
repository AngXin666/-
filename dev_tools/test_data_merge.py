#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试数据合并逻辑：验证空数据不会覆盖有效数据"""

from src.local_db import LocalDatabase
from datetime import datetime

db = LocalDatabase()

test_phone = "13800138888"
test_date = datetime.now().strftime('%Y-%m-%d')

print("=" * 60)
print("测试场景：同一账号多次运行，验证数据合并逻辑")
print("=" * 60)

# 第一次运行：完整数据（签到成功+转账成功）
print("\n【第1次运行】保存完整数据...")
record1 = {
    'phone': test_phone,
    'run_date': test_date,
    'nickname': '测试用户',
    'user_id': 'test_user_123',
    'balance_before': 100.50,
    'balance_after': 130.75,
    'points': 500,
    'vouchers': 10,
    'coupons': 5,
    'checkin_reward': 30.25,
    'checkin_total_times': 100,
    'checkin_balance_after': 130.75,
    'transfer_amount': 0,  # 第一次没转账
    'transfer_recipient': '',
    'status': '成功',
    'duration': 45.5,
    'timestamp': datetime.now().isoformat(),
    'owner': 'admin'
}

success1 = db.upsert_history_record(record1)
print(f"结果: {'✓ 成功' if success1 else '❌ 失败'}")

# 查询第一次的数据
conn = db._get_connection()
cursor = conn.cursor()
cursor.execute("""
    SELECT nickname, balance_before, balance_after, checkin_reward, transfer_amount, status
    FROM history_records WHERE phone = ? AND run_date = ?
""", (test_phone, test_date))
data1 = cursor.fetchone()
conn.close()

print(f"保存后的数据: 昵称={data1[0]}, 签到前余额={data1[1]}, 签到后余额={data1[2]}, "
      f"签到奖励={data1[3]}, 转账金额={data1[4]}, 状态={data1[5]}")

# 第二次运行：部分数据为空/0（模拟第二次运行时签到已完成，但有转账）
print("\n【第2次运行】保存部分数据（签到已完成，新增转账）...")
record2 = {
    'phone': test_phone,
    'run_date': test_date,
    'nickname': '',  # 空值，不应覆盖
    'user_id': 'test_user_123',
    'balance_before': 0,  # 0值，不应覆盖
    'balance_after': 0,  # 0值，不应覆盖
    'points': 0,  # 0值，不应覆盖
    'vouchers': 0,  # 0值，不应覆盖
    'coupons': 0,  # 0值，不应覆盖
    'checkin_reward': 0,  # 0值，不应覆盖
    'checkin_total_times': 0,  # 0值，不应覆盖
    'checkin_balance_after': 0,  # 0值，不应覆盖
    'transfer_amount': 100.50,  # 新增：转账金额
    'transfer_recipient': '收款人ID',  # 新增：收款人
    'status': '成功',
    'duration': 15.2,  # 更新：新的执行时间
    'timestamp': datetime.now().isoformat(),
    'owner': 'admin'
}

success2 = db.upsert_history_record(record2)
print(f"结果: {'✓ 成功' if success2 else '❌ 失败'}")

# 查询第二次的数据
conn = db._get_connection()
cursor = conn.cursor()
cursor.execute("""
    SELECT nickname, balance_before, balance_after, checkin_reward, transfer_amount, transfer_recipient, status, duration
    FROM history_records WHERE phone = ? AND run_date = ?
""", (test_phone, test_date))
data2 = cursor.fetchone()
conn.close()

print(f"保存后的数据: 昵称={data2[0]}, 签到前余额={data2[1]}, 签到后余额={data2[2]}, "
      f"签到奖励={data2[3]}, 转账金额={data2[4]}, 收款人={data2[5]}, 状态={data2[6]}, 执行时间={data2[7]}")

# 验证结果
print("\n" + "=" * 60)
print("验证结果：")
print("=" * 60)

errors = []

if data2[0] != '测试用户':
    errors.append(f"❌ 昵称被覆盖: 期望='测试用户', 实际='{data2[0]}'")
else:
    print("✓ 昵称保留: 测试用户")

if data2[1] != 100.50:
    errors.append(f"❌ 签到前余额被覆盖: 期望=100.50, 实际={data2[1]}")
else:
    print("✓ 签到前余额保留: 100.50")

if data2[2] != 130.75:
    errors.append(f"❌ 签到后余额被覆盖: 期望=130.75, 实际={data2[2]}")
else:
    print("✓ 签到后余额保留: 130.75")

if data2[3] != 30.25:
    errors.append(f"❌ 签到奖励被覆盖: 期望=30.25, 实际={data2[3]}")
else:
    print("✓ 签到奖励保留: 30.25")

if data2[4] != 100.50:
    errors.append(f"❌ 转账金额未更新: 期望=100.50, 实际={data2[4]}")
else:
    print("✓ 转账金额已更新: 100.50")

if data2[5] != '收款人ID':
    errors.append(f"❌ 收款人未更新: 期望='收款人ID', 实际='{data2[5]}'")
else:
    print("✓ 收款人已更新: 收款人ID")

if data2[7] != 15.2:
    errors.append(f"❌ 执行时间未更新: 期望=15.2, 实际={data2[7]}")
else:
    print("✓ 执行时间已更新: 15.2")

# 清理测试数据
conn = db._get_connection()
cursor = conn.cursor()
cursor.execute("DELETE FROM history_records WHERE phone = ?", (test_phone,))
conn.commit()
conn.close()
print(f"\n✓ 测试数据已清理")

# 最终结果
print("\n" + "=" * 60)
if errors:
    print("❌ 测试失败！")
    for error in errors:
        print(error)
else:
    print("✅ 测试通过！数据合并逻辑正确！")
    print("  - 第一次的有效数据被保留")
    print("  - 第二次的新增数据被累加")
    print("  - 空值和0值不会覆盖有效数据")
print("=" * 60)
