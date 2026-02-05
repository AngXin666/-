#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试转账日志分离功能"""

from src.transfer_history import get_transfer_history

# 获取转账历史管理器
history = get_transfer_history()

print("测试转账日志分离功能...\n")

# 测试成功的转账记录
print("1. 保存成功的转账记录...")
history.save_transfer_record(
    sender_phone="13800138000",
    sender_user_id="test_user_1",
    sender_name="测试用户A",
    recipient_phone="13900139000",
    recipient_name="测试用户B",
    amount=100.50,
    strategy="单级转账",
    success=True,
    error_message="",
    owner=""
)
print("   ✓ 成功记录已保存\n")

# 测试失败的转账记录
print("2. 保存失败的转账记录...")
history.save_transfer_record(
    sender_phone="13800138001",
    sender_user_id="test_user_2",
    sender_name="测试用户C",
    recipient_phone="13900139001",
    recipient_name="测试用户D",
    amount=50.25,
    strategy="单级转账",
    success=False,
    error_message="未能进入转账页面",
    owner=""
)
print("   ✓ 失败记录已保存\n")

print("=" * 60)
print("测试完成！请检查以下日志文件：")
print("  1. logs/transfer_YYYYMMDD.log - 所有转账记录（成功+失败）")
print("  2. logs/transfer_failure_YYYYMMDD.log - 只有失败记录")
print("=" * 60)
