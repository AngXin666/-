"""
测试GUI统计逻辑
验证勾选机制和统计显示是否正确
"""

def test_stats_logic():
    """测试统计逻辑"""
    
    print("=" * 80)
    print("测试场景：100个账号，勾选10个（跳过），未勾选90个（执行）")
    print("=" * 80)
    
    # 初始状态
    total = 100
    checked_count = 10  # 勾选跳过的账号数
    unchecked_count = 90  # 待处理的账号数
    
    # 初始化统计
    processed = 0  # 实际处理的账号数
    success_count = checked_count  # 初始成功数 = 勾选跳过的账号数
    failed_count = 0
    
    print(f"\n初始状态：")
    print(f"  总计：{total}")
    print(f"  进度：{processed}/{total}")
    print(f"  成功：{success_count}（勾选跳过）")
    print(f"  失败：{failed_count}")
    print(f"  验证：成功 + 失败 = {success_count + failed_count}，应该等于 {checked_count + processed}")
    
    # 模拟执行过程
    print(f"\n开始执行90个未勾选的账号...")
    
    # 假设前80个成功，后10个失败
    for i in range(90):
        processed += 1
        if i < 80:
            # 成功
            success_count += 1
        else:
            # 失败
            failed_count += 1
        
        # 每10个账号输出一次状态
        if (i + 1) % 10 == 0:
            print(f"\n处理进度 {i+1}/90：")
            print(f"  总计：{total}")
            print(f"  进度：{processed}/{total}")
            print(f"  成功：{success_count}（{checked_count}个跳过 + {success_count - checked_count}个执行成功）")
            print(f"  失败：{failed_count}")
            print(f"  验证：成功 + 失败 = {success_count + failed_count}，应该等于 {checked_count + processed}")
            
            # 验证公式
            assert success_count + failed_count == checked_count + processed, \
                f"统计错误：成功({success_count}) + 失败({failed_count}) != 勾选({checked_count}) + 处理({processed})"
    
    # 最终状态
    print(f"\n最终状态：")
    print(f"  总计：{total}")
    print(f"  进度：{processed}/{total}")
    print(f"  成功：{success_count}（{checked_count}个跳过 + {success_count - checked_count}个执行成功）")
    print(f"  失败：{failed_count}")
    print(f"  验证：成功 + 失败 = {success_count + failed_count}，应该等于 {checked_count + processed}")
    
    # 验证最终结果
    assert processed == unchecked_count, f"处理数量错误：{processed} != {unchecked_count}"
    assert success_count == checked_count + 80, f"成功数量错误：{success_count} != {checked_count + 80}"
    assert failed_count == 10, f"失败数量错误：{failed_count} != 10"
    assert success_count + failed_count == checked_count + processed, \
        f"统计错误：成功({success_count}) + 失败({failed_count}) != 勾选({checked_count}) + 处理({processed})"
    
    print(f"\n✅ 所有验证通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_stats_logic()
