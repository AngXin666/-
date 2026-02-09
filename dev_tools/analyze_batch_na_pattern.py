"""
分析批量出现N/A的模式
找出什么情况下会批量识别失败
"""

import sys
import os
from datetime import datetime
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.local_db import LocalDatabase


def analyze_batch_na_pattern():
    """分析批量N/A的模式"""
    print("\n" + "="*60)
    print("分析批量N/A模式")
    print("="*60)
    
    db = LocalDatabase()
    records = db.get_all_history_records()
    
    if not records:
        print("\n没有找到记录")
        return
    
    # 按时间分组（精确到分钟）
    time_groups = defaultdict(list)
    
    for record in records:
        checkin_time = record.get('checkin_time', '')
        total_times = record.get('checkin_total_times')
        phone = record.get('phone', '')
        
        if checkin_time:
            # 提取时间（精确到分钟）
            try:
                # 假设格式是 "2024-02-08 10:30:45"
                time_key = checkin_time[:16]  # "2024-02-08 10:30"
            except:
                time_key = checkin_time
            
            time_groups[time_key].append({
                'phone': phone,
                'total_times': total_times,
                'has_times': total_times is not None
            })
    
    # 找出批量N/A的时间段
    print("\n批量N/A时间段分析：")
    print(f"{'时间':<20} {'总数':<8} {'有次数':<8} {'无次数':<8} {'失败率':<10}")
    print("-"*60)
    
    batch_na_times = []
    
    for time_key in sorted(time_groups.keys(), reverse=True)[:50]:  # 只看最近50个时间段
        group = time_groups[time_key]
        total = len(group)
        has_times = sum(1 for r in group if r['has_times'])
        no_times = total - has_times
        fail_rate = no_times / total * 100 if total > 0 else 0
        
        # 如果失败率超过30%，认为是批量失败
        if fail_rate >= 30 and total >= 3:
            batch_na_times.append({
                'time': time_key,
                'total': total,
                'has_times': has_times,
                'no_times': no_times,
                'fail_rate': fail_rate,
                'phones': [r['phone'] for r in group if not r['has_times']]
            })
            print(f"{time_key:<20} {total:<8} {has_times:<8} {no_times:<8} {fail_rate:<9.1f}%")
    
    if not batch_na_times:
        print("\n✅ 未发现批量N/A的时间段（失败率<30%）")
    else:
        print(f"\n⚠️ 发现{len(batch_na_times)}个批量N/A时间段")
        
        # 详细分析每个批量失败时间段
        print("\n" + "="*60)
        print("批量失败详情")
        print("="*60)
        
        for i, batch in enumerate(batch_na_times[:5], 1):  # 只显示前5个
            print(f"\n{i}. 时间: {batch['time']}")
            print(f"   失败率: {batch['fail_rate']:.1f}% ({batch['no_times']}/{batch['total']})")
            print(f"   失败账号:")
            for phone in batch['phones']:
                print(f"     - {phone}")
    
    # 分析失败模式
    print("\n" + "="*60)
    print("失败模式分析")
    print("="*60)
    
    # 统计连续失败的账号
    consecutive_failures = defaultdict(int)
    
    for record in records[:100]:  # 只看最近100条
        phone = record.get('phone', '')
        total_times = record.get('checkin_total_times')
        
        if total_times is None:
            consecutive_failures[phone] += 1
    
    if consecutive_failures:
        print("\n连续失败的账号（最近100条记录）：")
        for phone, count in sorted(consecutive_failures.items(), key=lambda x: x[1], reverse=True):
            if count >= 2:
                print(f"  {phone}: {count}次连续失败")
    
    return batch_na_times


def analyze_ocr_failure_causes():
    """分析OCR识别失败的可能原因"""
    print("\n" + "="*60)
    print("OCR识别失败原因分析")
    print("="*60)
    
    print("\n可能导致批量失败的原因：")
    print("\n1. 【页面加载问题】")
    print("   - 签到页面加载不完整")
    print("   - 网络延迟导致文字未显示")
    print("   - 页面被其他元素遮挡")
    
    print("\n2. 【OCR引擎问题】")
    print("   - OCR线程池繁忙/超时")
    print("   - OCR引擎崩溃/重启")
    print("   - 内存不足导致OCR失败")
    
    print("\n3. 【页面状态问题】")
    print("   - 未正确进入签到页面")
    print("   - 进入了错误的页面（如广告页）")
    print("   - 页面检测误判")
    
    print("\n4. 【并发问题】")
    print("   - 多个账号同时签到")
    print("   - 资源竞争导致部分失败")
    print("   - 设备性能不足")
    
    print("\n5. 【应用问题】")
    print("   - 应用崩溃/重启")
    print("   - 应用更新导致页面变化")
    print("   - 签到页面UI改版")


def suggest_solutions():
    """建议解决方案"""
    print("\n" + "="*60)
    print("解决方案建议")
    print("="*60)
    
    print("\n【立即可实施】")
    print("1. 增加OCR超时重试")
    print("   - 当前超时: 2秒")
    print("   - 建议: 失败后重试1-2次")
    
    print("\n2. 增加页面加载等待")
    print("   - 进入签到页后等待更长时间")
    print("   - 确保页面完全加载")
    
    print("\n3. 添加降级方案")
    print("   - 如果OCR失败，使用历史数据推算")
    print("   - 记录失败原因，便于调试")
    
    print("\n【需要开发】")
    print("4. 使用YOLO检测签到次数区域")
    print("   - 先用YOLO定位文字区域")
    print("   - 再对区域进行OCR识别")
    print("   - 提高识别准确率")
    
    print("\n5. 添加OCR结果验证")
    print("   - 验证识别结果的合理性")
    print("   - 如果不合理，重新识别")
    
    print("\n6. 保存失败截图")
    print("   - 识别失败时保存截图")
    print("   - 便于分析失败原因")
    print("   - 用于训练YOLO模型")


def main():
    """主函数"""
    try:
        # 分析批量N/A模式
        batch_na_times = analyze_batch_na_pattern()
        
        # 分析失败原因
        analyze_ocr_failure_causes()
        
        # 建议解决方案
        suggest_solutions()
        
        print("\n" + "="*60)
        print("✅ 分析完成")
        print("="*60)
        
        if batch_na_times:
            print(f"\n⚠️ 发现{len(batch_na_times)}个批量失败时间段")
            print("建议：")
            print("1. 检查这些时间段的日志")
            print("2. 查看是否有系统异常")
            print("3. 实施上述解决方案")
        else:
            print("\n✅ 未发现明显的批量失败模式")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
