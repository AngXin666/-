"""
检查签到次数OCR识别问题
分析为什么很多签到次数没有识别出来
"""

import sys
import os
import re
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.local_db import LocalDatabase


def analyze_checkin_times():
    """分析签到次数识别情况"""
    print("\n" + "="*60)
    print("签到次数识别情况分析")
    print("="*60)
    
    # 连接数据库
    db = LocalDatabase()
    
    # 获取所有签到记录
    records = db.get_all_history_records()
    
    if not records:
        print("\n没有找到签到记录")
        return
    
    # 只取最近200条
    records = records[:200]
    
    # 统计
    total_count = len(records)
    has_times_count = 0
    no_times_count = 0
    
    # 按手机号分组
    phone_stats = {}
    
    for record in records:
        phone = record.get('phone', '')
        nickname = record.get('nickname', '')
        user_id = record.get('user_id', '')
        total_times = record.get('checkin_total_times')
        timestamp = record.get('checkin_time', '')  # 修正字段名
        
        if phone not in phone_stats:
            phone_stats[phone] = {
                'nickname': nickname,
                'user_id': user_id,
                'total_records': 0,
                'has_times': 0,
                'no_times': 0,
                'latest_times': None,
                'latest_timestamp': None
            }
        
        phone_stats[phone]['total_records'] += 1
        
        if total_times is not None and total_times != 'N/A':
            has_times_count += 1
            phone_stats[phone]['has_times'] += 1
            if phone_stats[phone]['latest_times'] is None:
                phone_stats[phone]['latest_times'] = total_times
                phone_stats[phone]['latest_timestamp'] = timestamp
        else:
            no_times_count += 1
            phone_stats[phone]['no_times'] += 1
    
    # 输出统计结果
    print(f"\n总记录数: {total_count}")
    print(f"有签到次数: {has_times_count} ({has_times_count/total_count*100:.1f}%)")
    print(f"无签到次数: {no_times_count} ({no_times_count/total_count*100:.1f}%)")
    
    # 输出每个账号的统计
    print("\n" + "="*60)
    print("各账号识别情况")
    print("="*60)
    print(f"{'手机号':<15} {'昵称':<12} {'总记录':<8} {'有次数':<8} {'无次数':<8} {'识别率':<10} {'最新次数':<10}")
    print("-"*60)
    
    # 按识别率排序
    sorted_phones = sorted(phone_stats.items(), 
                          key=lambda x: x[1]['has_times']/x[1]['total_records'] if x[1]['total_records'] > 0 else 0)
    
    for phone, stats in sorted_phones:
        nickname = stats['nickname'] or 'N/A'
        total = stats['total_records']
        has = stats['has_times']
        no = stats['no_times']
        rate = has/total*100 if total > 0 else 0
        latest = stats['latest_times'] if stats['latest_times'] else 'N/A'
        
        print(f"{phone:<15} {nickname:<12} {total:<8} {has:<8} {no:<8} {rate:<9.1f}% {latest:<10}")
    
    # 分析问题
    print("\n" + "="*60)
    print("问题分析")
    print("="*60)
    
    # 识别率低于50%的账号
    low_rate_phones = [phone for phone, stats in phone_stats.items() 
                       if stats['has_times']/stats['total_records'] < 0.5 and stats['total_records'] >= 3]
    
    if low_rate_phones:
        print(f"\n识别率低于50%的账号（{len(low_rate_phones)}个）：")
        for phone in low_rate_phones:
            stats = phone_stats[phone]
            rate = stats['has_times']/stats['total_records']*100
            print(f"  - {phone} ({stats['nickname']}): {rate:.1f}% ({stats['has_times']}/{stats['total_records']})")
    
    # 完全没有识别的账号
    zero_rate_phones = [phone for phone, stats in phone_stats.items() 
                        if stats['has_times'] == 0 and stats['total_records'] >= 2]
    
    if zero_rate_phones:
        print(f"\n完全没有识别的账号（{len(zero_rate_phones)}个）：")
        for phone in zero_rate_phones:
            stats = phone_stats[phone]
            print(f"  - {phone} ({stats['nickname']}): 0% (0/{stats['total_records']})")
    
    # 建议
    print("\n" + "="*60)
    print("优化建议")
    print("="*60)
    print("\n1. OCR识别问题：")
    print("   - 签到页面的文字可能被遮挡或模糊")
    print("   - OCR引擎识别精度不够")
    print("   - 文字格式变化导致正则匹配失败")
    
    print("\n2. 解决方案：")
    print("   ✓ 增强OCR预处理（灰度化、对比度增强）")
    print("   ✓ 增加更多正则匹配模式")
    print("   ✓ 使用YOLO检测签到次数区域后再OCR")
    print("   ✓ 添加降级方案（如果OCR失败，使用历史数据推算）")
    
    print("\n3. 当前代码已有的优化：")
    print("   ✓ 使用OCR图像预处理增强")
    print("   ✓ 支持多种文本格式匹配")
    print("   ✓ 跨文本匹配（处理分开识别的情况）")
    
    print("\n4. 需要进一步优化：")
    print("   ⚠️ 考虑使用YOLO检测签到次数区域")
    print("   ⚠️ 添加更多OCR识别模式")
    print("   ⚠️ 增加调试日志，记录OCR原始文本")


def check_ocr_patterns():
    """检查OCR识别模式"""
    print("\n" + "="*60)
    print("OCR识别模式检查")
    print("="*60)
    
    # 测试各种可能的文本格式
    test_texts = [
        "您总次数为108,您当日还有1次签到任务",
        "您总次数为107，您当日还有0次签到任务",
        "总次数: 108",
        "总次数：108",
        "总次数为108",
        "总次数108",
        "总 次数108",
        "总次 数108",
        "当日还有1次",
        "当日剩余: 1",
        "当日还有 1 次",
        "总次数 为 108 当日还有 1 次",
    ]
    
    print("\n测试正则匹配：")
    for text in test_texts:
        print(f"\n文本: '{text}'")
        
        # 测试总次数匹配
        match1 = re.search(r'总次数为(\d+)[,，].*?当日还有(\d+)次', text)
        if match1:
            print(f"  ✓ 格式1匹配: 总次数={match1.group(1)}, 剩余={match1.group(2)}")
        
        match2 = re.search(r'总次数[:：为]\s*(\d+)', text)
        if match2:
            print(f"  ✓ 格式2匹配: 总次数={match2.group(1)}")
        
        match3 = re.search(r'总\s*次\s*数\s*[:：为]?\s*(\d+)', text)
        if match3:
            print(f"  ✓ 格式3匹配: 总次数={match3.group(1)}")
        
        match4 = re.search(r'当日(?:还有|剩余)[:：]?\s*(\d+)次?', text)
        if match4:
            print(f"  ✓ 格式4匹配: 剩余={match4.group(1)}")
        
        if not (match1 or match2 or match3 or match4):
            print(f"  ✗ 没有匹配")


def main():
    """主函数"""
    try:
        # 分析签到次数识别情况
        analyze_checkin_times()
        
        # 检查OCR识别模式
        check_ocr_patterns()
        
        print("\n" + "="*60)
        print("✅ 分析完成")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
