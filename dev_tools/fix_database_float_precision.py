"""
修复数据库中的浮点数精度问题

功能：
1. 扫描数据库中所有历史记录
2. 识别有精度问题的记录（如 18.86999999）
3. 修复精度问题（统一保留2位小数）
4. 生成修复报告
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import sqlite3

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def check_float_precision(value, field_name):
    """检查浮点数是否有精度问题
    
    Args:
        value: 浮点数值
        field_name: 字段名
        
    Returns:
        (has_issue, corrected_value) - 是否有问题，修正后的值
    """
    if value is None:
        return False, None
    
    if not isinstance(value, (int, float)):
        return False, value
    
    # 转换为浮点数
    float_value = float(value)
    
    # 保留2位小数
    corrected_value = round(float_value, 2)
    
    # 检查是否有精度问题（差异超过0.0001）
    has_issue = abs(float_value - corrected_value) > 0.0001
    
    return has_issue, corrected_value


def scan_database():
    """扫描数据库，识别有精度问题的记录"""
    print("="*60)
    print("扫描数据库中的浮点数精度问题")
    print("="*60)
    
    db_path = Path("runtime_data") / "license.db"
    if not db_path.exists():
        print("❌ 数据库文件不存在")
        return []
    
    # 需要检查的浮点数字段
    float_fields = [
        'balance_before', 'balance_after', 'vouchers',
        'checkin_reward', 'checkin_balance_after',
        'transfer_amount', 'duration'
    ]
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 查询所有记录
    cursor.execute("""
        SELECT id, phone, run_date, balance_before, balance_after, vouchers,
               checkin_reward, checkin_balance_after, transfer_amount, duration
        FROM history_records
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    print(f"总记录数: {len(rows)}")
    
    # 分析每条记录
    issues = []
    for row in rows:
        record_id = row[0]
        phone = row[1]
        run_date = row[2]
        
        record_issues = []
        
        # 检查每个浮点数字段
        for i, field_name in enumerate(float_fields):
            value = row[3 + i]  # 从第4列开始是浮点数字段
            has_issue, corrected_value = check_float_precision(value, field_name)
            
            if has_issue:
                record_issues.append({
                    'field': field_name,
                    'original': value,
                    'corrected': corrected_value
                })
        
        if record_issues:
            issues.append({
                'id': record_id,
                'phone': phone,
                'run_date': run_date,
                'issues': record_issues
            })
    
    print(f"有精度问题的记录数: {len(issues)}")
    
    # 显示前10条问题记录
    if issues:
        print("\n前10条问题记录示例:")
        for i, issue in enumerate(issues[:10]):
            print(f"\n{i+1}. 记录ID: {issue['id']}, 手机号: {issue['phone']}, 日期: {issue['run_date']}")
            for field_issue in issue['issues']:
                print(f"   - {field_issue['field']}: {field_issue['original']} → {field_issue['corrected']}")
    
    return issues


def fix_database(issues, dry_run=True):
    """修复数据库中的精度问题
    
    Args:
        issues: 问题记录列表
        dry_run: 是否只是模拟运行（不实际修改数据库）
    """
    if not issues:
        print("\n✅ 没有需要修复的记录")
        return
    
    print("\n" + "="*60)
    if dry_run:
        print("模拟修复（不会实际修改数据库）")
    else:
        print("开始修复数据库")
    print("="*60)
    
    db_path = Path("runtime_data") / "license.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    fixed_count = 0
    
    for issue in issues:
        record_id = issue['id']
        
        # 构建更新语句
        update_fields = []
        values = []
        
        for field_issue in issue['issues']:
            update_fields.append(f"{field_issue['field']} = ?")
            values.append(field_issue['corrected'])
        
        if update_fields:
            values.append(record_id)
            sql = f"UPDATE history_records SET {', '.join(update_fields)} WHERE id = ?"
            
            if not dry_run:
                cursor.execute(sql, values)
            
            fixed_count += 1
    
    if not dry_run:
        conn.commit()
        print(f"✅ 已修复 {fixed_count} 条记录")
    else:
        print(f"模拟修复 {fixed_count} 条记录")
    
    conn.close()


def generate_report(issues):
    """生成修复报告"""
    if not issues:
        return
    
    print("\n" + "="*60)
    print("精度问题统计")
    print("="*60)
    
    # 统计每个字段的问题数量
    field_stats = {}
    for issue in issues:
        for field_issue in issue['issues']:
            field_name = field_issue['field']
            if field_name not in field_stats:
                field_stats[field_name] = 0
            field_stats[field_name] += 1
    
    print("\n各字段问题数量:")
    for field_name, count in sorted(field_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {field_name}: {count} 条")
    
    # 保存详细报告到文件
    report_path = Path("dev_tools") / f"float_precision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("数据库浮点数精度问题报告\n")
        f.write("="*60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"问题记录总数: {len(issues)}\n\n")
        
        f.write("各字段问题统计:\n")
        for field_name, count in sorted(field_stats.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {field_name}: {count} 条\n")
        
        f.write("\n详细问题列表:\n")
        f.write("="*60 + "\n")
        for i, issue in enumerate(issues):
            f.write(f"\n{i+1}. 记录ID: {issue['id']}, 手机号: {issue['phone']}, 日期: {issue['run_date']}\n")
            for field_issue in issue['issues']:
                f.write(f"   - {field_issue['field']}: {field_issue['original']} → {field_issue['corrected']}\n")
    
    print(f"\n详细报告已保存到: {report_path}")


def main():
    """主函数"""
    print("="*60)
    print("数据库浮点数精度修复工具")
    print("="*60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 步骤1: 扫描数据库
    issues = scan_database()
    
    if not issues:
        print("\n✅ 数据库中没有精度问题")
        return 0
    
    # 步骤2: 生成报告
    generate_report(issues)
    
    # 步骤3: 询问是否修复
    print("\n" + "="*60)
    print("是否要修复这些问题？")
    print("  1. 是 - 修复数据库")
    print("  2. 否 - 只生成报告")
    print("="*60)
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == '1':
        # 先模拟运行
        print("\n正在模拟修复...")
        fix_database(issues, dry_run=True)
        
        # 再次确认
        confirm = input("\n确认要修复数据库吗？(yes/no): ").strip().lower()
        if confirm == 'yes':
            fix_database(issues, dry_run=False)
            print("\n✅ 修复完成！")
        else:
            print("\n已取消修复")
    else:
        print("\n已取消修复，只生成了报告")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
