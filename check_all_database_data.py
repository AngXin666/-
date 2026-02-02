"""全面检查数据库中的所有数据"""
import sys
sys.path.insert(0, 'src')

from local_db import LocalDatabase


def check_all_data():
    """全面检查数据库数据"""
    print("=" * 60)
    print("全面检查数据库数据")
    print("=" * 60)
    
    db = LocalDatabase()
    
    # 1. 获取所有账号汇总
    print("\n[1/6] 获取所有账号汇总...")
    all_summaries = db.get_all_accounts_summary(limit=10000)
    print(f"✓ 共 {len(all_summaries)} 个账号")
    
    # 2. 检查昵称异常
    print("\n[2/6] 检查昵称异常...")
    nickname_issues = []
    for summary in all_summaries:
        phone = summary.get('phone', '')
        nickname = summary.get('nickname', '')
        
        if not nickname or nickname in ['-', 'N/A', '待更新', '']:
            nickname_issues.append({
                'phone': phone,
                'nickname': nickname,
                'user_id': summary.get('user_id', ''),
                'latest_date': summary.get('latest_date', '')
            })
    
    print(f"昵称异常的账号: {len(nickname_issues)} 个")
    if nickname_issues:
        print("\n昵称异常账号列表（前20个）:")
        for i, issue in enumerate(nickname_issues[:20], 1):
            print(f"  {i}. {issue['phone']} | 昵称='{issue['nickname']}' | 用户ID='{issue['user_id']}' | 最新日期={issue['latest_date']}")
    
    # 3. 检查用户ID异常
    print("\n[3/6] 检查用户ID异常...")
    user_id_issues = []
    for summary in all_summaries:
        phone = summary.get('phone', '')
        user_id = summary.get('user_id', '')
        
        # 检查用户ID是否为空或不是纯数字
        if not user_id or user_id in ['-', 'N/A', '待更新', '']:
            user_id_issues.append({
                'phone': phone,
                'nickname': summary.get('nickname', ''),
                'user_id': user_id,
                'latest_date': summary.get('latest_date', ''),
                'reason': '空值'
            })
        elif not user_id.isdigit():
            user_id_issues.append({
                'phone': phone,
                'nickname': summary.get('nickname', ''),
                'user_id': user_id,
                'latest_date': summary.get('latest_date', ''),
                'reason': '非数字'
            })
    
    print(f"用户ID异常的账号: {len(user_id_issues)} 个")
    if user_id_issues:
        print("\n用户ID异常账号列表（前20个）:")
        for i, issue in enumerate(user_id_issues[:20], 1):
            print(f"  {i}. {issue['phone']} | 昵称='{issue['nickname']}' | 用户ID='{issue['user_id']}' | 原因={issue['reason']} | 最新日期={issue['latest_date']}")
    
    # 4. 检查管理员字段
    print("\n[4/6] 检查管理员字段...")
    owner_stats = {
        'has_owner': 0,
        'no_owner': 0,
        'owner_list': {}
    }
    
    for summary in all_summaries:
        phone = summary.get('phone', '')
        # 获取最新记录的管理员
        records = db.get_history_records(phone, limit=1)
        if records:
            owner = records[0].get('owner', '')
            if owner and owner not in ['-', 'N/A', '', None]:
                owner_stats['has_owner'] += 1
                owner_stats['owner_list'][owner] = owner_stats['owner_list'].get(owner, 0) + 1
            else:
                owner_stats['no_owner'] += 1
        else:
            owner_stats['no_owner'] += 1
    
    print(f"有管理员的账号: {owner_stats['has_owner']} 个")
    print(f"无管理员的账号: {owner_stats['no_owner']} 个")
    
    if owner_stats['owner_list']:
        print("\n管理员分布:")
        for owner, count in sorted(owner_stats['owner_list'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {owner}: {count} 个账号")
    
    # 5. 检查最新状态
    print("\n[5/6] 检查最新状态...")
    status_stats = {}
    for summary in all_summaries:
        phone = summary.get('phone', '')
        records = db.get_history_records(phone, limit=1)
        if records:
            status = records[0].get('状态', '未知')
            status_stats[status] = status_stats.get(status, 0) + 1
    
    print("状态分布:")
    for status, count in sorted(status_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {status}: {count} 个账号")
    
    # 6. 检查数据完整性
    print("\n[6/6] 检查数据完整性...")
    incomplete_data = []
    for summary in all_summaries:
        phone = summary.get('phone', '')
        nickname = summary.get('nickname', '')
        user_id = summary.get('user_id', '')
        
        issues = []
        if not nickname or nickname in ['-', 'N/A', '待更新', '']:
            issues.append('昵称异常')
        if not user_id or user_id in ['-', 'N/A', '待更新', ''] or not user_id.isdigit():
            issues.append('用户ID异常')
        
        if issues:
            incomplete_data.append({
                'phone': phone,
                'nickname': nickname,
                'user_id': user_id,
                'issues': ', '.join(issues)
            })
    
    print(f"数据不完整的账号: {len(incomplete_data)} 个")
    if incomplete_data:
        print("\n数据不完整账号列表（前20个）:")
        for i, data in enumerate(incomplete_data[:20], 1):
            print(f"  {i}. {data['phone']} | 昵称='{data['nickname']}' | 用户ID='{data['user_id']}' | 问题={data['issues']}")
    
    # 总结
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    print(f"总账号数: {len(all_summaries)}")
    print(f"昵称异常: {len(nickname_issues)} 个")
    print(f"用户ID异常: {len(user_id_issues)} 个")
    print(f"数据不完整: {len(incomplete_data)} 个")
    print(f"有管理员: {owner_stats['has_owner']} 个")
    print(f"无管理员: {owner_stats['no_owner']} 个")
    
    if len(nickname_issues) == 0 and len(user_id_issues) == 0:
        print("\n✅ 数据库数据完全正常！")
    elif len(incomplete_data) <= 2:
        print(f"\n⚠️ 只有 {len(incomplete_data)} 个账号数据异常，基本正常")
    else:
        print(f"\n❌ 有 {len(incomplete_data)} 个账号数据异常，需要修复")


if __name__ == '__main__':
    check_all_data()
