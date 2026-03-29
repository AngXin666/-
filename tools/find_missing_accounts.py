"""查找历史记录中有但账号文件中没有的账号"""
import sqlite3
from src.encrypted_accounts_file import EncryptedAccountsFile

# 1. 从账号文件读取所有账号
print("正在读取账号文件...")
ef = EncryptedAccountsFile('data/accounts.txt')
accounts = ef.read_accounts()
account_phones = set(phone for phone, pwd in accounts)
print(f"账号文件中的账号数量: {len(account_phones)}")

# 2. 从历史记录数据库读取所有手机号
print("\n正在读取历史记录数据库...")

# 尝试多个可能的数据库文件
db_files = ['history_records.db', 'runtime_data/license.db']
conn = None
history_phones = []

for db_file in db_files:
    try:
        print(f"尝试读取: {db_file}")
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # 列出所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"  找到表: {[t[0] for t in tables]}")
        
        # 检查 history_records 表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history_records'")
        if cursor.fetchone():
            cursor.execute("SELECT DISTINCT phone FROM history_records WHERE phone IS NOT NULL ORDER BY phone")
            history_phones = [row[0] for row in cursor.fetchall()]
            print(f"  ✓ 找到 history_records 表，包含 {len(history_phones)} 个唯一手机号")
            break
        
        conn.close()
    except Exception as e:
        print(f"  ✗ 读取失败: {e}")
        if conn:
            conn.close()

if not history_phones:
    print("\n未找到任何历史记录数据")
    print("账号文件中已有 225 个账号")
    exit()

# 获取所有唯一手机号
print(f"历史记录中的唯一手机号数量: {len(history_phones)}")

# 3. 找出差异
missing_phones = [phone for phone in history_phones if phone not in account_phones]

print(f"\n在历史记录中但账号文件中没有的手机号数量: {len(missing_phones)}")
print("\n缺失的手机号列表:")
print("=" * 60)
for i, phone in enumerate(missing_phones, 1):
    # 获取该账号的最新记录信息
    cursor.execute("""
        SELECT nickname, user_id, balance_after, run_date 
        FROM history_records 
        WHERE phone = ? 
        ORDER BY created_at DESC 
        LIMIT 1
    """, (phone,))
    record = cursor.fetchone()
    
    if record:
        nickname, user_id, balance, run_date = record
        print(f"{i:3d}. {phone} | 昵称: {nickname or '未知'} | ID: {user_id or '未知'} | 余额: {balance or 0:.2f} | 最后运行: {run_date}")
    else:
        print(f"{i:3d}. {phone}")

print("=" * 60)

# 4. 保存到文件
output_file = "missing_accounts.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("# 在历史记录中但账号文件中没有的手机号\n")
    f.write(f"# 总数: {len(missing_phones)}\n")
    f.write("# 格式: 手机号----密码\n")
    f.write("# 请手动填写密码后导入\n\n")
    for phone in missing_phones:
        f.write(f"{phone}----\n")

print(f"\n已保存到文件: {output_file}")
print("请手动填写密码后导入到账号文件")

conn.close()
