import sqlite3

conn = sqlite3.connect('history_records.db')
cursor = conn.cursor()

# 查看所有表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('数据库中的表:')
for t in tables:
    print(f'  - {t[0]}')
    
    # 查看每个表的记录数
    cursor.execute(f'SELECT COUNT(*) FROM {t[0]}')
    count = cursor.fetchone()[0]
    print(f'    记录数: {count}')

conn.close()
