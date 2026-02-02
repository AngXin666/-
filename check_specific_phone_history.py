"""
查看特定手机号的日志历史
"""
import sys
import io

# 设置标准输出为UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_phone_logs(phone, date='20260201', time_pattern='04:17'):
    """查看特定手机号在特定时间段的日志"""
    log_file = f'logs/debug_{date}.log'
    
    print("=" * 80)
    print(f"查看手机号 {phone} 在 {date} {time_pattern} 附近的日志")
    print("=" * 80)
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # 找到包含手机号和时间的行
        target_indices = []
        for i, line in enumerate(lines):
            if phone in line and time_pattern in line:
                target_indices.append(i)
        
        if not target_indices:
            print(f"未找到包含 {phone} 和 {time_pattern} 的日志")
            return
        
        # 打印每个目标行前后20行
        for idx in target_indices:
            print(f"\n{'='*80}")
            print(f"目标行 {idx}: {lines[idx].rstrip()}")
            print(f"{'='*80}\n")
            
            start = max(0, idx - 20)
            end = min(len(lines), idx + 30)
            
            for i in range(start, end):
                marker = " >>> " if i == idx else "     "
                print(f"{marker}{lines[i].rstrip()}")
        
        print("\n" + "=" * 80)
        print(f"共找到 {len(target_indices)} 个匹配位置")
        print("=" * 80)
        
    except FileNotFoundError:
        print(f"日志文件不存在: {log_file}")
    except Exception as e:
        print(f"读取日志失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_phone_logs('19068460058', '20260201', '04:17')
