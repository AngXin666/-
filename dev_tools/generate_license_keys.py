"""
卡密生成工具
License Key Generator
"""

import random
import string
import json
from datetime import datetime
from pathlib import Path


def generate_license_key():
    """生成单个卡密
    
    格式：KIRO-XXXX-XXXX-XXXX-XXXX
    """
    # 前缀固定为 KIRO
    prefix = "KIRO"
    
    # 生成4组4位随机字符（大写字母+数字）
    groups = []
    for _ in range(4):
        group = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        groups.append(group)
    
    # 组合成完整卡密
    license_key = f"{prefix}-{'-'.join(groups)}"
    return license_key


def generate_batch(count=10, save_to_file=True):
    """批量生成卡密
    
    Args:
        count: 生成数量
        save_to_file: 是否保存到文件
    """
    print("=" * 60)
    print(f"卡密生成工具 - 生成 {count} 个卡密")
    print("=" * 60)
    print()
    
    keys = []
    for i in range(count):
        key = generate_license_key()
        keys.append(key)
        print(f"{i+1:3d}. {key}")
    
    if save_to_file:
        # 保存到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"license_keys_{timestamp}.txt"
        
        output_dir = Path("dev_tools") / "generated_keys"
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"卡密生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"生成数量：{count}\n")
            f.write("=" * 60 + "\n\n")
            
            for i, key in enumerate(keys, 1):
                f.write(f"{i:3d}. {key}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("注意：请妥善保管卡密，不要泄露给他人\n")
        
        print()
        print(f"✅ 卡密已保存到：{filepath}")
        
        # 同时保存JSON格式（便于程序读取）
        json_file = output_dir / f"license_keys_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'count': count,
                'keys': keys
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON格式已保存到：{json_file}")
    
    print()
    print("=" * 60)


def main():
    """主函数"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 18 + "卡密生成工具" + " " * 28 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    while True:
        try:
            count_input = input("请输入要生成的卡密数量（默认10，输入0退出）：").strip()
            
            if not count_input:
                count = 10
            else:
                count = int(count_input)
            
            if count == 0:
                print("退出程序")
                break
            
            if count < 1 or count > 1000:
                print("❌ 数量必须在 1-1000 之间")
                continue
            
            print()
            generate_batch(count)
            print()
            
            # 询问是否继续
            continue_input = input("是否继续生成？(y/n)：").strip().lower()
            if continue_input != 'y':
                break
            
            print()
            
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n程序被中断")
            break
        except Exception as e:
            print(f"❌ 错误：{e}")


if __name__ == "__main__":
    main()
