"""
测试API连接
"""

import json
from pathlib import Path

def test_api():
    """测试API连接"""
    print("=" * 60)
    print("测试API连接")
    print("=" * 60)
    
    # 读取配置
    config_file = Path("runtime_data") / "api_config.json"
    
    if not config_file.exists():
        print("\n[ERROR] 配置文件不存在")
        print(f"  请创建: {config_file}")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    api_url = config.get('api_url', '')
    
    if not api_url:
        print("\n[ERROR] API地址未配置")
        return
    
    print(f"\nAPI地址: {api_url}")
    print("\n正在测试连接...")
    
    try:
        import requests
        
        # 测试根路径
        response = requests.get(f"{api_url}/api/activate", timeout=10)
        
        if response.status_code == 200:
            print("[OK] API连接成功")
            result = response.json()
            print(f"  响应: {result}")
        elif response.status_code == 405:
            print("[OK] API服务器在线（需要POST请求）")
        else:
            print(f"[ERROR] API响应异常: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("[ERROR] 连接超时")
    except requests.exceptions.ConnectionError:
        print("[ERROR] 无法连接到服务器")
    except Exception as e:
        print(f"[ERROR] 错误: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_api()
    input("\n按回车键退出...")
