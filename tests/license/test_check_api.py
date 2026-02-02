"""
测试 /api/check 端点
"""

import requests
import json
from src.crypto_utils import crypto

# API 地址
API_URL = "https://license-serve-kk.vercel.app"

# 测试数据（使用实际的卡密和机器ID）
test_data = {
    'license_key': 'MQ86-NZ74-2DB1-MELL-KX3Q',
    'machine_id': '15d54e5e0171a64a173f97d1f82db7d121e85fcdcfc597f84cd5c8af1cf84d4b'
}

print("=" * 60)
print("测试 /api/check 端点")
print("=" * 60)

try:
    # 加密请求
    print("\n1. 加密请求数据...")
    encrypted_request = crypto.encrypt_request(test_data)
    print(f"   [OK] 加密成功")
    
    # 发送请求
    print(f"\n2. 发送请求到: {API_URL}/api/check")
    response = requests.post(
        f"{API_URL}/api/check",
        json=encrypted_request,
        timeout=15,  # 15秒超时
        headers={
            'Content-Type': 'application/json',
            'User-Agent': 'XiMengAutomation/1.0'
        }
    )
    
    print(f"   [OK] 响应状态码: {response.status_code}")
    
    # 解密响应
    print("\n3. 解密响应数据...")
    encrypted_response = response.json()
    result = crypto.decrypt_response(encrypted_response)
    
    print(f"   [OK] 解密成功")
    
    # 显示结果
    print("\n4. 响应内容:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if result.get('success'):
        print("\n[PASSED] 测试成功！/api/check 端点工作正常")
        data = result.get('data', {})
        print(f"\n许可证信息:")
        print(f"  - 状态: {data.get('status')}")
        print(f"  - 到期时间: {data.get('expires_at')}")
        print(f"  - 设备数: {data.get('device_count')}/{data.get('max_devices')}")
    else:
        print(f"\n[FAILED] 测试失败: {result.get('message')}")
    
except requests.exceptions.Timeout:
    print("\n[FAILED] 连接超时")
except requests.exceptions.ConnectionError:
    print("\n[FAILED] 无法连接到服务器")
except Exception as e:
    print(f"\n[FAILED] 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
