"""
测试加密解密功能
"""

import json
from src.crypto_utils import crypto

# 测试数据
test_data = {
    'license_key': 'KIRO-DEMO-AAAA-BBBB-CCCC',
    'machine_id': '15d54e5e0171a64a173f97d1f82db7d121e85fcdcfc597f84cd5c8af1cf84d4b'
}

print("=" * 60)
print("测试加密解密功能")
print("=" * 60)

# 加密
print("\n1. 加密数据...")
encrypted = crypto.encrypt_request(test_data)
print(f"   加密成功")
print(f"   - encrypted_data 长度: {len(encrypted.get('encrypted_data', ''))}")
print(f"   - nonce 长度: {len(encrypted.get('nonce', ''))}")
print(f"   - tag 长度: {len(encrypted.get('tag', ''))}")
print(f"   - key_hint 长度: {len(encrypted.get('key_hint', ''))}")
print(f"   - timestamp: {encrypted.get('timestamp', '')}")

# 解密
print("\n2. 解密数据...")
try:
    decrypted = crypto.decrypt_response({
        'encrypted_data': encrypted['encrypted_data'],
        'nonce': encrypted['nonce'],
        'tag': encrypted['tag'],
        'key_hint': encrypted['key_hint'],
        'timestamp': encrypted['timestamp']
    })
    print(f"   解密成功")
    print(f"   - license_key: {decrypted.get('license_key', 'N/A')}")
    print(f"   - machine_id: {decrypted.get('machine_id', 'N/A')[:20]}...")
    
    # 验证数据一致性
    if decrypted.get('license_key') == test_data['license_key']:
        print("\n[PASSED] 加密解密测试通过！")
    else:
        print("\n[FAILED] 数据不一致！")
        
except Exception as e:
    print(f"   [FAILED] 解密失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
