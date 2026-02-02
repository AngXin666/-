"""
直接测试 API 连接
"""

import requests
import json


def test_api_connection():
    """测试 API 连接"""
    
    api_url = "https://license-serve-kk.vercel.app"
    
    print("=" * 60)
    print("测试 API 连接")
    print("=" * 60)
    
    # 测试 1: GET 请求（健康检查）
    print(f"\n1. 测试健康检查: GET {api_url}/api/activate")
    try:
        response = requests.get(f"{api_url}/api/activate", timeout=10)
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.text}")
    except requests.exceptions.Timeout:
        print("   [FAILED] 超时")
    except Exception as e:
        print(f"   [FAILED] 错误: {e}")
    
    # 测试 2: POST 激活请求
    print(f"\n2. 测试激活请求: POST {api_url}/api/activate")
    try:
        data = {
            'license_key': 'KIRO-TEST-1234-5678-ABCD',
            'machine_id': 'test-machine-123'
        }
        
        response = requests.post(
            f"{api_url}/api/activate",
            json=data,
            timeout=30  # 增加超时时间
        )
        
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
    except requests.exceptions.Timeout:
        print("   [FAILED] 超时（30秒）")
    except Exception as e:
        print(f"   [FAILED] 错误: {e}")
    
    # 测试 3: 测试 check 接口
    print(f"\n3. 测试检查接口: POST {api_url}/api/check")
    try:
        data = {
            'license_key': 'KIRO-TEST-1234-5678-ABCD',
            'machine_id': 'test-machine-123'
        }
        
        response = requests.post(
            f"{api_url}/api/check",
            json=data,
            timeout=30
        )
        
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
    except requests.exceptions.Timeout:
        print("   [FAILED] 超时（30秒）")
    except Exception as e:
        print(f"   [FAILED] 错误: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_api_connection()
