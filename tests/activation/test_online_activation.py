"""
测试客户端在线激活功能
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from license_manager import LicenseManager


def test_online_activation():
    """测试在线激活"""
    
    print("=" * 60)
    print("测试客户端在线激活功能")
    print("=" * 60)
    
    # 创建许可证管理器
    manager = LicenseManager()
    
    # 获取机器ID
    machine_id = manager.get_machine_id()
    print(f"\n当前机器ID: {machine_id}")
    
    # 获取API地址
    api_url = manager._get_api_url()
    print(f"API地址: {api_url}")
    
    if not api_url:
        print("\n[FAILED] 错误：未配置API地址")
        print("请检查 server/.env 文件中的 LICENSE_API_URL 配置")
        return
    
    # 测试卡密
    test_key = "KIRO-64CS-6K8L-9Y73-BSQ6"
    print(f"\n使用真实卡密: {test_key}")
    
    # 清除旧的激活信息（如果存在）
    license_file = Path("runtime_data") / "license.dat"
    if license_file.exists():
        print(f"\n清除旧的激活信息...")
        license_file.unlink()
    
    # 执行激活
    print(f"\n正在连接服务器激活...")
    print(f"API: {api_url}/api/activate")
    
    success, message = manager.activate_license(test_key, machine_id)
    
    print("\n" + "=" * 60)
    if success:
        print("[PASSED] 激活成功！")
        print(f"消息: {message}")
        
        # 检查许可证信息
        print("\n许可证信息:")
        info = manager.get_license_info()
        if info:
            print(f"  卡密: {info.get('license_key', 'N/A')}")
            print(f"  激活时间: {info.get('activated_at', 'N/A')}")
            print(f"  过期时间: {info.get('expires_at', 'N/A')}")
            print(f"  剩余天数: {info.get('days_left', 'N/A')}")
            print(f"  设备限制: {info.get('device_count', 'N/A')}/{info.get('max_devices', 'N/A')}")
        
        # 验证许可证
        print("\n验证许可证状态:")
        valid, msg, _ = manager.check_license()
        if valid:
            print(f"[PASSED] 许可证有效: {msg}")
        else:
            print(f"[FAILED] 许可证无效: {msg}")
            
    else:
        print("[FAILED] 激活失败！")
        print(f"错误: {message}")
    
    print("=" * 60)


if __name__ == "__main__":
    test_online_activation()
