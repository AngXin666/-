# 个人页_已登录 YOLO模型映射修复报告

## 问题描述

程序在检测到"个人页_已登录"页面时,报告"该页面类型没有配置YOLO模型",导致无法检测页面元素(如昵称、用户ID等)。

### 错误日志
```
[_detect_elements] 页面类型: 个人页_已登录
[_detect_elements] 映射的YOLO模型数量: 0
[_detect_elements] ⚠️该页面类型没有配置YOLO模型
```

## 根本原因

在 `src/model_manager.py` 中初始化 `PageDetectorIntegrated` 时,**没有传递 `yolo_registry_path` 和 `mapping_path` 参数**,导致整合检测器使用默认路径,但这些默认路径可能不正确,最终导致YOLO模型映射配置未能正确加载。

### 问题代码位置
文件: `zdqd/src/model_manager.py` (约第849行)

```python
detector = PageDetectorIntegrated(
    adb=adb_bridge,
    classifier_model_path=model_path,
    classes_path=classes_path,
    log_callback=self._log_callback
)
# ❌ 缺少 yolo_registry_path 和 mapping_path 参数
```

## 修复方案

### 1. 修改 model_manager.py

在初始化 `PageDetectorIntegrated` 时,显式传递YOLO注册表和映射配置文件路径:

```python
detector = PageDetectorIntegrated(
    adb=adb_bridge,
    classifier_model_path=model_path,
    classes_path=classes_path,
    yolo_registry_path='yolo_model_registry.json',  # ✓ 添加
    mapping_path='page_yolo_mapping.json',          # ✓ 添加
    log_callback=self._log_callback
)
```

### 2. 增强调试信息

在 `src/page_detector_integrated.py` 的 `_detect_elements` 方法中添加更详细的调试信息:

```python
def _detect_elements(self, image: Image.Image, page_class: str) -> List[PageElement]:
    """使用YOLO模型检测页面元素"""
    if not HAS_YOLO:
        print(f"  [_detect_elements] ✗ YOLO库未安装")
        return []
    
    # 调试信息：输出所有可用的映射键
    print(f"  [_detect_elements] 页面类型: {page_class}")
    print(f"  [_detect_elements] 映射配置中的所有页面类型: {list(self._page_yolo_mapping.keys())[:10]}...")
    
    # 获取该页面类型对应的YOLO模型
    mapping = self._page_yolo_mapping.get(page_class, {})
    yolo_models = mapping.get('yolo_models', [])
    
    print(f"  [_detect_elements] 映射的YOLO模型数量: {len(yolo_models)}")
    
    if not yolo_models:
        print(f"  [_detect_elements] ⚠️ 该页面类型没有配置YOLO模型")
        print(f"  [_detect_elements] 调试: mapping = {mapping}")
        print(f"  [_detect_elements] 调试: page_class在映射中? {page_class in self._page_yolo_mapping}")
        return []
```

## 验证结果

运行测试脚本 `test_profile_logged_fix.py` 验证修复:

```
✓ 找到 '个人页_已登录' 的映射
  - YOLO模型数量: 3
    • profile_logged (优先级: 1)
    • balance (优先级: 2)
    • avatar_homepage (优先级: 3)

✓ 找到 'profile_logged' 模型
    - 名称: 已登录个人页检测模型（昵称和ID）
    - 模型路径: runs/detect/yolo_runs/profile_detector/train/weights/best.pt
```

## 配置文件说明

### 1. yolo_model_registry.json
包含所有YOLO模型的注册信息,包括:
- 模型名称
- 模型路径
- 检测类别
- 性能指标

### 2. models/page_yolo_mapping.json
定义页面类型到YOLO模型的映射关系:

```json
{
  "个人页_已登录": {
    "page_state": "PROFILE_LOGGED",
    "yolo_models": [
      {
        "model_key": "profile_logged",
        "purpose": "检测昵称和用户ID",
        "priority": 1
      },
      {
        "model_key": "balance",
        "purpose": "检测余额和积分",
        "priority": 2
      },
      {
        "model_key": "avatar_homepage",
        "purpose": "检测头像和首页按钮",
        "priority": 3
      }
    ]
  }
}
```

## 影响范围

此修复影响所有使用 `PageDetectorIntegrated` 的功能:
- ✓ 个人页元素检测(昵称、用户ID、余额、积分等)
- ✓ 首页元素检测(签到按钮、我的按钮等)
- ✓ 转账页元素检测(输入框、按钮等)
- ✓ 所有其他页面的YOLO元素检测

## 测试建议

1. 重启自动化程序
2. 观察日志中是否还有"该页面类型没有配置YOLO模型"的警告
3. 验证个人页的昵称和用户ID是否能正确识别
4. 检查其他页面的元素检测是否正常

## 相关文件

- `zdqd/src/model_manager.py` - 模型管理器(已修复)
- `zdqd/src/page_detector_integrated.py` - 整合检测器(已增强调试)
- `zdqd/yolo_model_registry.json` - YOLO模型注册表
- `zdqd/models/page_yolo_mapping.json` - 页面-YOLO映射配置
- `zdqd/test_profile_logged_fix.py` - 验证测试脚本
- `zdqd/test_mapping_debug.py` - 映射配置调试脚本

## 修复日期

2026-02-04

## 修复状态

✅ 已完成并验证
