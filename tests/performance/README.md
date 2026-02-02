# 启动和导航性能测试

本目录包含启动和导航性能优化的测试脚本。

## 测试文件

### 1. test_startup_navigation_performance.py
完整的性能测试套件，包括：
- 页面检测速度测试（模板匹配 vs OCR）
- 缓存有效性测试
- 导航到个人页面性能测试
- 启动流程性能测试

### 2. performance_comparison.py
性能对比分析工具，用于：
- 对比优化前后的性能数据
- 生成详细的对比报告
- 分析各优化点的贡献度

## 快速开始

### 前提条件
1. 模拟器已启动（MuMu或雷电）
2. 应用已安装（溪盟商城）
3. ADB服务正在运行

### 运行完整性能测试

```bash
# 使用默认配置（MuMu模拟器，端口16384）
python test_performance_quick.py

# 指定设备ID
python test_performance_quick.py 127.0.0.1:16384

# 指定设备ID和应用包名
python test_performance_quick.py 127.0.0.1:16384 com.ry.xmsc
```

### 运行功能验证测试

```bash
# 使用默认配置
python -m tests.integration.test_functional_verification

# 指定设备ID
python -m tests.integration.test_functional_verification 127.0.0.1:16384
```

## 测试结果

### 性能指标

测试将验证以下性能目标：

| 指标 | 优化前 | 目标 | 说明 |
|------|--------|------|------|
| 启动流程 | 30-60秒 | < 15秒 | 从应用启动到进入主界面 |
| 导航到个人页 | 10-15秒 | < 3秒 | 从首页到个人页 |
| 模板匹配检测 | N/A | < 0.1秒 | 页面类型识别 |
| OCR识别 | 1-3秒 | < 2秒 | 按钮位置识别 |

### 功能验证

测试将验证以下功能：
- ✓ 页面识别准确性（模板匹配 vs OCR一致性）
- ✓ 弹窗关闭功能
- ✓ 广告页检测和处理
- ✓ 导航功能（首页 ↔ 个人页）

## 测试报告

测试完成后会生成以下报告：

1. **控制台输出**：实时显示测试进度和结果
2. **性能对比报告**：`performance_comparison_report_YYYYMMDD_HHMMSS.txt`
   - 详细的性能对比数据
   - 各优化点的贡献度分析
   - 总体评估和结论

## 优化点说明

根据设计文档，主要优化点包括：

1. **模板匹配替代OCR**（贡献度40%）
   - 页面类型检测优先使用模板匹配
   - 速度提升10-20倍
   - 预期节省10-20秒

2. **轮询检测替代固定等待**（贡献度30%）
   - 弹窗关闭后使用轮询检测
   - 广告页使用轮询检测（每1秒）
   - 预期节省8-12秒

3. **检测缓存**（贡献度15%）
   - 短期缓存（TTL 0.5秒）
   - 避免重复检测
   - 预期节省3-5秒

4. **导航路径优化**（贡献度15%）
   - 减少不必要的返回操作
   - 减少验证性检测
   - 预期节省3-5秒

## 故障排查

### 连接失败
```
✗ 无法连接到设备: 127.0.0.1:16384
```

解决方法：
1. 检查模拟器是否已启动
2. 检查ADB服务：`adb devices`
3. 手动连接：`adb connect 127.0.0.1:16384`
4. 检查端口是否正确（MuMu默认16384，雷电默认5555）

### 测试超时
```
⚠️ 等待超时，继续检测...
```

解决方法：
1. 确保应用已正确安装
2. 确保模拟器性能足够
3. 增加超时时间（修改测试脚本中的max_wait_time参数）

### 页面检测失败
```
✗ 无法检测页面状态
```

解决方法：
1. 确保应用在前台运行
2. 检查模拟器分辨率是否为540x960
3. 更新模板图片（如果界面有变化）

## 开发说明

### 添加新测试

在 `test_startup_navigation_performance.py` 中添加新的测试方法：

```python
async def test_new_feature(self) -> Dict[str, Any]:
    """测试新功能
    
    Returns:
        测试结果字典
    """
    self.log("=" * 80)
    self.log("测试：新功能")
    self.log("=" * 80)
    
    # 测试逻辑
    # ...
    
    result = {
        "test_name": "新功能",
        "success": True,
        # 其他结果数据
    }
    
    return result
```

然后在 `run_all_tests()` 中调用：

```python
result = await self.test_new_feature()
all_results.append(result)
self.results.append(result)
```

### 修改性能目标

在 `performance_comparison.py` 中修改 `PERFORMANCE_TARGETS` 字典：

```python
PERFORMANCE_TARGETS = {
    "startup_time": 15.0,  # 修改启动流程目标
    "navigation_time": 3.0,  # 修改导航目标
    # ...
}
```

## 参考文档

- [需求文档](../../.kiro/specs/startup-navigation-performance/requirements.md)
- [设计文档](../../.kiro/specs/startup-navigation-performance/design.md)
- [任务列表](../../.kiro/specs/startup-navigation-performance/tasks.md)
