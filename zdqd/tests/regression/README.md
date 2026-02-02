# 性能回归测试

## 概述

性能回归测试用于确保ModelManager的性能改善持续，防止未来的代码变更导致性能退化。

## 测试内容

### 1. 模型加载时间测试 (`test_model_loading_time`)
- 验证所有模型加载时间不超过基准值（15秒）
- 允许20%的性能波动

### 2. 模型访问时间测试 (`test_model_access_time`)
- 验证模型访问时间保持在毫秒级（<10ms）
- 测试100次访问的平均时间和最大时间

### 3. 单账号处理时间测试 (`test_single_account_processing_time`)
- 验证单账号处理时间不超过基准值（1秒）
- 包括获取所有模型的时间

### 4. 内存占用测试 (`test_memory_usage`)
- 验证总内存占用不超过基准值（1.5GB）
- 验证访问模型不会显著增加内存（<10MB）

### 5. 模型实例复用测试 (`test_model_instance_reuse`)
- 验证多次获取同一模型返回相同实例
- 验证访问时间稳定（标准差小）

### 6. 并发访问性能测试 (`test_concurrent_access_performance`)
- 验证10个线程并发访问模型的性能
- 验证线程安全性和访问时间

### 7. 初始化一致性测试 (`test_initialization_consistency`)
- 验证多次初始化加载的模型一致
- 验证初始化时间在合理范围内

### 8. 性能对比测试 (`test_compare_with_baseline`)
- 对比当前性能与基准性能
- 计算性能改善百分比

## 性能基准值

```python
PERFORMANCE_BASELINES = {
    'model_loading_time': 15.0,  # 秒，所有模型加载时间
    'model_access_time': 0.01,   # 秒，模型访问时间
    'single_account_time': 1.0,  # 秒，单账号处理时间
    'memory_per_model': 500 * 1024 * 1024,  # 字节，每个模型内存
    'total_memory': 1500 * 1024 * 1024,  # 字节，总内存
}
```

## 运行测试

### 运行所有性能回归测试
```bash
pytest tests/regression/test_performance_regression.py -v
```

### 运行特定测试
```bash
# 只测试模型加载时间
pytest tests/regression/test_performance_regression.py::TestPerformanceRegression::test_model_loading_time -v

# 只测试内存占用
pytest tests/regression/test_performance_regression.py::TestPerformanceRegression::test_memory_usage -v
```

### 显示详细输出
```bash
pytest tests/regression/test_performance_regression.py -v -s
```

## 性能容忍度

测试允许20%的性能波动（`TOLERANCE = 1.2`），以适应不同硬件环境和系统负载的影响。

例如：
- 如果基准值是10秒，最大允许值是12秒（10 * 1.2）
- 如果基准值是100MB，最大允许值是120MB（100 * 1.2）

## 更新基准值

如果进行了性能优化，可以更新基准值：

1. 运行当前测试并记录实际性能
2. 更新 `PERFORMANCE_BASELINES` 字典中的值
3. 重新运行测试确保通过
4. 提交更新后的基准值

## 持续集成

建议在CI/CD流程中运行性能回归测试：

```yaml
# .github/workflows/performance-test.yml
name: Performance Regression Tests

on: [push, pull_request]

jobs:
  performance-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      - name: Run performance regression tests
        run: pytest tests/regression/test_performance_regression.py -v
```

## 故障排查

### 测试失败：模型加载时间超标
- 检查是否有其他程序占用GPU
- 检查模型文件是否完整
- 检查网络连接（如果模型需要下载）

### 测试失败：内存占用超标
- 检查是否有内存泄漏
- 检查是否有未释放的资源
- 运行 `cleanup()` 方法确保资源释放

### 测试失败：并发访问性能
- 检查线程锁是否正确实现
- 检查是否有死锁或竞态条件
- 增加日志输出调试

## 相关文档

- [ModelManager README](../../src/MODEL_MANAGER_README.md)
- [性能基准测试](../../benchmark_model_manager.py)
- [性能报告生成](../../generate_performance_report.py)
