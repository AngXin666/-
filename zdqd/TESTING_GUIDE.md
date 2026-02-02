# 自动化测试指南

## 概述

本项目包含自动化测试，用于防止常见的代码问题，特别是：
- 导入错误
- 核心方法缺失
- 废弃代码引用

## 测试文件

### 1. `tests/test_imports.py` - 导入测试
**目的**：确保所有模块可以正确导入

**测试内容**：
- ✅ 核心模块可以导入（XimengAutomation, AutoLogin, GUI 等）
- ✅ PageState 可以从正确的模块导入
- ✅ 跨模块导入的一致性
- ✅ 已删除的模块不被引用

**防止的问题**：
- `ModuleNotFoundError: No module named 'src.page_detector_hybrid_optimized'`
- 导入了已删除的模块
- 导入路径不一致

### 2. `tests/test_core_methods.py` - 核心方法测试
**目的**：确保关键方法存在且可调用

**测试内容**：
- ✅ XimengAutomation.run_full_workflow 方法存在
- ✅ XimengAutomation.handle_startup_flow_integrated 方法存在
- ✅ XimengAutomation._navigate_to_profile_with_ad_handling 方法存在
- ✅ GUI._process_account_with_instance 方法正确导入 time 模块
- ✅ 废弃方法已被删除

**防止的问题**：
- `'XimengAutomation' object has no attribute 'run_full_workflow'`
- `UnboundLocalError: cannot access local variable 'time'`
- 核心方法被误删除

---

## 运行测试

### 方法 1：使用测试脚本（推荐）
```bash
python run_tests.py
```

### 方法 2：直接使用 pytest
```bash
# 运行所有测试
pytest tests/test_imports.py tests/test_core_methods.py -v

# 只运行导入测试
pytest tests/test_imports.py -v

# 只运行核心方法测试
pytest tests/test_core_methods.py -v
```

### 方法 3：运行特定测试类
```bash
# 运行特定测试类
pytest tests/test_imports.py::TestCoreImports -v

# 运行特定测试方法
pytest tests/test_imports.py::TestCoreImports::test_import_ximeng_automation -v
```

---

## 测试输出示例

### ✅ 成功输出
```
============================================================
运行自动化测试
============================================================

测试参数: tests/test_imports.py tests/test_core_methods.py -v --tb=short --color=yes

tests/test_imports.py::TestCoreImports::test_import_ximeng_automation PASSED
tests/test_imports.py::TestCoreImports::test_import_auto_login PASSED
tests/test_core_methods.py::TestXimengAutomationMethods::test_run_full_workflow_exists PASSED

============================================================
✅ 所有测试通过！
============================================================
```

### ❌ 失败输出
```
tests/test_imports.py::TestPageStateImports::test_pagestate_not_from_hybrid_optimized FAILED

FAILED tests/test_imports.py::TestPageStateImports::test_pagestate_not_from_hybrid_optimized
AssertionError: 在 src/auto_login.py:75 发现对已删除模块 'page_detector_hybrid_optimized' 的引用

============================================================
❌ 测试失败！
============================================================
```

---

## 自动化测试

### Pre-commit Hook
在每次 Git 提交前自动运行测试：

**Windows**：
```bash
# Hook 已配置在 .git/hooks/pre-commit.bat
# 每次 git commit 时自动运行
```

**Linux/Mac**：
```bash
# Hook 已配置在 .git/hooks/pre-commit
# 需要设置执行权限
chmod +x .git/hooks/pre-commit
```

### CI/CD (GitHub Actions)
在每次推送到 GitHub 时自动运行测试：

**配置文件**：`.github/workflows/tests.yml`

**触发条件**：
- Push 到 master/main/develop 分支
- Pull Request 到 master/main/develop 分支

---

## 添加新测试

### 1. 创建新测试文件
```python
# tests/test_new_feature.py
import pytest

class TestNewFeature:
    def test_something(self):
        assert True
```

### 2. 运行新测试
```bash
pytest tests/test_new_feature.py -v
```

### 3. 添加到测试套件
编辑 `run_tests.py`，添加新测试文件：
```python
args = [
    'tests/test_imports.py',
    'tests/test_core_methods.py',
    'tests/test_new_feature.py',  # 新增
    '-v',
]
```

---

## 测试最佳实践

### 1. 测试命名
- 测试文件：`test_*.py`
- 测试类：`Test*`
- 测试方法：`test_*`

### 2. 测试结构
```python
class TestFeature:
    """测试某个功能"""
    
    def test_basic_case(self):
        """测试基本情况"""
        # Arrange（准备）
        # Act（执行）
        # Assert（断言）
        pass
    
    def test_edge_case(self):
        """测试边界情况"""
        pass
    
    def test_error_case(self):
        """测试错误情况"""
        pass
```

### 3. 断言消息
```python
# 好的断言
assert value is not None, "值不应该为 None"

# 不好的断言
assert value is not None
```

### 4. 使用 pytest 特性
```python
# 测试异常
with pytest.raises(ValueError):
    some_function()

# 参数化测试
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
])
def test_multiply(input, expected):
    assert input * 2 == expected
```

---

## 故障排除

### 问题 1：pytest 未安装
```bash
pip install pytest pytest-asyncio
```

### 问题 2：导入错误
```bash
# 确保在项目根目录运行测试
cd zdqd
python -m pytest tests/test_imports.py -v
```

### 问题 3：测试失败
1. 查看详细错误信息
2. 检查相关代码
3. 修复问题
4. 重新运行测试

---

## 维护测试

### 定期检查
- 每周运行一次完整测试套件
- 在重大重构前后运行测试
- 在发布前运行测试

### 更新测试
- 添加新功能时，添加相应测试
- 修复 Bug 时，添加回归测试
- 删除代码时，更新相关测试

### 测试覆盖率
```bash
# 安装 coverage
pip install pytest-cov

# 运行测试并生成覆盖率报告
pytest tests/ --cov=src --cov-report=html

# 查看报告
# 打开 htmlcov/index.html
```

---

## 相关文档

- `BUGFIX_COMPLETE_SUMMARY.md` - Bug 修复总结
- `COMPLETE_CLEANUP_SUMMARY.md` - 代码清理总结
- `pytest.ini` - Pytest 配置文件
- `.github/workflows/tests.yml` - CI/CD 配置

---

## 总结

自动化测试帮助我们：
- ✅ 及早发现问题
- ✅ 防止回归
- ✅ 提高代码质量
- ✅ 增强信心

**记住**：测试不是负担，而是保护网！
