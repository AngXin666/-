# ModelManager 使用指南

## 概述

ModelManager是一个全局单例模式的模型管理器，负责在程序启动时预加载所有深度学习模型，并在整个程序生命周期中共享这些模型实例。

## 核心功能

1. **单例模式**：确保全局只有一个ModelManager实例
2. **预加载模型**：在程序启动时加载所有模型
3. **线程安全**：支持多线程并发访问
4. **配置驱动**：通过配置文件控制模型加载行为
5. **错误处理**：完善的错误处理和降级机制

## 快速开始

### 1. 获取单例实例

```python
from src.model_manager import ModelManager

# 获取单例实例
manager = ModelManager.get_instance()
```

### 2. 初始化所有模型

```python
# 在程序启动时初始化所有模型
stats = manager.initialize_all_models(
    adb_bridge=adb,
    log_callback=print,
    progress_callback=lambda msg, cur, total: print(f"[{cur}/{total}] {msg}")
)

# 查看加载统计
print(f"加载时间: {stats['total_time']:.2f}秒")
print(f"内存占用: {stats['memory_delta'] / 1024 / 1024:.1f}MB")
print(f"已加载模型: {stats['models_loaded']}")
```

### 3. 获取模型实例

```python
# 获取深度学习页面分类器
detector = manager.get_page_detector_integrated()

# 获取YOLO检测器
hybrid_detector = manager.get_page_detector_hybrid()

# 获取OCR线程池
ocr_pool = manager.get_ocr_thread_pool()
```

## 配置文件

### 配置文件位置

配置文件名为 `model_config.json`，应放在项目根目录。

### 配置文件示例

```json
{
  "models": {
    "page_detector_integrated": {
      "enabled": true,
      "model_path": "page_classifier_pytorch_best.pth",
      "classes_path": "page_classes.json",
      "device": "auto",
      "quantize": false
    },
    "page_detector_hybrid": {
      "enabled": true,
      "yolo_registry_path": "yolo_model_registry.json",
      "mapping_path": "page_yolo_mapping.json",
      "device": "auto"
    },
    "ocr_thread_pool": {
      "enabled": true,
      "thread_count": 4,
      "use_gpu": true
    }
  },
  "startup": {
    "show_progress": true,
    "log_loading_time": true,
    "log_memory_usage": true
  }
}
```

### 配置说明

#### models.page_detector_integrated

- `enabled`: 是否启用该模型（默认：true）
- `model_path`: PyTorch模型文件路径
- `classes_path`: 类别映射文件路径
- `device`: 设备类型（auto/cuda/cpu，默认：auto）
- `quantize`: 是否量化模型（默认：false）

#### models.page_detector_hybrid

- `enabled`: 是否启用该模型（默认：true）
- `yolo_registry_path`: YOLO模型注册表路径
- `mapping_path`: 页面YOLO映射文件路径
- `device`: 设备类型（auto/cuda/cpu，默认：auto）

#### models.ocr_thread_pool

- `enabled`: 是否启用该模型（默认：true）
- `thread_count`: OCR线程数（默认：4）
- `use_gpu`: 是否使用GPU（默认：true）

#### startup

- `show_progress`: 是否显示加载进度（默认：true）
- `log_loading_time`: 是否记录加载时间（默认：true）
- `log_memory_usage`: 是否记录内存使用（默认：true）

## API参考

### ModelManager类

#### 类方法

##### get_instance() -> ModelManager

获取单例实例。

```python
manager = ModelManager.get_instance()
```

#### 实例方法

##### initialize_all_models(adb_bridge, log_callback=None, progress_callback=None) -> Dict

初始化所有模型。

**参数：**
- `adb_bridge`: ADB桥接器实例
- `log_callback`: 日志回调函数（可选）
- `progress_callback`: 进度回调函数（可选），签名为 `(message: str, current: int, total: int)`

**返回：**
包含加载统计信息的字典：
```python
{
    'success': bool,
    'models_loaded': List[str],
    'total_time': float,
    'memory_before': int,
    'memory_after': int,
    'memory_delta': int,
    'errors': List[str]
}
```

##### get_page_detector_integrated() -> PageDetectorIntegrated

获取深度学习页面分类器（线程安全）。

**返回：** PageDetectorIntegrated实例

**异常：** RuntimeError - 如果模型未初始化

##### get_page_detector_hybrid() -> PageDetectorHybridOptimized

获取YOLO检测器（线程安全）。

**返回：** PageDetectorHybridOptimized实例

**异常：** RuntimeError - 如果模型未初始化

##### get_ocr_thread_pool() -> OCRThreadPool

获取OCR线程池（线程安全）。

**返回：** OCRThreadPool实例

**异常：** RuntimeError - 如果模型未初始化

##### is_initialized() -> bool

检查模型是否已初始化。

**返回：** 如果至少有一个模型已加载返回True

##### get_loading_stats() -> Dict

获取模型加载统计信息。

**返回：** 包含统计信息的字典

##### get_model_info(model_name: str) -> Dict

获取特定模型的信息。

**参数：**
- `model_name`: 模型名称

**返回：** 模型信息字典

**异常：** KeyError - 如果模型不存在

##### cleanup()

清理所有模型资源。

## 使用示例

### 示例1：基本使用

```python
from src.model_manager import ModelManager
from src.adb_bridge import ADBBridge

# 初始化ADB
adb = ADBBridge()

# 获取ModelManager实例
manager = ModelManager.get_instance()

# 初始化所有模型
stats = manager.initialize_all_models(adb)

if stats['success']:
    print("✓ 模型加载成功")
    
    # 使用模型
    detector = manager.get_page_detector_integrated()
    page_type = detector.detect_page()
    print(f"当前页面: {page_type}")
else:
    print("✗ 模型加载失败")
    for error in stats['errors']:
        print(f"  - {error}")
```

### 示例2：自定义回调

```python
def log_callback(message):
    """自定义日志回调"""
    print(f"[LOG] {message}")

def progress_callback(message, current, total):
    """自定义进度回调"""
    percentage = (current / total) * 100
    print(f"[{percentage:.0f}%] {message}")

# 使用自定义回调
stats = manager.initialize_all_models(
    adb_bridge=adb,
    log_callback=log_callback,
    progress_callback=progress_callback
)
```

### 示例3：在XimengAutomation中使用

```python
class XimengAutomation:
    def __init__(self, ui_automation, screen_capture, auto_login, 
                 adb_bridge=None, log_callback=None):
        # 从ModelManager获取模型
        from src.model_manager import ModelManager
        model_manager = ModelManager.get_instance()
        
        # 获取共享的检测器实例
        self.integrated_detector = model_manager.get_page_detector_integrated()
        self.hybrid_detector = model_manager.get_page_detector_hybrid()
        
        # 初始化其他组件...
```

### 示例4：程序退出时清理

```python
import atexit

def cleanup_models():
    """程序退出时清理模型"""
    manager = ModelManager.get_instance()
    if manager.is_initialized():
        print("正在清理模型资源...")
        manager.cleanup()
        print("✓ 清理完成")

# 注册退出处理函数
atexit.register(cleanup_models)
```

## 性能优化

### 预期性能收益

基于30个账号的测试场景：

- **时间节省**：每个账号节省约4秒模型加载时间，30个账号共节省约2分钟
- **内存节省**：从每个账号约300MB降低到全局共享约300MB，节省约8.7GB内存
- **日志清洁**：消除重复的模型加载日志，日志更清晰易读

### 性能监控

```python
# 获取加载统计
stats = manager.get_loading_stats()

print(f"总模型数: {stats['total_models']}")
print(f"已加载: {stats['loaded_models']}")
print(f"失败: {stats['failed_models']}")
print(f"总耗时: {stats['total_time']:.2f}秒")
print(f"内存增量: {stats['memory_delta'] / 1024 / 1024:.1f}MB")

# 查看各模型加载时间
for model_name, load_time in stats['model_times'].items():
    print(f"  - {model_name}: {load_time:.2f}秒")
```

## 故障排除

### 问题1：模型文件不存在

**错误信息：** `FileNotFoundError: 模型文件不存在: xxx.pth`

**解决方案：**
1. 检查模型文件是否存在
2. 检查配置文件中的路径是否正确
3. 确保模型文件已下载或训练完成

### 问题2：GPU不可用

**错误信息：** `⚠ GPU不可用，降级到CPU模式`

**解决方案：**
1. 检查CUDA是否正确安装
2. 检查PyTorch是否支持CUDA
3. 如果不需要GPU，可以在配置中设置 `"device": "cpu"`

### 问题3：模型未初始化

**错误信息：** `RuntimeError: 模型未初始化，请先调用initialize_all_models()`

**解决方案：**
1. 确保在使用模型前调用了 `initialize_all_models()`
2. 检查初始化是否成功（查看返回的stats）
3. 确保在程序启动时就初始化ModelManager

### 问题4：内存不足

**错误信息：** `MemoryError` 或 `CUDA out of memory`

**解决方案：**
1. 启用模型量化：设置 `"quantize": true`
2. 减少OCR线程数：设置 `"thread_count": 2`
3. 禁用不必要的模型：设置 `"enabled": false`
4. 使用CPU模式：设置 `"device": "cpu"`

## 调试加载问题

### 启用详细日志

```python
# 创建自定义日志回调，记录到文件
def debug_log_callback(message):
    with open('model_loading_debug.log', 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now()}: {message}\n")
    print(message)

# 使用调试日志
stats = manager.initialize_all_models(
    adb_bridge=adb,
    log_callback=debug_log_callback
)
```

### 生成详细报告

```python
# 生成并保存详细的加载报告
report = manager.generate_loading_report()

# 保存到文件
with open('model_loading_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
```

### 检查模型状态

```python
# 检查是否已初始化
if not manager.is_initialized():
    print("⚠ ModelManager未初始化")
else:
    print("✓ ModelManager已初始化")
    
    # 查看已加载的模型
    stats = manager.get_loading_stats()
    print(f"已加载模型: {stats['loaded_models']}")
    
    # 查看每个模型的详细信息
    for model_name in stats['models_loaded']:
        try:
            info = manager.get_model_info(model_name)
            print(f"\n{model_name}:")
            print(f"  设备: {info['device']}")
            print(f"  加载时间: {info['load_time']:.2f}秒")
            print(f"  内存占用: {info['memory_usage_mb']:.1f}MB")
        except KeyError:
            print(f"  ⚠ 无法获取模型信息")
```

### 测试模型功能

```python
# 测试页面分类器
try:
    detector = manager.get_page_detector_integrated()
    print("✓ 页面分类器可用")
    
    # 简单测试
    # result = detector.detect_page(device_id)
    # print(f"  测试结果: {result}")
except RuntimeError as e:
    print(f"✗ 页面分类器不可用: {e}")

# 测试YOLO检测器
try:
    hybrid = manager.get_page_detector_hybrid()
    print("✓ YOLO检测器可用")
except RuntimeError as e:
    print(f"✗ YOLO检测器不可用: {e}")

# 测试OCR线程池
try:
    ocr = manager.get_ocr_thread_pool()
    print("✓ OCR线程池可用")
except RuntimeError as e:
    print(f"✗ OCR线程池不可用: {e}")
```

## 添加新模型

### 步骤1：创建模型加载函数

在ModelManager类中添加新的加载函数：

```python
def _load_my_new_model(self, adb_bridge) -> 'MyNewModel':
    """加载新模型
    
    Args:
        adb_bridge: ADB桥接器实例
    
    Returns:
        MyNewModel: 新模型实例
    
    Raises:
        FileNotFoundError: 如果模型文件不存在
        RuntimeError: 如果模型加载失败
    """
    try:
        from .my_new_model import MyNewModel
    except (ImportError, ValueError):
        try:
            from src.my_new_model import MyNewModel
        except ImportError:
            import my_new_model
            MyNewModel = my_new_model.MyNewModel
    
    config = self._config['models']['my_new_model']
    
    # 验证文件存在
    model_path = config['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建实例
    self._log(f"  - 模型路径: {model_path}")
    
    model = MyNewModel(
        adb=adb_bridge,
        model_path=model_path,
        log_callback=self._log_callback
    )
    
    return model
```

### 步骤2：添加到模型列表

在 `initialize_all_models()` 方法中添加新模型：

```python
# 要加载的模型列表
models_to_load = [
    ('page_detector_integrated', self._load_page_detector_integrated),
    ('page_detector_hybrid', self._load_page_detector_hybrid),
    ('ocr_thread_pool', self._load_ocr_thread_pool),
    ('my_new_model', self._load_my_new_model),  # 新增
]
```

### 步骤3：添加访问方法

添加线程安全的访问方法：

```python
def get_my_new_model(self) -> 'MyNewModel':
    """获取新模型（线程安全）
    
    Returns:
        MyNewModel: 新模型实例
    
    Raises:
        RuntimeError: 如果模型未初始化
    """
    with self._lock:
        if 'my_new_model' not in self._models:
            raise RuntimeError(
                "MyNewModel未初始化，请先调用initialize_all_models()"
            )
        return self._models['my_new_model']
```

### 步骤4：更新配置文件

在 `model_config.json` 中添加新模型配置：

```json
{
  "models": {
    "my_new_model": {
      "enabled": true,
      "model_path": "my_model.pth",
      "device": "auto"
    }
  }
}
```

### 步骤5：更新默认配置

在 `_load_config()` 方法的默认配置中添加：

```python
default_config = {
    'models': {
        'my_new_model': {
            'enabled': True,
            'model_path': 'my_model.pth',
            'device': 'auto'
        }
    }
}
```

### 步骤6：更新文件验证

在 `_validate_model_files()` 方法中添加文件验证：

```python
# 检查新模型
if self._is_model_enabled('my_new_model'):
    config = self._config['models']['my_new_model']
    model_path = config['model_path']
    if not os.path.exists(model_path):
        missing_files.append(model_path)
```

### 步骤7：测试新模型

```python
# 测试新模型加载
manager = ModelManager.get_instance()
stats = manager.initialize_all_models(adb)

if 'my_new_model' in stats['models_loaded']:
    print("✓ 新模型加载成功")
    
    # 获取并使用新模型
    model = manager.get_my_new_model()
    # result = model.predict(...)
else:
    print("✗ 新模型加载失败")
```

## 最佳实践

1. **在程序启动时初始化**：在GUI显示前初始化所有模型
2. **使用配置文件**：通过配置文件控制模型加载行为
3. **检查初始化状态**：在使用模型前检查 `is_initialized()`
4. **处理错误**：捕获并处理模型加载失败的情况
5. **清理资源**：程序退出时调用 `cleanup()`

## 注意事项

1. ModelManager是单例模式，全局只有一个实例
2. 所有模型在程序启动时加载，之后不会重新加载
3. 模型实例在所有组件之间共享，不要修改模型状态
4. 线程安全：可以在多线程环境中安全使用
5. 配置文件可选：如果不存在，使用默认配置

## 更新日志

### v1.0.0 (2026-01-29)

- 初始版本
- 实现单例模式基础框架
- 实现配置加载功能
- 支持线程安全访问
- 支持配置驱动加载
