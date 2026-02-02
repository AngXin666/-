# ModelManager 配置文件说明

## 概述

`model_config.json` 是 ModelManager 的配置文件，用于控制模型加载行为。通过修改此文件，您可以：

- 启用或禁用特定模型
- 配置模型文件路径
- 选择使用 GPU 或 CPU
- 调整性能参数
- 自定义错误处理行为

## 配置文件位置

配置文件应放置在项目根目录：

```
项目根目录/
├── model_config.json          # 实际使用的配置文件
├── model_config.json.example  # 示例配置文件
└── src/
    └── model_manager.py
```

## 配置文件结构

### 1. models 配置节

定义要加载的模型及其参数。

#### page_detector_integrated（深度学习页面分类器）

```json
{
  "page_detector_integrated": {
    "enabled": true,                              // 是否启用此模型
    "model_path": "page_classifier_pytorch_best.pth",  // PyTorch模型文件路径
    "classes_path": "page_classes.json",          // 类别定义文件路径
    "device": "auto",                             // 设备选择：auto/cuda/cpu
    "quantize": false                             // 是否启用模型量化
  }
}
```

**参数说明：**

- `enabled`: 是否加载此模型（true/false）
- `model_path`: PyTorch 模型文件的路径（相对于项目根目录）
- `classes_path`: 页面类别定义文件的路径
- `device`: 
  - `auto`: 自动选择（有 GPU 用 GPU，否则用 CPU）
  - `cuda`: 强制使用 GPU（如果不可用会降级到 CPU）
  - `cpu`: 强制使用 CPU
- `quantize`: 是否启用模型量化（可减少内存占用和提升速度）

#### page_detector_hybrid（YOLO 检测器）

```json
{
  "page_detector_hybrid": {
    "enabled": true,                              // 是否启用此模型
    "yolo_registry_path": "yolo_model_registry.json",  // YOLO模型注册表
    "mapping_path": "page_yolo_mapping.json",     // 页面-模型映射文件
    "device": "auto"                              // 设备选择：auto/cuda/cpu
  }
}
```

**参数说明：**

- `enabled`: 是否加载此模型
- `yolo_registry_path`: YOLO 模型注册表文件路径
- `mapping_path`: 页面类型到 YOLO 模型的映射文件路径
- `device`: 设备选择（同上）

#### ocr_thread_pool（OCR 线程池）

```json
{
  "ocr_thread_pool": {
    "enabled": true,                              // 是否启用此模型
    "thread_count": 4,                            // OCR线程数
    "use_gpu": true                               // 是否使用GPU加速
  }
}
```

**参数说明：**

- `enabled`: 是否创建 OCR 线程池
- `thread_count`: OCR 工作线程数（建议设置为 CPU 核心数的一半）
- `use_gpu`: 是否使用 GPU 加速 OCR（如果可用）

### 2. startup 配置节

控制启动时的行为。

```json
{
  "startup": {
    "show_progress": true,        // 是否显示加载进度
    "log_loading_time": true,     // 是否记录加载时间
    "log_memory_usage": true      // 是否记录内存使用
  }
}
```

**参数说明：**

- `show_progress`: 是否在控制台显示模型加载进度
- `log_loading_time`: 是否记录每个模型的加载时间
- `log_memory_usage`: 是否记录内存使用情况

### 3. performance 配置节（可选）

性能优化选项。

```json
{
  "performance": {
    "parallel_loading": false,    // 是否并行加载模型（实验性）
    "lazy_loading": false         // 是否延迟加载模型
  }
}
```

**参数说明：**

- `parallel_loading`: 是否并行加载多个模型（实验性功能，可能不稳定）
- `lazy_loading`: 是否延迟加载（首次使用时才加载，而不是启动时加载）

### 4. error_handling 配置节（可选）

错误处理行为。

```json
{
  "error_handling": {
    "retry_count": 3,             // 加载失败时的重试次数
    "retry_delay": 1,             // 重试之间的延迟（秒）
    "critical_models": [          // 关键模型列表
      "page_detector_integrated",
      "page_detector_hybrid"
    ]
  }
}
```

**参数说明：**

- `retry_count`: 模型加载失败时的重试次数
- `retry_delay`: 每次重试之间的延迟时间（秒）
- `critical_models`: 关键模型列表（这些模型加载失败会阻止程序启动）

## 使用场景

### 场景 1：开发环境（使用 GPU）

```json
{
  "models": {
    "page_detector_integrated": {
      "enabled": true,
      "device": "cuda",
      "quantize": false
    },
    "page_detector_hybrid": {
      "enabled": true,
      "device": "cuda"
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

### 场景 2：生产环境（自动选择设备）

```json
{
  "models": {
    "page_detector_integrated": {
      "enabled": true,
      "device": "auto",
      "quantize": true
    },
    "page_detector_hybrid": {
      "enabled": true,
      "device": "auto"
    },
    "ocr_thread_pool": {
      "enabled": true,
      "thread_count": 8,
      "use_gpu": true
    }
  },
  "startup": {
    "show_progress": false,
    "log_loading_time": false,
    "log_memory_usage": false
  }
}
```

### 场景 3：测试环境（仅 CPU，最小配置）

```json
{
  "models": {
    "page_detector_integrated": {
      "enabled": true,
      "device": "cpu",
      "quantize": false
    },
    "page_detector_hybrid": {
      "enabled": false
    },
    "ocr_thread_pool": {
      "enabled": false
    }
  },
  "startup": {
    "show_progress": true,
    "log_loading_time": true,
    "log_memory_usage": true
  }
}
```

### 场景 4：禁用某个模型

如果某个模型暂时不需要，可以禁用它：

```json
{
  "models": {
    "page_detector_integrated": {
      "enabled": true
    },
    "page_detector_hybrid": {
      "enabled": false  // 禁用 YOLO 检测器
    },
    "ocr_thread_pool": {
      "enabled": true
    }
  }
}
```

## 配置文件加载逻辑

1. **查找配置文件**：在项目根目录查找 `model_config.json`
2. **加载配置**：如果文件存在且格式正确，加载用户配置
3. **合并配置**：用户配置会递归合并到默认配置中
4. **使用默认值**：如果文件不存在或加载失败，使用内置的默认配置

## 配置验证

ModelManager 会在加载模型前验证配置：

- 检查必需的模型文件是否存在
- 验证设备配置是否有效
- 检查线程数等参数是否合理

如果配置无效，会：
- 记录警告日志
- 使用默认值
- 继续运行（除非是关键错误）

## 故障排除

### 问题 1：配置文件不生效

**原因**：配置文件格式错误或位置不对

**解决方案**：
1. 确认文件名为 `model_config.json`（不是 `.example`）
2. 确认文件在项目根目录
3. 使用 JSON 验证工具检查格式是否正确
4. 查看启动日志，确认配置是否被加载

### 问题 2：模型加载失败

**原因**：模型文件路径不正确

**解决方案**：
1. 检查 `model_path` 等路径是否正确
2. 确认模型文件确实存在
3. 使用相对路径（相对于项目根目录）

### 问题 3：GPU 不可用

**原因**：系统没有 GPU 或驱动未安装

**解决方案**：
1. 将 `device` 设置为 `auto` 或 `cpu`
2. ModelManager 会自动降级到 CPU 模式
3. 查看启动日志确认设备选择

## 最佳实践

1. **使用版本控制**：
   - 将 `model_config.json.example` 提交到版本控制
   - 将 `model_config.json` 添加到 `.gitignore`（包含本地配置）

2. **环境特定配置**：
   - 开发环境：启用详细日志，使用 GPU
   - 生产环境：禁用详细日志，启用量化
   - 测试环境：使用 CPU，最小配置

3. **性能优化**：
   - 根据硬件配置调整 `thread_count`
   - 在生产环境启用 `quantize` 减少内存占用
   - 使用 `auto` 设备选择以适应不同环境

4. **错误处理**：
   - 将关键模型添加到 `critical_models` 列表
   - 适当设置 `retry_count` 以应对临时错误
   - 监控启动日志以发现配置问题

## 相关文件

- `src/model_manager.py`: ModelManager 实现
- `src/MODEL_MANAGER_README.md`: ModelManager 使用文档
- `model_config.json.example`: 示例配置文件
- `.kiro/specs/model-singleton-optimization/`: 功能设计文档

## 更新日志

- **v1.0.0** (2026-01-29): 初始版本
  - 支持三种模型配置
  - 支持启动行为配置
  - 支持性能和错误处理配置
