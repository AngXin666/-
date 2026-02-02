# 文件清理计划 - src 目录

## 分析结果

### 📊 统计
- **总文件数**: 94 个
- **使用中**: 53 个（56%）
- **可清理**: 33 个（35%）
  - 空文件: 5 个
  - 备份文件: 8 个
  - 未使用文件: 20 个
- **可清理空间**: 0.60 MB

## 清理分类

### 🗑️ 空文件（5个）- 立即删除

这些文件完全是空的，没有任何代码：

1. `dl_page_guard.py` - 0 bytes
2. `encoding_utils.py` - 0 bytes
3. `ocr_helper.py` - 0 bytes
4. `startup_flow_integrated.py` - 0 bytes
5. `transfer_detail_reader.py` - 0 bytes

**建议**: ✅ 立即删除

---

### 📦 备份/优化版本文件（8个）- 建议删除

这些是旧版本的备份文件，已被新版本替代：

1. `page_detector_hybrid_backup.py` - 54,543 bytes
2. `page_detector_hybrid_optimized.py` - 6,185 bytes
3. `page_detector_ocr_backup.py` - 11,912 bytes
4. `page_detector_optimized.py` - 5,688 bytes
5. `page_detector_pixel_backup.py` - 27,187 bytes
6. `ximeng_automation_backup_20260129_123808.py` - 133,327 bytes
7. `ximeng_automation_backup_20260202.py` - 173,655 bytes ⚠️ 今天刚创建
8. `ximeng_automation_optimized.py` - 25,618 bytes

**建议**: 
- ✅ 删除 1-6 和 8（旧备份）
- ⚠️ 保留 7（今天的备份，作为安全保障）

---

### ⚠️ 未使用的文件（20个）- 需要确认

这些文件没有被主程序导入，可能是：
- 废弃的旧代码
- 示例代码
- 工具脚本

#### 可能废弃的文件（建议删除）

1. **activation_dialog.py** (8,240 bytes)
   - 已被 `simple_activation_dialog.py` 替代

2. **debug_logger.py** (5,841 bytes)
   - 已有 `logger.py` 和 `logging_config.py`

3. **error_handling.py** (9,231 bytes)
   - 功能已整合到其他模块

4. **local_db_refactored.py** (17,732 bytes)
   - 重构版本，已被 `local_db.py` 替代

5. **login_handler.py** (11,684 bytes)
   - 功能已整合到 `auto_login.py`

6. **multi_emulator_manager.py** (10,937 bytes)
   - 已被 `instance_manager.py` 替代

7. **native_activation_dialog.py** (7,412 bytes)
   - 已被 `simple_activation_dialog.py` 替代

8. **page_detector_ocr.py** (11,912 bytes)
   - 已被 `page_detector_integrated.py` 替代

9. **profile_data_reader.py** (18,229 bytes)
   - 功能已整合到 `profile_reader.py`

10. **single_instance.py** (5,370 bytes)
    - 单例功能已整合

11. **workflow_controller.py** (13,738 bytes)
    - 功能已整合到 `orchestrator.py`

#### 示例/工具文件（可选删除）

12. **ocr_image_processor_example.py** (4,255 bytes)
    - 示例代码，可删除

13. **ocr_usage_examples.py** (8,365 bytes)
    - 示例代码，可删除

14. **template_encryptor.py** (7,737 bytes)
    - 工具脚本，如果不再使用可删除

#### 可能有用的文件（建议保留）

15. **main.py** (4,518 bytes)
    - 可能是备用入口，建议保留

16. **model_updater.py** (7,401 bytes)
    - 模型更新工具，可能有用

17. **dl_page_guard_config.py** (7,777 bytes)
    - 配置文件，可能有用

18. **recipient_selector.py** (6,984 bytes)
    - 收款人选择器，可能有用

19. **resource_manager.py** (12,831 bytes)
    - 资源管理器，可能有用

20. **transfer_retry.py** (5,652 bytes)
    - 转账重试逻辑，可能有用

---

## 清理方案

### 方案 1：保守清理（推荐）

只删除明确废弃的文件：

**立即删除**：
- 5 个空文件
- 7 个旧备份文件（保留今天的备份）
- 11 个明确废弃的文件

**总计**: 23 个文件，约 0.35 MB

### 方案 2：激进清理

删除所有未使用的文件：

**删除**：
- 5 个空文件
- 8 个备份文件（包括今天的）
- 20 个未使用文件

**总计**: 33 个文件，约 0.60 MB

---

## 执行步骤

### 步骤 1：创建备份

```bash
# 创建整个 src 目录的备份
cp -r zdqd/src zdqd/src_backup_20260202
```

### 步骤 2：删除空文件

```bash
rm zdqd/src/dl_page_guard.py
rm zdqd/src/encoding_utils.py
rm zdqd/src/ocr_helper.py
rm zdqd/src/startup_flow_integrated.py
rm zdqd/src/transfer_detail_reader.py
```

### 步骤 3：删除旧备份文件

```bash
rm zdqd/src/page_detector_hybrid_backup.py
rm zdqd/src/page_detector_hybrid_optimized.py
rm zdqd/src/page_detector_ocr_backup.py
rm zdqd/src/page_detector_optimized.py
rm zdqd/src/page_detector_pixel_backup.py
rm zdqd/src/ximeng_automation_backup_20260129_123808.py
rm zdqd/src/ximeng_automation_optimized.py
```

### 步骤 4：删除废弃文件

```bash
rm zdqd/src/activation_dialog.py
rm zdqd/src/debug_logger.py
rm zdqd/src/error_handling.py
rm zdqd/src/local_db_refactored.py
rm zdqd/src/login_handler.py
rm zdqd/src/multi_emulator_manager.py
rm zdqd/src/native_activation_dialog.py
rm zdqd/src/page_detector_ocr.py
rm zdqd/src/profile_data_reader.py
rm zdqd/src/single_instance.py
rm zdqd/src/workflow_controller.py
```

### 步骤 5：删除示例文件（可选）

```bash
rm zdqd/src/ocr_image_processor_example.py
rm zdqd/src/ocr_usage_examples.py
rm zdqd/src/template_encryptor.py
```

### 步骤 6：验证

```bash
# 运行程序测试
python zdqd/run.py

# 检查是否有导入错误
python -m py_compile zdqd/src/*.py
```

---

## 风险评估

### 低风险 ✅
- 空文件：完全没有代码
- 旧备份文件：已有新版本
- 明确废弃的文件：已被替代

### 中风险 ⚠️
- 未使用但可能有用的文件：需要确认是否真的不需要

---

## 预期收益

1. **文件数量**: 从 94 个减少到 71 个（减少 24%）
2. **代码清晰度**: 大幅提升
3. **维护成本**: 降低
4. **磁盘空间**: 节省 0.35-0.60 MB

---

## 状态
⏳ 待执行

## 日期
2026-02-02
