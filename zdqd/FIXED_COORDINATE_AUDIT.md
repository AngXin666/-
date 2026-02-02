# 固定坐标使用审查报告

## 审查日期
2026-02-02

## 审查范围
检查所有使用固定坐标 `(450, 920)` 点击"我的"按钮的代码位置

## 发现的问题

### 1. ✅ ximeng_automation.py - `_navigate_to_profile_with_ad_handling` 方法
**状态**: 已修复

**位置**: 第549行

**修复前**:
```python
# 点击底部导航栏"我的"按钮
MY_TAB = (450, 920)
await self.adb.tap(device_id, MY_TAB[0], MY_TAB[1])
```

**修复后**:
```python
# 优先使用YOLO检测"我的"按钮位置
my_button_pos = await self.integrated_detector.find_button_yolo(
    device_id, 
    'homepage',  # 首页模型
    '我的按钮',
    conf_threshold=0.5
)

if my_button_pos:
    log(f"  YOLO检测到'我的'按钮: {my_button_pos}")
    await self.adb.tap(device_id, my_button_pos[0], my_button_pos[1])
else:
    # 降级：使用固定坐标
    MY_TAB = (450, 920)
    log(f"  YOLO未检测到按钮，使用固定坐标: {MY_TAB}")
    await self.adb.tap(device_id, MY_TAB[0], MY_TAB[1])
```

**影响**: 这是获取最终余额时使用的导航方法，修复后会优先使用YOLO检测

---

### 2. ✅ navigator.py - `navigate_to_profile` 方法
**状态**: 已修复（之前的修复）

**位置**: 第680行

**当前代码**:
```python
# 优先使用YOLO检测"我的"按钮位置（更准确）
self._silent_log.log(f"[导航到我的页面] 使用YOLO检测'我的'按钮位置...")
my_button_pos = await self.detector.find_button_yolo(
    device_id, 
    'homepage',  # 首页模型
    '我的按钮',
    conf_threshold=0.5
)

# 如果YOLO检测成功，使用检测到的坐标；否则使用固定坐标
if my_button_pos:
    self._silent_log.log(f"[导航到我的页面] YOLO检测到'我的'按钮: {my_button_pos}")
    await self.adb.tap(device_id, my_button_pos[0], my_button_pos[1])
else:
    self._silent_log.log(f"[导航到我的页面] YOLO未检测到按钮，使用固定坐标: {self.TAB_MY}")
    await self.adb.tap(device_id, self.TAB_MY[0], self.TAB_MY[1])
```

**状态**: 已经正确实现YOLO优先，固定坐标作为降级方案

---

### 3. ⚠️ navigator.py - `navigate_to_profile_optimized` 方法
**状态**: 废弃代码（未被调用）

**位置**: 第1180行

**代码**:
```python
log(f"  [导航优化] 点击'我的'按钮: {self.TAB_MY}")
await self.adb.tap(device_id, self.TAB_MY[0], self.TAB_MY[1])
```

**建议**: 这个方法没有被任何地方调用，是废弃代码。可以考虑删除或标记为废弃。

---

### 4. ⚠️ balance_reader.py - `get_account_info` 方法
**状态**: 需要评估是否修复

**位置**: 第74行

**代码**:
```python
# 确保在我的页面
if navigate_to_profile:
    result = await self.page_detector.detect_page(device_id)
    if result.state == PageState.HOME:
        await self.adb.tap(device_id, self.TAB_MY[0], self.TAB_MY[1])
        import asyncio
        await asyncio.sleep(2)
```

**分析**:
- `BalanceReader` 类已经被初始化但**从未被实际使用**
- 实际的余额读取使用的是 `ProfileReader.get_full_profile_parallel()` 方法
- `ProfileReader` 不负责导航，导航由 `ximeng_automation.py` 的 `_navigate_to_profile_with_ad_handling` 完成

**建议**: 
- 如果 `BalanceReader` 未来会被使用，应该修复这个固定坐标
- 如果确认不再使用，可以标记为废弃或删除

---

### 5. ✅ auto_login.py - `login` 方法
**状态**: 已修复

**位置**: 第677行

**修复前**:
```python
# 3. 点击"我的"标签
log("点击'我的'标签...")
await self.adb.tap(device_id, self.TAB_MY[0], self.TAB_MY[1])
await wait_after_action(min_wait=0.5, max_wait=2.0)
```

**修复后**:
```python
# 3. 点击"我的"标签
log("点击'我的'标签...")

# 优先使用YOLO检测"我的"按钮位置
my_button_pos = await self.detector.find_button_yolo(
    device_id, 
    'homepage',  # 首页模型
    '我的按钮',
    conf_threshold=0.5
)

if my_button_pos:
    log(f"  YOLO检测到'我的'按钮: {my_button_pos}")
    await self.adb.tap(device_id, my_button_pos[0], my_button_pos[1])
else:
    log(f"  YOLO未检测到按钮，使用固定坐标: {self.TAB_MY}")
    await self.adb.tap(device_id, self.TAB_MY[0], self.TAB_MY[1])

await wait_after_action(min_wait=0.5, max_wait=2.0)
```

**影响**: 登录流程现在会优先使用YOLO检测，提高准确性

---

## 修复优先级

### 高优先级（已完成）
1. ✅ `ximeng_automation.py` - `_navigate_to_profile_with_ad_handling` - **已修复**
   - 这是获取最终余额的关键路径
   - 用户反馈的问题就是这里

2. ✅ `navigator.py` - `navigate_to_profile` - **已修复**
   - 这是主要的导航方法
   - 已经在之前的修复中完成

3. ✅ `auto_login.py` - `login` 方法 - **已修复**
   - 登录流程仍在使用
   - 现在优先使用YOLO检测

### 低优先级（可选）
4. ⚠️ `balance_reader.py` - `get_account_info` 方法
   - 当前未被使用
   - 如果未来使用，需要修复

5. ⚠️ `navigator.py` - `navigate_to_profile_optimized` 方法
   - 废弃代码，未被调用
   - 可以删除或标记为废弃

---

## 修复策略

### 标准修复模式
```python
# 优先使用YOLO检测
my_button_pos = await self.detector.find_button_yolo(
    device_id, 
    'homepage',  # 首页模型
    '我的按钮',
    conf_threshold=0.5
)

if my_button_pos:
    log(f"YOLO检测到'我的'按钮: {my_button_pos}")
    await self.adb.tap(device_id, my_button_pos[0], my_button_pos[1])
else:
    # 降级：使用固定坐标
    TAB_MY = (450, 920)
    log(f"YOLO未检测到按钮，使用固定坐标: {TAB_MY}")
    await self.adb.tap(device_id, TAB_MY[0], TAB_MY[1])
```

### 优势
1. **YOLO优先**: 最准确的检测方式，适应不同分辨率
2. **固定坐标降级**: 确保在YOLO失败时仍能工作
3. **日志记录**: 清楚记录使用了哪种方法

---

## 测试建议

### 测试场景1: YOLO检测成功
**预期日志**:
```
YOLO检测到'我的'按钮: (450, 920)
✓ 到达个人页（耗时: 1.2秒，关闭广告: 0次）
✓ 最终余额: 123.45 元
```

### 测试场景2: YOLO失败，降级到固定坐标
**预期日志**:
```
YOLO未检测到按钮，使用固定坐标: (450, 920)
✓ 到达个人页（耗时: 1.5秒，关闭广告: 0次）
✓ 最终余额: 123.45 元
```

### 测试场景3: 有广告的情况
**预期日志**:
```
YOLO检测到'我的'按钮: (450, 920)
⚠️ 检测到个人页广告，立即关闭...
YOLO检测到关闭按钮: (437, 555)
✓ 到达个人页（耗时: 2.1秒，关闭广告: 1次）
✓ 最终余额: 123.45 元
```

---

## 总结

### 已完成的修复
1. ✅ `ximeng_automation.py` - 获取最终余额的导航路径
2. ✅ `navigator.py` - 主要导航方法
3. ✅ `auto_login.py` - 登录流程

### 待评估的位置
1. ⚠️ `balance_reader.py` - 余额读取器（当前未使用）
2. ⚠️ `navigator.py` - 优化版导航（废弃代码）

### 修复效果
- **获取最终余额**的流程现在会优先使用YOLO检测"我的"按钮
- **登录流程**现在会优先使用YOLO检测"我的"按钮
- 如果YOLO失败，会降级到固定坐标，确保功能可用
- 所有操作都有详细的日志记录，便于调试

### 下一步
1. 运行程序，观察日志，确认YOLO检测是否正常工作
2. 如果YOLO检测成功率高，可以考虑移除固定坐标降级
3. 清理废弃代码（`navigate_to_profile_optimized`、`balance_reader`）

---

## 相关文件
- `PROFILE_AD_FIX_AUDIT.md` - 个人页广告修复审查
- `PROFILE_AD_CLOSE_FIX.md` - 个人页广告修复详细说明
- Git commit: [待提交]
