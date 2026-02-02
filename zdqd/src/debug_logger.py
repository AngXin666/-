"""
调试日志模块 - 记录详细的执行过程和OCR识别结果
Debug Logger Module - Record detailed execution process and OCR results
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from io import BytesIO

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class DebugLogger:
    """调试日志记录器"""
    
    def __init__(self, log_dir: str = "./debug_logs"):
        """初始化调试日志记录器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建当前会话的日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.session_dir / "debug.log"
        self.screenshot_dir = self.session_dir / "screenshots"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        self.step_counter = 0
        
        # 写入日志头
        self._write_log(f"=== 调试日志开始 ===")
        self._write_log(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log(f"日志目录: {self.session_dir}")
        self._write_log("=" * 70)
    
    def _write_log(self, message: str):
        """写入日志文件"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def log_step(self, step_name: str, details: str = ""):
        """记录步骤
        
        Args:
            step_name: 步骤名称
            details: 详细信息
        """
        self.step_counter += 1
        self._write_log("")
        self._write_log(f"{'='*70}")
        self._write_log(f"步骤 {self.step_counter}: {step_name}")
        if details:
            self._write_log(f"详情: {details}")
        self._write_log(f"{'='*70}")
    
    def log_info(self, message: str):
        """记录信息"""
        self._write_log(f"[INFO] {message}")
    
    def log_warning(self, message: str):
        """记录警告"""
        self._write_log(f"[WARNING] {message}")
    
    def log_error(self, message: str):
        """记录错误"""
        self._write_log(f"[ERROR] {message}")
    
    def log_page_detection(self, state: str, confidence: float, details: str, ocr_texts: list = None):
        """记录页面检测结果
        
        Args:
            state: 页面状态
            confidence: 置信度
            details: 详细信息
            ocr_texts: OCR识别的文本列表
        """
        self._write_log(f"  页面状态: {state}")
        self._write_log(f"  置信度: {confidence:.2f}")
        self._write_log(f"  详情: {details}")
        
        if ocr_texts:
            self._write_log(f"  OCR识别文本 ({len(ocr_texts)}个):")
            for i, text in enumerate(ocr_texts[:20], 1):  # 最多显示20个
                self._write_log(f"    {i}. {text}")
            if len(ocr_texts) > 20:
                self._write_log(f"    ... 还有 {len(ocr_texts) - 20} 个文本")
    
    async def save_screenshot(self, adb, device_id: str, name: str, description: str = ""):
        """保存截图
        
        Args:
            adb: ADB桥接器
            device_id: 设备ID
            name: 截图名称
            description: 描述
        """
        try:
            if not HAS_PIL:
                self._write_log(f"  [截图] 跳过 {name} (PIL未安装)")
                return None
            
            screenshot = await adb.screencap(device_id)
            if screenshot:
                img = Image.open(BytesIO(screenshot))
                
                # 使用步骤计数器作为前缀，方便按顺序查看
                filename = f"{self.step_counter:03d}_{name}.png"
                filepath = self.screenshot_dir / filename
                
                img.save(filepath)
                self._write_log(f"  [截图] 已保存: {filename}")
                if description:
                    self._write_log(f"         描述: {description}")
                
                return filepath
        except Exception as e:
            self._write_log(f"  [截图] 保存失败: {e}")
        
        return None
    
    def log_action(self, action: str, params: dict = None):
        """记录操作
        
        Args:
            action: 操作名称
            params: 操作参数
        """
        self._write_log(f"  [操作] {action}")
        if params:
            for key, value in params.items():
                self._write_log(f"         {key}: {value}")
    
    def log_result(self, success: bool, message: str = ""):
        """记录结果
        
        Args:
            success: 是否成功
            message: 结果消息
        """
        status = "✓ 成功" if success else "✗ 失败"
        self._write_log(f"  [结果] {status}")
        if message:
            self._write_log(f"         {message}")
    
    def close(self):
        """关闭日志"""
        self._write_log("")
        self._write_log("=" * 70)
        self._write_log(f"=== 调试日志结束 ===")
        self._write_log(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log(f"总步骤数: {self.step_counter}")
        self._write_log("=" * 70)
        
        # 返回日志文件路径
        return str(self.log_file)

