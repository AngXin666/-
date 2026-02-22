"""
训练数据收集工具 - GUI版本
"""
import asyncio
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime
import threading

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.adb_bridge import ADBBridge


# 页面类别 - 从实际收集的目录读取
def get_page_classes():
    """从training_data目录读取所有页面类别"""
    data_dir = Path("training_data")
    if not data_dir.exists():
        # 如果目录不存在,返回默认类别
        return [
            '首页', '个人页_已登录', '个人页_未登录', '交易流水', '分类页',
            '加载页', '启动页服务弹窗', '商品列表', '广告页', '我的优惠劵',
            '手机号码不存在', '搜索页', '文章页', '模拟器桌面', '温馨提示',
            '用户名或密码错误弹窗', '登录页', '积分页', '签到弹窗', '签到页',
            '设置', '转账页', '钱包页', '首页公告'
        ]
    
    # 读取所有子目录作为类别
    classes = []
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            classes.append(item.name)
    
    return classes if classes else ['首页']  # 至少返回一个类别


PAGE_CLASSES = get_page_classes()


class DataCollectorGUI:
    """数据收集工具GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("训练数据收集工具")
        self.root.geometry("600x900")  # 增加高度以显示所有内容
        self.root.resizable(False, False)
        
        # 居中显示
        self.center_window()
        
        # 初始化变量
        self.screenshot_count = {page_class: 0 for page_class in PAGE_CLASSES}
        self.selected_class = tk.StringVar(value=PAGE_CLASSES[0])
        self.auto_mode = tk.BooleanVar(value=False)
        self.auto_interval = tk.IntVar(value=2)
        self.is_collecting = False
        
        # 初始化 ADB
        adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
        self.adb = ADBBridge(adb_path=adb_path)
        self.device_id = "127.0.0.1:16384"
        
        # 创建数据集目录
        self.data_dir = Path("training_data")
        self.data_dir.mkdir(exist_ok=True)
        for page_class in PAGE_CLASSES:
            (self.data_dir / page_class).mkdir(exist_ok=True)
        
        # 读取已有的截图数量
        self.load_existing_counts()
        
        # 创建界面
        self.create_widgets()
        
        # 更新统计
        self.update_stats()
    
    def load_existing_counts(self):
        """读取已有的截图数量"""
        for page_class in PAGE_CLASSES:
            class_dir = self.data_dir / page_class
            if class_dir.exists():
                # 统计该目录下的 .png 文件数量
                png_files = list(class_dir.glob("*.png"))
                self.screenshot_count[page_class] = len(png_files)
    
    def center_window(self):
        """窗口居中"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """创建界面组件"""
        # 标题
        title_frame = tk.Frame(self.root, bg='#2196F3', height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="📸 训练数据收集工具",
            font=('微软雅黑', 16, 'bold'),
            bg='#2196F3',
            fg='white'
        )
        title_label.pack(pady=15)
        
        # 主容器
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 设备信息
        info_frame = tk.LabelFrame(main_frame, text="设备信息", font=('微软雅黑', 10, 'bold'), padx=10, pady=10)
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(info_frame, text=f"设备: {self.device_id}", font=('微软雅黑', 9)).pack(anchor=tk.W)
        tk.Label(info_frame, text=f"数据集目录: {self.data_dir.absolute()}", font=('微软雅黑', 9)).pack(anchor=tk.W)
        
        # 页面类别选择
        class_frame = tk.LabelFrame(main_frame, text="选择页面类别", font=('微软雅黑', 10, 'bold'), padx=10, pady=10)
        class_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 使用网格布局显示按钮
        for i, page_class in enumerate(PAGE_CLASSES):
            row = i // 3
            col = i % 3
            
            btn = tk.Radiobutton(
                class_frame,
                text=page_class,
                variable=self.selected_class,
                value=page_class,
                font=('微软雅黑', 9),
                indicatoron=False,
                width=15,
                height=2,
                bg='#E3F2FD',
                activebackground='#2196F3',
                selectcolor='#2196F3',
                fg='black',
                activeforeground='white'
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
        
        # 配置列权重
        for i in range(3):
            class_frame.columnconfigure(i, weight=1)
        
        # 截图控制
        control_frame = tk.LabelFrame(main_frame, text="截图控制", font=('微软雅黑', 10, 'bold'), padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 按钮容器
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 手动截图按钮
        self.capture_btn = tk.Button(
            btn_frame,
            text="📷 立即截图",
            command=self.capture_screenshot,
            font=('微软雅黑', 11, 'bold'),
            bg='#4CAF50',
            fg='white',
            activebackground='#45a049',
            height=2,
            cursor='hand2'
        )
        self.capture_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # 清理缓存按钮
        self.clear_cache_btn = tk.Button(
            btn_frame,
            text="🗑️ 清理缓存",
            command=self.clear_app_cache,
            font=('微软雅黑', 11, 'bold'),
            bg='#FF9800',
            fg='white',
            activebackground='#F57C00',
            height=2,
            cursor='hand2'
        )
        self.clear_cache_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # 开始训练按钮
        self.train_btn = tk.Button(
            control_frame,
            text="🚀 开始训练模型",
            command=self.start_training,
            font=('微软雅黑', 11, 'bold'),
            bg='#2196F3',
            fg='white',
            activebackground='#1976D2',
            height=2,
            cursor='hand2'
        )
        self.train_btn.pack(fill=tk.X, pady=(10, 0))
        
        # 自动截图模式
        auto_frame = tk.Frame(control_frame)
        auto_frame.pack(fill=tk.X)
        
        self.auto_check = tk.Checkbutton(
            auto_frame,
            text="自动截图模式",
            variable=self.auto_mode,
            command=self.toggle_auto_mode,
            font=('微软雅黑', 9)
        )
        self.auto_check.pack(side=tk.LEFT)
        
        tk.Label(auto_frame, text="间隔:", font=('微软雅黑', 9)).pack(side=tk.LEFT, padx=(10, 5))
        
        interval_spin = tk.Spinbox(
            auto_frame,
            from_=1,
            to=10,
            textvariable=self.auto_interval,
            width=5,
            font=('微软雅黑', 9)
        )
        interval_spin.pack(side=tk.LEFT)
        
        tk.Label(auto_frame, text="秒", font=('微软雅黑', 9)).pack(side=tk.LEFT, padx=(5, 0))
        
        # 统计信息
        stats_frame = tk.LabelFrame(main_frame, text="收集统计", font=('微软雅黑', 10, 'bold'), padx=10, pady=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 创建表格
        columns = ('类别', '数量', '进度')
        self.stats_tree = ttk.Treeview(stats_frame, columns=columns, show='headings', height=15)  # 增加高度显示所有类别
        
        self.stats_tree.heading('类别', text='类别')
        self.stats_tree.heading('数量', text='数量')
        self.stats_tree.heading('进度', text='进度')
        
        self.stats_tree.column('类别', width=150, anchor=tk.W)
        self.stats_tree.column('数量', width=80, anchor=tk.CENTER)
        self.stats_tree.column('进度', width=200, anchor=tk.W)
        
        self.stats_tree.pack(fill=tk.BOTH, expand=True)
        
        # 总计标签
        self.total_label = tk.Label(
            main_frame,
            text="总计: 0 张",
            font=('微软雅黑', 11, 'bold'),
            fg='#2196F3'
        )
        self.total_label.pack(pady=(0, 10))
        
        # 状态栏
        self.status_label = tk.Label(
            main_frame,
            text="就绪",
            font=('微软雅黑', 9),
            fg='green',
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X)
    
    def update_stats(self):
        """更新统计信息"""
        # 清空表格
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        # 添加数据
        total = 0
        for page_class in PAGE_CLASSES:
            count = self.screenshot_count[page_class]
            total += count
            
            # 计算进度（目标30张）
            target = 30
            progress = min(count / target * 100, 100)
            progress_bar = '█' * int(progress / 10) + '░' * (10 - int(progress / 10))
            progress_text = f"{progress_bar} {count}/{target}"
            
            self.stats_tree.insert('', tk.END, values=(page_class, count, progress_text))
        
        # 更新总计
        self.total_label.config(text=f"总计: {total} 张")
    
    def clear_app_cache(self):
        """清理应用缓存"""
        if self.is_collecting:
            return
        
        # 在新线程中执行
        thread = threading.Thread(target=self._clear_cache_async)
        thread.daemon = True
        thread.start()
    
    def _clear_cache_async(self):
        """异步清理缓存"""
        try:
            self.is_collecting = True
            self.update_status("正在清理应用缓存...", 'blue')
            
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            package_name = "com.jingtuapp.app"  # 应用包名
            activity_name = "com.jingtuapp.app.MainActivity"
            
            # 1. 停止应用
            self.update_status("1/3 停止应用...", 'blue')
            loop.run_until_complete(
                self.adb.stop_app(self.device_id, package_name)
            )
            loop.run_until_complete(asyncio.sleep(1))
            
            # 2. 清理缓存
            self.update_status("2/3 清理缓存...", 'blue')
            # 方法1：尝试使用 pm clear-cache
            result = loop.run_until_complete(
                self.adb.shell(self.device_id, f"pm clear-cache {package_name}")
            )
            
            if "Unknown" in result or "Error" in result:
                # 方法2：如果不支持，使用 rm 命令
                result = loop.run_until_complete(
                    self.adb.shell(self.device_id, f"rm -rf /data/data/{package_name}/cache/*")
                )
            
            loop.run_until_complete(asyncio.sleep(2))
            
            # 3. 重新启动应用
            self.update_status("3/3 启动应用...", 'blue')
            success = loop.run_until_complete(
                self.adb.start_app(self.device_id, package_name, activity_name)
            )
            
            loop.run_until_complete(asyncio.sleep(3))
            
            if success:
                self.update_status("✓ 缓存清理成功，应用已重启", 'green')
            else:
                self.update_status("⚠️ 应用启动失败，请手动启动", 'orange')
            
        except Exception as e:
            self.update_status(f"❌ 清理失败: {e}", 'red')
        finally:
            self.is_collecting = False
    
    def start_training(self):
        """开始训练模型"""
        if self.is_collecting:
            return
        
        # 检查数据量
        total = sum(self.screenshot_count.values())
        if total < 50:
            messagebox.showwarning(
                "数据不足",
                f"当前只有 {total} 张截图\n\n建议至少收集 100 张截图再训练\n(每个类别至少 20 张)"
            )
            return
        
        # 确认训练
        result = messagebox.askyesno(
            "确认训练",
            f"即将开始训练模型\n\n"
            f"总截图数: {total} 张\n"
            f"类别数: {len(PAGE_CLASSES)} 个\n\n"
            f"训练可能需要 10-20 分钟\n"
            f"确定要开始吗?"
        )
        
        if not result:
            return
        
        # 在新线程中执行训练
        thread = threading.Thread(target=self._train_async)
        thread.daemon = True
        thread.start()
    
    def _train_async(self):
        """异步训练模型"""
        try:
            self.is_collecting = True
            self.update_status("正在训练模型,请稍候...", 'blue')
            
            # 禁用按钮
            self.root.after(0, lambda: self.train_btn.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.capture_btn.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.clear_cache_btn.config(state=tk.DISABLED))
            
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 调用训练脚本
            import subprocess
            result = subprocess.run(
                ["python", "train_page_classifier.py"],
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )
            
            if result.returncode == 0:
                self.update_status("✓ 模型训练完成!", 'green')
                self.root.after(0, lambda: messagebox.showinfo(
                    "训练完成",
                    "模型训练成功!\n\n"
                    "模型文件: page_classifier.h5\n"
                    "现在可以集成到主程序中使用了"
                ))
            else:
                error_msg = result.stderr if result.stderr else "未知错误"
                self.update_status(f"✗ 训练失败: {error_msg[:50]}", 'red')
                self.root.after(0, lambda: messagebox.showerror(
                    "训练失败",
                    f"训练过程出错:\n\n{error_msg[:200]}"
                ))
            
        except subprocess.TimeoutExpired:
            self.update_status("✗ 训练超时(30分钟)", 'red')
            self.root.after(0, lambda: messagebox.showerror(
                "训练超时",
                "训练时间超过30分钟,已自动停止"
            ))
        except Exception as e:
            self.update_status(f"✗ 训练异常: {e}", 'red')
            self.root.after(0, lambda: messagebox.showerror(
                "训练异常",
                f"训练过程出现异常:\n\n{str(e)}"
            ))
        finally:
            self.is_collecting = False
            # 恢复按钮
            self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.capture_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.clear_cache_btn.config(state=tk.NORMAL))
    
    def capture_screenshot(self):
        """截图"""
        if self.is_collecting:
            return
        
        page_class = self.selected_class.get()
        
        # 在新线程中执行异步操作
        thread = threading.Thread(target=self._capture_async, args=(page_class,))
        thread.daemon = True
        thread.start()
    
    def _capture_async(self, page_class):
        """异步截图"""
        try:
            self.is_collecting = True
            self.update_status(f"正在截图: {page_class}...", 'blue')
            
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 截图
            screenshot_data = loop.run_until_complete(self.adb.screencap(self.device_id))
            
            if not screenshot_data:
                self.update_status("❌ 截图失败", 'red')
                return
            
            # 保存截图
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{page_class}_{timestamp}.png"
            filepath = self.data_dir / page_class / filename
            
            with open(filepath, 'wb') as f:
                f.write(screenshot_data)
            
            # 更新计数
            self.screenshot_count[page_class] += 1
            
            # 更新界面
            self.root.after(0, self.update_stats)
            self.update_status(f"✓ 已保存: {page_class} ({self.screenshot_count[page_class]} 张)", 'green')
            
        except Exception as e:
            self.update_status(f"❌ 错误: {e}", 'red')
        finally:
            self.is_collecting = False
    
    def toggle_auto_mode(self):
        """切换自动模式"""
        if self.auto_mode.get():
            self.start_auto_capture()
        else:
            self.stop_auto_capture()
    
    def start_auto_capture(self):
        """开始自动截图"""
        self.capture_btn.config(state=tk.DISABLED)
        self.update_status("🔄 自动截图模式已启动", 'blue')
        self.auto_capture_loop()
    
    def stop_auto_capture(self):
        """停止自动截图"""
        self.capture_btn.config(state=tk.NORMAL)
        self.update_status("⏸ 自动截图模式已停止", 'orange')
    
    def auto_capture_loop(self):
        """自动截图循环"""
        if self.auto_mode.get():
            self.capture_screenshot()
            # 继续下一次
            interval = self.auto_interval.get() * 1000
            self.root.after(interval, self.auto_capture_loop)
    
    def update_status(self, message, color='black'):
        """更新状态"""
        self.root.after(0, lambda: self.status_label.config(text=message, fg=color))
    
    def on_closing(self):
        """关闭窗口"""
        total = sum(self.screenshot_count.values())
        if total > 0:
            result = messagebox.askyesno(
                "确认退出",
                f"已收集 {total} 张截图\n\n确定要退出吗？"
            )
            if result:
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """主函数"""
    root = tk.Tk()
    app = DataCollectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
