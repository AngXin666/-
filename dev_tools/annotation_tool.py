"""
简化标注工具 - 用于标注按钮和元素位置
"""
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from pathlib import Path
import json
from collections import Counter


# 元素类别定义
ELEMENT_CLASSES = [
    "同意按钮",
    "拒绝按钮",
    "确认按钮",
    "关闭按钮",
    "跳过按钮",
    "返回按钮",
    "登陆按钮",  # 登录页的登陆按钮
    "签到按钮",
    "每日签到按钮",  # 首页的每日签到入口按钮
    "转账按钮",
    "转增按钮",  # 钱包页的转增按钮
    "提交按钮",  # 转账页的提交按钮
    "首页按钮",  # 分类页到首页
    "我的按钮",  # 首页到个人页
    "请登陆按钮",  # 未登陆页面的登录按钮
    "抵扣劵数字",  # 个人页的抵扣劵数字
    "优惠劵数字",  # 个人页的优惠劵数字
    "协议勾选框",  # 登录页的协议勾选框
    "签到成功文本",  # 签到弹窗的"签到成功"文字
    "签到金额",  # 签到弹窗的金额数字
    "签到次数",  # 签到页的签到次数文本区域
    "余额数字",
    "积分数字",
    "昵称文本",
    "用户ID",
    "头像",  # 个人页的用户头像
    "账号输入框",  # 登录页的账号输入框
    "密码输入框",  # 登录页的密码输入框
    "转账金额输入框",  # 转账页的金额输入框
    "ID输入框",  # 转账页的ID输入框
    "全部转账按钮",  # 转账页的全部转账按钮
    "转账确认ID",  # 转账确认弹窗的ID
    "转账确认昵称",  # 转账确认弹窗的昵称
    "转账确认金额",  # 转账确认弹窗的金额
    "转账明细文本",  # 转账明细的文字区域（用于OCR识别）
    "其他",
]


class AnnotationTool:
    """标注工具"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("元素标注工具")
        self.root.geometry("1200x800")
        
        # 数据目录
        self.data_dir = Path("training_data")
        
        # 布局配置文件
        self.layout_config_file = Path("annotation_layouts.json")
        self.layout_configs = self.load_layout_configs()
        
        # 当前状态
        self.current_category = None
        self.current_images = []
        self.current_index = 0
        self.current_image = None
        self.current_photo = None
        self.annotations = {}  # {image_path: [annotations]}
        
        # 临时标注
        self.temp_rect = None
        self.start_x = None
        self.start_y = None
        
        # 多边形标注模式
        self.polygon_mode = False  # 是否启用多边形模式
        self.polygon_points = []  # 多边形的点列表
        self.temp_polygon = None  # 临时多边形线条
        self.is_drawing = False  # 是否正在绘制
        
        # 调整模式相关
        self.selected_annotation = None  # 当前选中的标注索引
        self.dragging_annotation = None  # 正在拖动的标注
        self.drag_start_x = None
        self.drag_start_y = None
        self.resize_handle = None  # 正在调整大小的句柄（'tl', 'tr', 'bl', 'br', 'edge'）
        
        # 显示选项
        self.show_saved_annotations = False  # 默认不显示已保存的标注
        
        # OCR识别结果
        self.ocr_results = []  # 存储OCR识别的文字和位置
        self.show_ocr_results = False  # 是否显示OCR结果
        
        # 创建界面
        self.create_widgets()
        
        # 加载类别
        self.load_categories()
    
    def load_layout_configs(self):
        """加载布局配置"""
        if self.layout_config_file.exists():
            try:
                with open(self.layout_config_file, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
                print(f"✓ 已加载布局配置文件: {self.layout_config_file}")
                print(f"✓ 包含 {len(configs)} 个类别的布局配置: {list(configs.keys())}")
                return configs
            except Exception as e:
                print(f"✗ 加载布局配置失败: {e}")
                return {}
        else:
            print(f"ℹ 布局配置文件不存在: {self.layout_config_file}")
            print(f"ℹ 使用'学习布局'功能后会自动创建")
            return {}
    
    def save_layout_config(self, category, layout):
        """保存布局配置"""
        # 更新内存中的配置（立即生效）
        self.layout_configs[category] = layout
        
        # 确保目录存在
        self.layout_config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存到文件
        with open(self.layout_config_file, 'w', encoding='utf-8') as f:
            json.dump(self.layout_configs, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 已保存 {category} 的布局配置到 {self.layout_config_file}")
        print(f"✓ 当前配置包含 {len(self.layout_configs)} 个类别: {list(self.layout_configs.keys())}")
        print(f"✓ 布局配置已立即生效，无需重启")
    
    def learn_layout_from_annotations(self):
        """从当前标注学习布局（手动触发）"""
        if not self.current_category or not self.current_images:
            messagebox.showwarning("警告", "请先选择页面类别")
            return
        
        image_path = str(self.current_images[self.current_index])
        if image_path not in self.annotations or not self.annotations[image_path]:
            messagebox.showinfo("提示", "当前图片没有标注，无法学习布局")
            return
        
        # 获取图片尺寸
        img_width, img_height = self.current_image.size
        
        # 找到整体边界框
        anns = self.annotations[image_path]
        
        # 调试信息
        print(f"\n=== 学习布局 ===")
        print(f"类别: {self.current_category}")
        print(f"图片: {Path(image_path).name}")
        print(f"标注数量: {len(anns)}")
        
        min_x = min(ann['x1'] for ann in anns)
        min_y = min(ann['y1'] for ann in anns)
        max_x = max(ann['x2'] for ann in anns)
        max_y = max(ann['y2'] for ann in anns)
        
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        print(f"整体边界框: ({min_x:.1f}, {min_y:.1f}) -> ({max_x:.1f}, {max_y:.1f})")
        print(f"边界框尺寸: {box_width:.1f} x {box_height:.1f}")
        
        # 检查边界框尺寸
        if box_width <= 0 or box_height <= 0:
            messagebox.showerror("错误", "标注框尺寸无效，无法学习布局")
            return
        
        # 计算每个元素相对于整体框的位置（归一化）
        layout = {}
        for ann in anns:
            element_class = ann['class']
            x1_ratio = (ann['x1'] - min_x) / box_width
            y1_ratio = (ann['y1'] - min_y) / box_height
            x2_ratio = (ann['x2'] - min_x) / box_width
            y2_ratio = (ann['y2'] - min_y) / box_height
            
            layout[element_class] = {
                'x1_ratio': x1_ratio,
                'y1_ratio': y1_ratio,
                'x2_ratio': x2_ratio,
                'y2_ratio': y2_ratio
            }
            
            print(f"  {element_class}: ({x1_ratio:.3f}, {y1_ratio:.3f}) -> ({x2_ratio:.3f}, {y2_ratio:.3f})")
        
        # 保存配置
        self.save_layout_config(self.current_category, layout)
        print(f"✓ 布局配置已保存到: {self.layout_config_file}")
        print(f"✓ 配置内容: {len(layout)} 个元素")
        
        messagebox.showinfo("成功", f"已学习并保存 {self.current_category} 的布局配置\n包含 {len(layout)} 个元素")
    
    def auto_learn_layout(self):
        """自动学习布局（每次整体标记后或调整后自动触发）"""
        print(f"\n>>> auto_learn_layout() 被调用")
        print(f">>> 当前类别: {self.current_category}")
        print(f">>> 当前图片索引: {self.current_index}")
        
        if not self.current_category or not self.current_images:
            print(f">>> 跳过: 没有类别或图片")
            return
        
        image_path = str(self.current_images[self.current_index])
        print(f">>> 图片路径: {image_path}")
        
        if image_path not in self.annotations or not self.annotations[image_path]:
            print(f">>> 跳过: 图片没有标注")
            return
        
        # 只有多个标注时才学习布局
        anns = self.annotations[image_path]
        print(f">>> 标注数量: {len(anns)}")
        
        if len(anns) < 2:
            print(f">>> 跳过: 标注数量少于2个")
            return
        
        # 获取图片尺寸
        img_width, img_height = self.current_image.size
        print(f">>> 图片尺寸: {img_width} x {img_height}")
        
        # 找到整体边界框
        min_x = min(ann['x1'] for ann in anns)
        min_y = min(ann['y1'] for ann in anns)
        max_x = max(ann['x2'] for ann in anns)
        max_y = max(ann['y2'] for ann in anns)
        
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        print(f">>> 整体边界框: ({min_x:.1f}, {min_y:.1f}) -> ({max_x:.1f}, {max_y:.1f})")
        print(f">>> 边界框尺寸: {box_width:.1f} x {box_height:.1f}")
        
        # 检查边界框尺寸
        if box_width <= 0 or box_height <= 0:
            print(f">>> 跳过: 边界框尺寸无效")
            return
        
        # 计算每个元素相对于整体框的位置（归一化）
        layout = {}
        for ann in anns:
            element_class = ann['class']
            x1_ratio = (ann['x1'] - min_x) / box_width
            y1_ratio = (ann['y1'] - min_y) / box_height
            x2_ratio = (ann['x2'] - min_x) / box_width
            y2_ratio = (ann['y2'] - min_y) / box_height
            
            layout[element_class] = {
                'x1_ratio': x1_ratio,
                'y1_ratio': y1_ratio,
                'x2_ratio': x2_ratio,
                'y2_ratio': y2_ratio
            }
            
            print(f">>>   {element_class}: ({x1_ratio:.3f}, {y1_ratio:.3f}) -> ({x2_ratio:.3f}, {y2_ratio:.3f})")
        
        # 保存配置（静默保存，不弹窗）
        self.save_layout_config(self.current_category, layout)
        print(f">>> ✓ 自动记录布局: {self.current_category} ({len(layout)} 个元素)")
    
    def create_widgets(self):
        """创建界面"""
        # 左侧面板
        left_frame = tk.Frame(self.root, width=250, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)
        
        # 类别选择
        tk.Label(left_frame, text="选择页面类别", font=('微软雅黑', 12, 'bold'), bg='#f0f0f0').pack(pady=10)
        
        self.category_listbox = tk.Listbox(left_frame, font=('微软雅黑', 10), height=15)
        self.category_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.category_listbox.bind('<<ListboxSelect>>', self.on_category_select)
        
        # 统计信息
        self.stats_label = tk.Label(left_frame, text="", font=('微软雅黑', 9), bg='#f0f0f0', justify=tk.LEFT)
        self.stats_label.pack(pady=10)
        
        # 元素类别选择
        tk.Label(left_frame, text="元素类别", font=('微软雅黑', 10, 'bold'), bg='#f0f0f0').pack(pady=(20, 5))
        
        self.element_var = tk.StringVar(value=ELEMENT_CLASSES[0])
        self.element_combo = ttk.Combobox(left_frame, textvariable=self.element_var, 
                                         values=ELEMENT_CLASSES, state='readonly', font=('微软雅黑', 9))
        self.element_combo.pack(padx=10, pady=5, fill=tk.X)
        
        # 快捷键提示
        help_text = """
快捷键:
• 拖动: 框选 | P: 画笔
• Ctrl+拖: 调整 | C: 复制
• 空格: 下一张 | Del: 删除
• Ctrl+S: 保存 | H: 显/隐
• B: 整体 | A: 调整
• 1-9: 快选类别

画笔模式:
• 按住左键拖动绘制
• 松开鼠标完成
• 自动生成边界框
        """
        tk.Label(left_frame, text=help_text, font=('微软雅黑', 8), 
                bg='#f0f0f0', justify=tk.LEFT, fg='#666').pack(pady=10)
        
        # 初始化变量（在右侧面板使用）
        self.show_saved_var = tk.BooleanVar(value=True)
        self.filter_class_var = tk.BooleanVar(value=False)
        self.batch_mode_var = tk.BooleanVar(value=False)
        self.adjust_mode_var = tk.BooleanVar(value=False)

        
        # 右侧面板
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 顶部信息栏
        info_frame = tk.Frame(right_frame, bg='#e0e0e0', height=50)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        info_frame.pack_propagate(False)
        
        self.info_label = tk.Label(info_frame, text="请选择页面类别开始标注", 
                                   font=('微软雅黑', 11), bg='#e0e0e0')
        self.info_label.pack(pady=15)
        
        # 图片显示区域
        canvas_frame = tk.Frame(right_frame, bg='#333')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='#333', cursor='cross')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标事件
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Button-3>', self.on_right_click)  # 右键下一张或完成多边形
        self.canvas.bind('<Double-Button-1>', self.on_double_click)  # 双击完成多边形
        self.canvas.bind('<Motion>', self.on_mouse_move)  # 鼠标移动（用于调整模式）
        
        # 底部控制栏
        control_frame = tk.Frame(right_frame, bg='#f0f0f0', height=120)
        control_frame.pack(fill=tk.X, pady=(5, 0))
        control_frame.pack_propagate(False)
        
        # 复选框区域（第一行）
        checkbox_frame = tk.Frame(control_frame, bg='#f0f0f0')
        checkbox_frame.pack(pady=(10, 5))
        
        # 多边形模式
        self.polygon_mode_var = tk.BooleanVar(value=False)
        tk.Checkbutton(checkbox_frame, text="🖌️ 画笔 (P)", variable=self.polygon_mode_var,
                      command=self.toggle_polygon_mode,
                      font=('微软雅黑', 9, 'bold'),
                      bg='#f0f0f0', fg='#ff6600').pack(side=tk.LEFT, padx=8)
        
        # 整体标记模式
        tk.Checkbutton(checkbox_frame, text="📦 整体 (B)", variable=self.batch_mode_var,
                      font=('微软雅黑', 9, 'bold'),
                      bg='#f0f0f0', fg='#ff0000').pack(side=tk.LEFT, padx=8)
        
        # 调整模式
        tk.Checkbutton(checkbox_frame, text="✏️ 调整 (A)", variable=self.adjust_mode_var,
                      command=self.toggle_adjust_mode,
                      font=('微软雅黑', 9, 'bold'),
                      bg='#f0f0f0', fg='#0000ff').pack(side=tk.LEFT, padx=8)
        
        # 显示已保存
        tk.Checkbutton(checkbox_frame, text="👁 显示已保存", variable=self.show_saved_var,
                      command=self.toggle_display, font=('微软雅黑', 9),
                      bg='#f0f0f0').pack(side=tk.LEFT, padx=8)
        
        # 过滤当前类别
        tk.Checkbutton(checkbox_frame, text="🔍 过滤类别", variable=self.filter_class_var,
                      command=self.toggle_display, font=('微软雅黑', 9),
                      bg='#f0f0f0').pack(side=tk.LEFT, padx=8)
        
        # 按钮区域（第二行）
        btn_frame = tk.Frame(control_frame, bg='#f0f0f0')
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="⬅ 上一张", command=self.prev_image, 
                 font=('微软雅黑', 10), width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="下一张 ➡", command=self.next_image, 
                 font=('微软雅黑', 10), width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="🗑 删除最后", command=self.delete_last, 
                 font=('微软雅黑', 10), width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="💾 保存", command=self.save_annotations, 
                 font=('微软雅黑', 10), width=10, bg='#4CAF50', fg='white').pack(side=tk.LEFT, padx=5)
        
        # 绑定快捷键
        self.root.bind('<space>', lambda e: self.next_image())
        self.root.bind('<BackSpace>', lambda e: self.prev_image())
        self.root.bind('<Delete>', lambda e: self.delete_last())
        self.root.bind('<Control-s>', lambda e: self.save_annotations())
        self.root.bind('h', lambda e: self.toggle_show_saved())
        self.root.bind('p', lambda e: self.toggle_polygon_mode_key())  # P键切换多边形模式
        self.root.bind('H', lambda e: self.toggle_show_saved())
        self.root.bind('f', lambda e: self.toggle_filter())
        self.root.bind('F', lambda e: self.toggle_filter())
        self.root.bind('b', lambda e: self.toggle_batch_mode())
        self.root.bind('B', lambda e: self.toggle_batch_mode())
        self.root.bind('a', lambda e: self.toggle_adjust_mode())
        self.root.bind('A', lambda e: self.toggle_adjust_mode())
        self.root.bind('c', lambda e: self.copy_from_previous())
        self.root.bind('C', lambda e: self.copy_from_previous())
        
        # 数字键快速选择类别
        for i in range(1, 10):
            if i <= len(ELEMENT_CLASSES):
                self.root.bind(str(i), lambda e, idx=i-1: self.quick_select_class(idx))
    
    def load_categories(self):
        """加载页面类别"""
        if not self.data_dir.exists():
            messagebox.showerror("错误", f"数据目录不存在: {self.data_dir}")
            return
        
        categories = []
        for item in sorted(self.data_dir.iterdir()):
            if item.is_dir():
                png_count = len(list(item.glob("*.png")))
                if png_count > 0:
                    categories.append(item.name)
        
        for cat in categories:
            self.category_listbox.insert(tk.END, cat)
    
    def on_category_select(self, event):
        """选择类别"""
        selection = self.category_listbox.curselection()
        if not selection:
            return
        
        self.current_category = self.category_listbox.get(selection[0])
        self.load_images()
    
    def load_images(self):
        """加载图片列表"""
        category_dir = self.data_dir / self.current_category
        self.current_images = sorted(list(category_dir.glob("*.png")))
        
        if not self.current_images:
            messagebox.showinfo("提示", f"{self.current_category} 没有图片")
            return
        
        self.current_index = 0
        self.load_annotations()
        self.show_image()
        self.update_stats()
    
    def load_annotations(self):
        """加载已有标注"""
        annotation_file = self.data_dir / self.current_category / "annotations.json"
        if annotation_file.exists():
            with open(annotation_file, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}
    
    def show_image(self):
        """显示当前图片"""
        if not self.current_images:
            return
        
        image_path = self.current_images[self.current_index]
        
        # 加载图片
        self.current_image = Image.open(image_path)
        
        # 调整大小以适应画布
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # 计算缩放比例
            img_width, img_height = self.current_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            display_image = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            display_image = self.current_image
        
        self.current_photo = ImageTk.PhotoImage(display_image)
        
        # 清空画布
        self.canvas.delete('all')
        
        # 显示图片
        self.canvas.create_image(
            canvas_width // 2 if canvas_width > 1 else 0,
            canvas_height // 2 if canvas_height > 1 else 0,
            image=self.current_photo,
            anchor=tk.CENTER
        )
        
        # 总是显示当前图片的标注（黄色）
        self.draw_current_annotations()
        
        # 更新信息（显示标注的元素类型和数量）
        image_path_str = str(image_path)
        annotations = self.annotations.get(image_path_str, [])
        
        if annotations:
            # 统计各类元素的数量
            class_counts = Counter([ann['class'] for ann in annotations])
            # 格式化为 "元素1×2, 元素2×3" 的形式
            ann_detail = ", ".join([f"{cls}×{count}" for cls, count in sorted(class_counts.items())])
            info_text = f"{self.current_category} - {self.current_index + 1}/{len(self.current_images)} - {image_path.name} - {ann_detail}"
        else:
            info_text = f"{self.current_category} - {self.current_index + 1}/{len(self.current_images)} - {image_path.name} - 未标注"
        
        self.info_label.config(text=info_text)
    
    def draw_current_annotations(self):
        """绘制当前图片的标注（黄色）"""
        image_path = str(self.current_images[self.current_index])
        if image_path not in self.annotations:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.current_image.size
        
        # 计算缩放和偏移
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        display_width = int(img_width * scale)
        display_height = int(img_height * scale)
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2
        
        # 获取当前选择的类别（用于过滤）
        current_class = self.element_var.get()
        filter_enabled = self.filter_class_var.get()
        
        # 绘制标注框（黄色）
        for idx, ann in enumerate(self.annotations[image_path]):
            # 如果启用过滤，只显示当前类别
            if filter_enabled and ann['class'] != current_class:
                continue
            
            x1 = ann['x1'] * scale + offset_x
            y1 = ann['y1'] * scale + offset_y
            x2 = ann['x2'] * scale + offset_x
            y2 = ann['y2'] * scale + offset_y
            
            # 判断是否是选中的标注框（在调整模式下）
            is_selected = self.adjust_mode_var.get() and idx == self.selected_annotation
            
            # 判断是否有多边形数据
            has_polygon = 'polygon' in ann and ann['polygon']
            
            # 绘制矩形框或多边形
            outline_color = '#00ff00' if is_selected else ('#ff6600' if has_polygon else '#ffff00')  # 选中时绿色，多边形橙色，否则黄色
            line_width = 3 if is_selected else 2
            
            if has_polygon:
                # 绘制多边形
                canvas_points = []
                for px, py in ann['polygon']:
                    canvas_x = px * scale + offset_x
                    canvas_y = py * scale + offset_y
                    canvas_points.extend([canvas_x, canvas_y])
                
                self.canvas.create_polygon(
                    canvas_points,
                    outline=outline_color,
                    width=line_width,
                    fill=''
                )
                
                # 绘制多边形的点
                for px, py in ann['polygon']:
                    canvas_x = px * scale + offset_x
                    canvas_y = py * scale + offset_y
                    self.canvas.create_oval(
                        canvas_x - 3, canvas_y - 3,
                        canvas_x + 3, canvas_y + 3,
                        fill=outline_color,
                        outline='white',
                        width=1
                    )
            else:
                # 绘制矩形框
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline=outline_color,
                    width=line_width
                )
            
            # 如果是选中的标注框，绘制调整句柄
            if is_selected:
                handle_size = 6
                # 四个角点
                self.canvas.create_rectangle(
                    x1 - handle_size, y1 - handle_size, x1 + handle_size, y1 + handle_size,
                    fill='#00ff00', outline='white', width=1
                )
                self.canvas.create_rectangle(
                    x2 - handle_size, y1 - handle_size, x2 + handle_size, y1 + handle_size,
                    fill='#00ff00', outline='white', width=1
                )
                self.canvas.create_rectangle(
                    x1 - handle_size, y2 - handle_size, x1 + handle_size, y2 + handle_size,
                    fill='#00ff00', outline='white', width=1
                )
                self.canvas.create_rectangle(
                    x2 - handle_size, y2 - handle_size, x2 + handle_size, y2 + handle_size,
                    fill='#00ff00', outline='white', width=1
                )
                
                # 四条边的中点
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                self.canvas.create_rectangle(
                    x1 - handle_size, mid_y - handle_size, x1 + handle_size, mid_y + handle_size,
                    fill='#00ff00', outline='white', width=1
                )
                self.canvas.create_rectangle(
                    x2 - handle_size, mid_y - handle_size, x2 + handle_size, mid_y + handle_size,
                    fill='#00ff00', outline='white', width=1
                )
                self.canvas.create_rectangle(
                    mid_x - handle_size, y1 - handle_size, mid_x + handle_size, y1 + handle_size,
                    fill='#00ff00', outline='white', width=1
                )
                self.canvas.create_rectangle(
                    mid_x - handle_size, y2 - handle_size, mid_x + handle_size, y2 + handle_size,
                    fill='#00ff00', outline='white', width=1
                )
            
            # 绘制类别标签（使用支持中文的字体）
            try:
                from tkinter import font
                # 尝试使用系统中文字体
                label_font = font.Font(family='Microsoft YaHei', size=10, weight='bold')
            except:
                # 如果失败，使用默认字体
                label_font = ('Arial', 10, 'bold')
            
            label_color = '#00ff00' if is_selected else '#ffff00'
            self.canvas.create_text(
                x1, y1 - 5,
                text=ann['class'],
                fill=label_color,
                anchor=tk.SW,
                font=label_font
            )
    
    def draw_saved_annotations(self):
        """绘制已保存的标注（绿色 - 已保存）"""
        # 这个方法用于显示从文件加载的标注
        # 目前我们的标注都在内存中，所以这个方法暂时不需要实现
        # 如果需要区分"已保存"和"未保存"，可以在这里实现
        pass
    
    def draw_annotations(self):
        """绘制已有标注（保留用于兼容）"""
        # 这个方法现在由 draw_current_annotations 替代
        pass
    
    def toggle_adjust_mode(self):
        """切换调整模式"""
        is_adjust = self.adjust_mode_var.get()
        if is_adjust:
            self.canvas.config(cursor='hand2')
            print("✓ 进入调整模式 - 可以拖动和调整标注框")
        else:
            self.canvas.config(cursor='cross')
            self.selected_annotation = None
            self.show_image()  # 刷新显示
            print("✓ 退出调整模式")
    
    def toggle_polygon_mode(self):
        """切换多边形模式"""
        is_polygon = self.polygon_mode_var.get()
        if is_polygon:
            self.canvas.config(cursor='pencil')
            self.polygon_points = []
            self.temp_polygon = None
            self.is_drawing = False
            print("✓ 进入画笔模式 - 按住鼠标左键拖动绘制轮廓")
            print("  提示：松开鼠标完成绘制，会自动生成边界框")
        else:
            self.canvas.config(cursor='cross')
            self.polygon_points = []
            self.is_drawing = False
            if self.temp_polygon:
                self.canvas.delete(self.temp_polygon)
                self.temp_polygon = None
            self.show_image()  # 刷新显示
            print("✓ 退出画笔模式")
    
    def toggle_polygon_mode_key(self):
        """通过快捷键切换多边形模式"""
        current = self.polygon_mode_var.get()
        self.polygon_mode_var.set(not current)
        self.toggle_polygon_mode()
    
    def add_polygon_point(self, event):
        """添加多边形点（画笔模式：鼠标按下开始绘制）"""
        if not self.current_images or self.current_index >= len(self.current_images):
            return
        
        # 开始绘制
        self.is_drawing = True
        self.polygon_points = []
        
        # 计算坐标转换参数
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.current_image.size
        
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        display_width = int(img_width * scale)
        display_height = int(img_height * scale)
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2
        
        # 转换为图片坐标
        img_x = (event.x - offset_x) / scale
        img_y = (event.y - offset_y) / scale
        
        # 限制在图片范围内
        img_x = max(0, min(img_width, img_x))
        img_y = max(0, min(img_height, img_y))
        
        # 添加第一个点
        self.polygon_points.append((img_x, img_y))
        print(f"开始绘制轮廓...")
    
    def continue_drawing(self, event):
        """继续绘制（画笔模式：鼠标拖动）"""
        if not self.is_drawing or not self.polygon_points:
            return
        
        # 计算坐标转换参数
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.current_image.size
        
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        display_width = int(img_width * scale)
        display_height = int(img_height * scale)
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2
        
        # 转换为图片坐标
        img_x = (event.x - offset_x) / scale
        img_y = (event.y - offset_y) / scale
        
        # 限制在图片范围内
        img_x = max(0, min(img_width, img_x))
        img_y = max(0, min(img_height, img_y))
        
        # 添加点（每隔几个像素添加一个点，避免点太密集）
        last_x, last_y = self.polygon_points[-1]
        distance = ((img_x - last_x) ** 2 + (img_y - last_y) ** 2) ** 0.5
        
        if distance > 3:  # 距离大于3像素才添加新点
            self.polygon_points.append((img_x, img_y))
            self.draw_temp_polygon()
    
    def draw_temp_polygon(self):
        """绘制临时多边形（画笔轨迹）"""
        if len(self.polygon_points) < 2:
            return
        
        # 删除旧的临时多边形
        if self.temp_polygon:
            for item in self.temp_polygon:
                self.canvas.delete(item)
        
        self.temp_polygon = []
        
        # 计算坐标转换参数
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.current_image.size
        
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        display_width = int(img_width * scale)
        display_height = int(img_height * scale)
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2
        
        # 绘制线条连接所有点
        for i in range(len(self.polygon_points) - 1):
            x1, y1 = self.polygon_points[i]
            x2, y2 = self.polygon_points[i + 1]
            
            canvas_x1 = x1 * scale + offset_x
            canvas_y1 = y1 * scale + offset_y
            canvas_x2 = x2 * scale + offset_x
            canvas_y2 = y2 * scale + offset_y
            
            line = self.canvas.create_line(
                canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                fill='#ff6600',
                width=3,
                tags='temp_polygon'
            )
            self.temp_polygon.append(line)
        
        # 绘制起点和终点
        if self.polygon_points:
            # 起点（绿色）
            x, y = self.polygon_points[0]
            canvas_x = x * scale + offset_x
            canvas_y = y * scale + offset_y
            circle = self.canvas.create_oval(
                canvas_x - 5, canvas_y - 5,
                canvas_x + 5, canvas_y + 5,
                fill='#00ff00',
                outline='white',
                width=2,
                tags='temp_polygon'
            )
            self.temp_polygon.append(circle)
            
            # 终点（红色）
            x, y = self.polygon_points[-1]
            canvas_x = x * scale + offset_x
            canvas_y = y * scale + offset_y
            circle = self.canvas.create_oval(
                canvas_x - 4, canvas_y - 4,
                canvas_x + 4, canvas_y + 4,
                fill='#ff0000',
                outline='white',
                width=1,
                tags='temp_polygon'
            )
            self.temp_polygon.append(circle)
    
    def finish_polygon(self):
        """完成多边形标注"""
        if len(self.polygon_points) < 3:
            print("⚠ 多边形至少需要3个点")
            return
        
        # 计算多边形的边界框
        xs = [p[0] for p in self.polygon_points]
        ys = [p[1] for p in self.polygon_points]
        
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)
        
        # 添加标注（使用边界框）
        image_path = str(self.current_images[self.current_index])
        if image_path not in self.annotations:
            self.annotations[image_path] = []
        
        element_class = self.element_var.get()
        annotation = {
            'class': element_class,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'polygon': self.polygon_points.copy()  # 保存多边形点
        }
        
        self.annotations[image_path].append(annotation)
        print(f"✓ 添加多边形标注: {element_class} (边界框: {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        print(f"  多边形点数: {len(self.polygon_points)}")
        
        # 清空多边形点
        self.polygon_points = []
        if self.temp_polygon:
            self.canvas.delete(self.temp_polygon)
            self.temp_polygon = None
        
        # 刷新显示
        self.show_image()
        self.update_info()
    
    def on_mouse_move(self, event):
        """鼠标移动 - 用于调整模式下的交互"""
        # 检查是否按住 Ctrl 键或启用了调整模式
        ctrl_pressed = (event.state & 0x0004) != 0  # Ctrl 键的状态位
        is_adjust_mode = self.adjust_mode_var.get() or ctrl_pressed
        
        if not is_adjust_mode:
            return
        
        if not self.current_images or self.current_index >= len(self.current_images):
            return
        
        image_path = str(self.current_images[self.current_index])
        if image_path not in self.annotations or not self.annotations[image_path]:
            return
        
        # 计算坐标转换参数
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.current_image.size
        
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        display_width = int(img_width * scale)
        display_height = int(img_height * scale)
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2
        
        # 检查鼠标是否在某个标注框上
        mouse_x = event.x
        mouse_y = event.y
        
        found_annotation = False
        for idx, ann in enumerate(self.annotations[image_path]):
            x1 = ann['x1'] * scale + offset_x
            y1 = ann['y1'] * scale + offset_y
            x2 = ann['x2'] * scale + offset_x
            y2 = ann['y2'] * scale + offset_y
            
            # 检查是否在边缘（用于调整大小）
            edge_threshold = 8
            on_left = abs(mouse_x - x1) < edge_threshold and y1 - edge_threshold < mouse_y < y2 + edge_threshold
            on_right = abs(mouse_x - x2) < edge_threshold and y1 - edge_threshold < mouse_y < y2 + edge_threshold
            on_top = abs(mouse_y - y1) < edge_threshold and x1 - edge_threshold < mouse_x < x2 + edge_threshold
            on_bottom = abs(mouse_y - y2) < edge_threshold and x1 - edge_threshold < mouse_x < x2 + edge_threshold
            
            # 检查是否在角点
            on_tl = abs(mouse_x - x1) < edge_threshold and abs(mouse_y - y1) < edge_threshold
            on_tr = abs(mouse_x - x2) < edge_threshold and abs(mouse_y - y1) < edge_threshold
            on_bl = abs(mouse_x - x1) < edge_threshold and abs(mouse_y - y2) < edge_threshold
            on_br = abs(mouse_x - x2) < edge_threshold and abs(mouse_y - y2) < edge_threshold
            
            # 检查是否在框内（用于拖动）
            in_box = x1 < mouse_x < x2 and y1 < mouse_y < y2
            
            if on_tl or on_tr or on_bl or on_br or on_left or on_right or on_top or on_bottom:
                # 在边缘或角点，显示调整大小光标
                if on_tl or on_br:
                    self.canvas.config(cursor='size_nw_se')
                elif on_tr or on_bl:
                    self.canvas.config(cursor='size_ne_sw')
                elif on_left or on_right:
                    self.canvas.config(cursor='size_we')
                elif on_top or on_bottom:
                    self.canvas.config(cursor='size_ns')
                found_annotation = True
                break
            elif in_box:
                # 在框内，显示移动光标
                self.canvas.config(cursor='fleur')
                found_annotation = True
                break
        
        if not found_annotation:
            if ctrl_pressed:
                self.canvas.config(cursor='hand2')
            else:
                self.canvas.config(cursor='cross')
    
    def on_mouse_down(self, event):
        """鼠标按下"""
        # 多边形模式
        if self.polygon_mode_var.get():
            self.add_polygon_point(event)
            return
        
        # 检查是否按住 Ctrl 键或启用了调整模式
        ctrl_pressed = (event.state & 0x0004) != 0  # Ctrl 键的状态位
        is_adjust_mode = self.adjust_mode_var.get() or ctrl_pressed
        
        # 调整模式
        if is_adjust_mode:
            if not self.current_images or self.current_index >= len(self.current_images):
                return
            
            image_path = str(self.current_images[self.current_index])
            if image_path not in self.annotations or not self.annotations[image_path]:
                return
            
            # 计算坐标转换参数
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_width, img_height = self.current_image.size
            
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            display_width = int(img_width * scale)
            display_height = int(img_height * scale)
            offset_x = (canvas_width - display_width) // 2
            offset_y = (canvas_height - display_height) // 2
            
            mouse_x = event.x
            mouse_y = event.y
            
            # 查找点击的标注框
            for idx, ann in enumerate(self.annotations[image_path]):
                x1 = ann['x1'] * scale + offset_x
                y1 = ann['y1'] * scale + offset_y
                x2 = ann['x2'] * scale + offset_x
                y2 = ann['y2'] * scale + offset_y
                
                # 检查边缘和角点
                edge_threshold = 8
                on_left = abs(mouse_x - x1) < edge_threshold and y1 - edge_threshold < mouse_y < y2 + edge_threshold
                on_right = abs(mouse_x - x2) < edge_threshold and y1 - edge_threshold < mouse_y < y2 + edge_threshold
                on_top = abs(mouse_y - y1) < edge_threshold and x1 - edge_threshold < mouse_x < x2 + edge_threshold
                on_bottom = abs(mouse_y - y2) < edge_threshold and x1 - edge_threshold < mouse_x < x2 + edge_threshold
                
                on_tl = abs(mouse_x - x1) < edge_threshold and abs(mouse_y - y1) < edge_threshold
                on_tr = abs(mouse_x - x2) < edge_threshold and abs(mouse_y - y1) < edge_threshold
                on_bl = abs(mouse_x - x1) < edge_threshold and abs(mouse_y - y2) < edge_threshold
                on_br = abs(mouse_x - x2) < edge_threshold and abs(mouse_y - y2) < edge_threshold
                
                in_box = x1 < mouse_x < x2 and y1 < mouse_y < y2
                
                if on_tl:
                    self.selected_annotation = idx
                    self.resize_handle = 'tl'
                    self.drag_start_x = mouse_x
                    self.drag_start_y = mouse_y
                    self.show_image()  # 刷新显示，高亮选中的框
                    return
                elif on_tr:
                    self.selected_annotation = idx
                    self.resize_handle = 'tr'
                    self.drag_start_x = mouse_x
                    self.drag_start_y = mouse_y
                    self.show_image()
                    return
                elif on_bl:
                    self.selected_annotation = idx
                    self.resize_handle = 'bl'
                    self.drag_start_x = mouse_x
                    self.drag_start_y = mouse_y
                    self.show_image()
                    return
                elif on_br:
                    self.selected_annotation = idx
                    self.resize_handle = 'br'
                    self.drag_start_x = mouse_x
                    self.drag_start_y = mouse_y
                    self.show_image()
                    return
                elif on_left:
                    self.selected_annotation = idx
                    self.resize_handle = 'left'
                    self.drag_start_x = mouse_x
                    self.drag_start_y = mouse_y
                    self.show_image()
                    return
                elif on_right:
                    self.selected_annotation = idx
                    self.resize_handle = 'right'
                    self.drag_start_x = mouse_x
                    self.drag_start_y = mouse_y
                    self.show_image()
                    return
                elif on_top:
                    self.selected_annotation = idx
                    self.resize_handle = 'top'
                    self.drag_start_x = mouse_x
                    self.drag_start_y = mouse_y
                    self.show_image()
                    return
                elif on_bottom:
                    self.selected_annotation = idx
                    self.resize_handle = 'bottom'
                    self.drag_start_x = mouse_x
                    self.drag_start_y = mouse_y
                    self.show_image()
                    return
                elif in_box:
                    self.selected_annotation = idx
                    self.dragging_annotation = idx
                    self.drag_start_x = mouse_x
                    self.drag_start_y = mouse_y
                    self.show_image()
                    return
            
            return
        
        # 标注模式
        self.start_x = event.x
        self.start_y = event.y
    
    def on_mouse_drag(self, event):
        """鼠标拖动"""
        # 画笔模式：继续绘制
        if self.polygon_mode_var.get() and self.is_drawing:
            self.continue_drawing(event)
            return
        
        # 检查是否按住 Ctrl 键或启用了调整模式
        ctrl_pressed = (event.state & 0x0004) != 0  # Ctrl 键的状态位
        is_adjust_mode = self.adjust_mode_var.get() or ctrl_pressed
        
        # 调整模式
        if is_adjust_mode:
            if self.selected_annotation is None:
                return
            
            if not self.current_images or self.current_index >= len(self.current_images):
                return
            
            image_path = str(self.current_images[self.current_index])
            if image_path not in self.annotations:
                return
            
            # 计算坐标转换参数
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_width, img_height = self.current_image.size
            
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            display_width = int(img_width * scale)
            display_height = int(img_height * scale)
            offset_x = (canvas_width - display_width) // 2
            offset_y = (canvas_height - display_height) // 2
            
            # 计算鼠标移动距离（原始图片坐标）
            dx = (event.x - self.drag_start_x) / scale
            dy = (event.y - self.drag_start_y) / scale
            
            ann = self.annotations[image_path][self.selected_annotation]
            
            # 拖动整个框
            if self.dragging_annotation is not None:
                ann['x1'] += dx
                ann['y1'] += dy
                ann['x2'] += dx
                ann['y2'] += dy
                
                # 限制在图片范围内
                ann['x1'] = max(0, min(ann['x1'], img_width))
                ann['y1'] = max(0, min(ann['y1'], img_height))
                ann['x2'] = max(0, min(ann['x2'], img_width))
                ann['y2'] = max(0, min(ann['y2'], img_height))
            
            # 调整大小
            elif self.resize_handle:
                if self.resize_handle == 'tl':
                    ann['x1'] += dx
                    ann['y1'] += dy
                elif self.resize_handle == 'tr':
                    ann['x2'] += dx
                    ann['y1'] += dy
                elif self.resize_handle == 'bl':
                    ann['x1'] += dx
                    ann['y2'] += dy
                elif self.resize_handle == 'br':
                    ann['x2'] += dx
                    ann['y2'] += dy
                elif self.resize_handle == 'left':
                    ann['x1'] += dx
                elif self.resize_handle == 'right':
                    ann['x2'] += dx
                elif self.resize_handle == 'top':
                    ann['y1'] += dy
                elif self.resize_handle == 'bottom':
                    ann['y2'] += dy
                
                # 确保x1 < x2, y1 < y2
                if ann['x1'] > ann['x2']:
                    ann['x1'], ann['x2'] = ann['x2'], ann['x1']
                if ann['y1'] > ann['y2']:
                    ann['y1'], ann['y2'] = ann['y2'], ann['y1']
                
                # 限制在图片范围内
                ann['x1'] = max(0, min(ann['x1'], img_width))
                ann['y1'] = max(0, min(ann['y1'], img_height))
                ann['x2'] = max(0, min(ann['x2'], img_width))
                ann['y2'] = max(0, min(ann['y2'], img_height))
            
            # 更新拖动起点
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            
            # 重新绘制
            self.show_image()
            return
        
        # 标注模式
        if self.start_x is None:
            return
        
        # 删除临时矩形
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
        
        # 绘制新矩形
        self.temp_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='#ff0000',
            width=2
        )
    
    def on_mouse_up(self, event):
        """鼠标释放"""
        # 画笔模式：完成绘制
        if self.polygon_mode_var.get() and self.is_drawing:
            self.is_drawing = False
            if len(self.polygon_points) >= 3:
                self.finish_polygon()
            else:
                print("⚠ 绘制的轮廓太短，至少需要3个点")
                self.polygon_points = []
                if self.temp_polygon:
                    self.canvas.delete(self.temp_polygon)
                    self.temp_polygon = None
            return
        
        # 检查是否按住 Ctrl 键或启用了调整模式
        ctrl_pressed = (event.state & 0x0004) != 0  # Ctrl 键的状态位
        is_adjust_mode = self.adjust_mode_var.get() or ctrl_pressed
        
        # 调整模式
        if is_adjust_mode:
            self.selected_annotation = None
            self.dragging_annotation = None
            self.resize_handle = None
            self.drag_start_x = None
            self.drag_start_y = None
            
            # 自动保存调整后的标注
            if self.current_images and self.current_index < len(self.current_images):
                self.auto_save_current()
                # 自动记录调整后的布局
                self.auto_learn_layout()
            
            return
        
        # 标注模式
        if self.start_x is None:
            return
        
        # 删除临时矩形
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None
        
        # 计算实际坐标
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.current_image.size
        
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        display_width = int(img_width * scale)
        display_height = int(img_height * scale)
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2
        
        # 转换为原始图片坐标
        x1 = max(0, min((self.start_x - offset_x) / scale, img_width))
        y1 = max(0, min((self.start_y - offset_y) / scale, img_height))
        x2 = max(0, min((event.x - offset_x) / scale, img_width))
        y2 = max(0, min((event.y - offset_y) / scale, img_height))
        
        # 确保x1 < x2, y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # 检查框的大小
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            self.start_x = None
            self.start_y = None
            return
        
        # 检查是否是整体标记模式
        if self.batch_mode_var.get():
            self.batch_annotate(x1, y1, x2, y2)
        else:
            # 保存单个标注
            image_path = str(self.current_images[self.current_index])
            if image_path not in self.annotations:
                self.annotations[image_path] = []
            
            self.annotations[image_path].append({
                'class': self.element_var.get(),
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
        
        # 重新绘制（显示新标注）
        self.show_image()
        
        # 更新统计信息
        self.update_stats()
        
        self.start_x = None
        self.start_y = None
    
    def on_right_click(self, event):
        """右键点击 - 下一张（画笔模式下无效）"""
        if self.polygon_mode_var.get():
            return  # 画笔模式下右键无效
        self.next_image()
    
    def on_double_click(self, event):
        """双击 - 无操作（画笔模式下无效）"""
        pass
    
    def delete_last(self):
        """删除最后一个标注"""
        image_path = str(self.current_images[self.current_index])
        if image_path in self.annotations and self.annotations[image_path]:
            self.annotations[image_path].pop()
            self.show_image()
            self.update_stats()  # 更新统计
    
    def copy_from_previous(self):
        """复制上一张图片的标注到当前图片（带自动微调）"""
        if not self.current_images or self.current_index == 0:
            print("✗ 无法复制：这是第一张图片")
            return
        
        # 获取上一张图片的路径
        prev_image_path = str(self.current_images[self.current_index - 1])
        current_image_path = str(self.current_images[self.current_index])
        
        # 检查上一张图片是否有标注
        if prev_image_path not in self.annotations or not self.annotations[prev_image_path]:
            print("✗ 无法复制：上一张图片没有标注")
            return
        
        # 复制标注（深拷贝）
        import copy
        import random
        self.annotations[current_image_path] = copy.deepcopy(self.annotations[prev_image_path])
        
        # 获取图片尺寸
        img_width, img_height = self.current_image.size
        
        print(f"\n=== 复制标注（自动微调） ===")
        print(f"从: {Path(prev_image_path).name}")
        print(f"到: {Path(current_image_path).name}")
        print(f"✓ 已复制 {len(self.annotations[current_image_path])} 个标注")
        
        # 自动微调每个标注框
        for ann in self.annotations[current_image_path]:
            # 计算框的尺寸
            box_width = ann['x2'] - ann['x1']
            box_height = ann['y2'] - ann['y1']
            
            # 随机偏移量：±2-5像素（根据框的大小调整）
            # 小框偏移少一点，大框偏移多一点
            max_offset_x = min(5, box_width * 0.05)  # 最多偏移框宽度的5%
            max_offset_y = min(5, box_height * 0.05)  # 最多偏移框高度的5%
            
            offset_x = random.uniform(-max_offset_x, max_offset_x)
            offset_y = random.uniform(-max_offset_y, max_offset_y)
            
            # 随机缩放：±1-3像素（让框的大小也有微小变化）
            scale_offset = random.uniform(-2, 2)
            
            # 应用偏移和缩放
            ann['x1'] += offset_x - scale_offset / 2
            ann['y1'] += offset_y - scale_offset / 2
            ann['x2'] += offset_x + scale_offset / 2
            ann['y2'] += offset_y + scale_offset / 2
            
            # 限制在图片范围内
            ann['x1'] = max(0, min(ann['x1'], img_width))
            ann['y1'] = max(0, min(ann['y1'], img_height))
            ann['x2'] = max(0, min(ann['x2'], img_width))
            ann['y2'] = max(0, min(ann['y2'], img_height))
            
            # 确保框的大小有效
            if ann['x2'] <= ann['x1']:
                ann['x2'] = ann['x1'] + box_width
            if ann['y2'] <= ann['y1']:
                ann['y2'] = ann['y1'] + box_height
            
            print(f"  - {ann['class']}: ({ann['x1']:.1f}, {ann['y1']:.1f}) -> ({ann['x2']:.1f}, {ann['y2']:.1f}) [偏移: {offset_x:.1f}, {offset_y:.1f}]")
        
        print(f"✓ 自动微调完成（随机偏移 ±{max_offset_x:.1f}px）")
        
        # 刷新显示
        self.show_image()
        self.update_stats()
    
    def next_image(self):
        """下一张图片"""
        # 自动保存当前图片的标注
        self.auto_save_current()
        
        if self.current_index < len(self.current_images) - 1:
            self.current_index += 1
            self.show_image()
            self.update_stats()
    
    def prev_image(self):
        """上一张图片"""
        # 自动保存当前图片的标注
        self.auto_save_current()
        
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()
            self.update_stats()
    
    def auto_save_current(self):
        """自动保存当前图片的标注"""
        if not self.current_category or not self.current_images:
            return
        
        # 保存JSON格式
        annotation_file = self.data_dir / self.current_category / "annotations.json"
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        
        # 只保存当前图片的YOLO格式（提高性能）
        current_image_path = str(self.current_images[self.current_index])
        if current_image_path in self.annotations:
            self.save_yolo_format_single(current_image_path)
    
    def quick_select_class(self, index):
        """快速选择类别"""
        if index < len(ELEMENT_CLASSES):
            self.element_var.set(ELEMENT_CLASSES[index])
            # 如果启用了过滤，重新显示图片
            if self.filter_class_var.get():
                self.show_image()
    
    def toggle_show_saved(self):
        """切换显示已保存标注"""
        self.show_saved_var.set(not self.show_saved_var.get())
        if self.current_images:
            self.show_image()
    
    def toggle_filter(self):
        """切换过滤当前类别"""
        self.filter_class_var.set(not self.filter_class_var.get())
        if self.current_images:
            self.show_image()
    
    def toggle_batch_mode(self):
        """切换整体标记模式"""
        self.batch_mode_var.set(not self.batch_mode_var.get())
        mode_text = "整体标记模式" if self.batch_mode_var.get() else "单个标记模式"
        print(f"切换到: {mode_text}")
    
    def batch_annotate(self, box_x1, box_y1, box_x2, box_y2):
        """整体标记 - 根据当前类别自动标记多个元素"""
        image_path = str(self.current_images[self.current_index])
        if image_path not in self.annotations:
            self.annotations[image_path] = []
        
        # 计算框的尺寸
        box_width = box_x2 - box_x1
        box_height = box_y2 - box_y1
        
        # 检查是否有保存的布局配置
        category = self.current_category
        
        print(f"\n=== 整体标记 ===")
        print(f"类别: {category}")
        print(f"框选区域: ({box_x1:.1f}, {box_y1:.1f}) -> ({box_x2:.1f}, {box_y2:.1f})")
        print(f"框选尺寸: {box_width:.1f} x {box_height:.1f}")
        print(f"已加载的布局配置: {list(self.layout_configs.keys())}")
        
        if category in self.layout_configs:
            # 使用上一次的布局
            layout = self.layout_configs[category]
            print(f"✓ 使用上一次的布局配置（包含 {len(layout)} 个元素）")
            
            for element_class, ratios in layout.items():
                x1 = box_x1 + box_width * ratios['x1_ratio']
                y1 = box_y1 + box_height * ratios['y1_ratio']
                x2 = box_x1 + box_width * ratios['x2_ratio']
                y2 = box_y1 + box_height * ratios['y2_ratio']
                
                self.annotations[image_path].append({
                    'class': element_class,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })
                
                print(f"  - {element_class}: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
            
            print(f"✓ 已标记 {len(layout)} 个元素（使用上一次的布局）")
        
        elif category == "转账确认弹窗":
            # 使用默认布局（转账确认弹窗）
            print(f"✓ 使用默认布局（转账确认弹窗）")
            
            # ID区域（顶部，约占20%高度）
            id_y1 = box_y1 + box_height * 0.05
            id_y2 = box_y1 + box_height * 0.25
            self.annotations[image_path].append({
                'class': '转账确认ID',
                'x1': box_x1 + box_width * 0.15,
                'y1': id_y1,
                'x2': box_x2 - box_width * 0.15,
                'y2': id_y2
            })
            
            # 金额区域（中间，约占30%高度）
            amount_y1 = box_y1 + box_height * 0.30
            amount_y2 = box_y1 + box_height * 0.55
            self.annotations[image_path].append({
                'class': '转账确认金额',
                'x1': box_x1 + box_width * 0.20,
                'y1': amount_y1,
                'x2': box_x2 - box_width * 0.20,
                'y2': amount_y2
            })
            
            # 确认按钮（底部，约占15%高度）
            button_y1 = box_y1 + box_height * 0.75
            button_y2 = box_y1 + box_height * 0.92
            self.annotations[image_path].append({
                'class': '确认按钮',
                'x1': box_x1 + box_width * 0.08,
                'y1': button_y1,
                'x2': box_x2 - box_width * 0.08,
                'y2': button_y2
            })
            
            print(f"✓ 已标记转账确认弹窗的3个元素（使用默认布局）")
        
        else:
            # 其他类别暂时使用单个标注
            print(f"✓ 使用单个标注模式")
            self.annotations[image_path].append({
                'class': self.element_var.get(),
                'x1': box_x1,
                'y1': box_y1,
                'x2': box_x2,
                'y2': box_y2
            })
            print(f"✓ 已标记1个元素")
        
        # 不要在整体标记后自动记录布局
        # 因为自动标记的位置可能不准确
        # 只在手动调整后才记录布局
    
    def toggle_display(self):
        """切换显示选项"""
        if self.current_images:
            self.show_image()
    
    def toggle_saved_annotations(self):
        """切换显示已保存的标注"""
        self.show_saved_annotations = self.show_saved_var.get()
        if self.current_images:
            self.show_image()
    
    def save_annotations(self):
        """保存标注"""
        if not self.current_category:
            return
        
        annotation_file = self.data_dir / self.current_category / "annotations.json"
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        
        # 同时保存YOLO格式
        self.save_yolo_format()
        
        # 更新统计信息（不弹窗，方便连续标注）
        self.update_stats()
        
        # 在信息栏显示保存成功
        current_text = self.info_label.cget('text')
        self.info_label.config(text=f"{current_text} - ✓ 已保存")
        
        # 1秒后刷新显示（使用show_image来显示完整的标注信息）
        self.root.after(1000, self.show_image)
    
    def save_yolo_format_single(self, image_path):
        """保存单张图片的YOLO格式标注"""
        if image_path not in self.annotations or not self.annotations[image_path]:
            return
        
        # 创建类别映射
        class_to_id = {cls: idx for idx, cls in enumerate(ELEMENT_CLASSES)}
        
        # 获取图片尺寸
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # 生成YOLO格式标注
        yolo_lines = []
        for ann in self.annotations[image_path]:
            class_id = class_to_id.get(ann['class'], 0)
            
            # 计算中心点和宽高(归一化)
            center_x = ((ann['x1'] + ann['x2']) / 2) / img_width
            center_y = ((ann['y1'] + ann['y2']) / 2) / img_height
            width = (ann['x2'] - ann['x1']) / img_width
            height = (ann['y2'] - ann['y1']) / img_height
            
            yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        # 保存到txt文件
        txt_path = Path(image_path).with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
    
    def save_yolo_format(self):
        """保存所有图片的YOLO格式标注（用于手动保存）"""
        # 创建类别映射
        class_to_id = {cls: idx for idx, cls in enumerate(ELEMENT_CLASSES)}
        
        for image_path, anns in self.annotations.items():
            if not anns:
                continue
            
            # 获取图片尺寸
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # 生成YOLO格式标注
            yolo_lines = []
            for ann in anns:
                class_id = class_to_id.get(ann['class'], 0)
                
                # 计算中心点和宽高(归一化)
                center_x = ((ann['x1'] + ann['x2']) / 2) / img_width
                center_y = ((ann['y1'] + ann['y2']) / 2) / img_height
                width = (ann['x2'] - ann['x1']) / img_width
                height = (ann['y2'] - ann['y1']) / img_height
                
                yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            
            # 保存到txt文件
            txt_path = Path(image_path).with_suffix('.txt')
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
    
    def update_stats(self):
        """更新统计信息"""
        if not self.current_category:
            return
        
        total = len(self.current_images)
        annotated = sum(1 for img in self.current_images if str(img) in self.annotations)
        
        stats_text = f"总图片: {total}\n已标注: {annotated}\n未标注: {total - annotated}"
        self.stats_label.config(text=stats_text)


def main():
    """主函数"""
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
