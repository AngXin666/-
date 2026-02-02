"""
自定义滑动开关组件
Custom Toggle Switch Widget
"""

import tkinter as tk
from tkinter import Canvas


class ToggleSwitch(Canvas):
    """滑动开关组件"""
    
    def __init__(self, parent, width=60, height=30, command=None, **kwargs):
        """初始化滑动开关
        
        Args:
            parent: 父组件
            width: 开关宽度
            height: 开关高度
            command: 状态改变时的回调函数
        """
        super().__init__(parent, width=width, height=height, 
                        highlightthickness=0, **kwargs)
        
        self.width = width
        self.height = height
        self.command = command
        self.state = False  # False=关闭, True=开启
        
        # 颜色配置
        self.bg_on = "#4CAF50"  # 开启时的背景色（绿色）
        self.bg_off = "#CCCCCC"  # 关闭时的背景色（灰色）
        self.circle_color = "#FFFFFF"  # 圆形滑块颜色（白色）
        
        # 绘制开关
        self._draw()
        
        # 绑定点击事件
        self.bind("<Button-1>", self._on_click)
    
    def _draw(self):
        """绘制开关"""
        self.delete("all")
        
        # 绘制背景（圆角矩形）
        bg_color = self.bg_on if self.state else self.bg_off
        radius = self.height / 2
        
        # 绘制圆角矩形背景
        self.create_oval(0, 0, self.height, self.height, 
                        fill=bg_color, outline="")
        self.create_oval(self.width - self.height, 0, 
                        self.width, self.height, 
                        fill=bg_color, outline="")
        self.create_rectangle(radius, 0, 
                             self.width - radius, self.height, 
                             fill=bg_color, outline="")
        
        # 绘制滑块（圆形）
        circle_x = self.width - radius if self.state else radius
        circle_radius = radius * 0.8
        self.create_oval(circle_x - circle_radius, 
                        radius - circle_radius,
                        circle_x + circle_radius, 
                        radius + circle_radius,
                        fill=self.circle_color, outline="")
        
        # 绘制文字
        if self.state:
            self.create_text(radius + 5, radius, 
                           text="ON", fill="white", 
                           font=("Arial", 10, "bold"))
        else:
            self.create_text(self.width - radius - 5, radius, 
                           text="OFF", fill="gray", 
                           font=("Arial", 10, "bold"))
    
    def _on_click(self, event):
        """点击事件处理"""
        self.toggle()
    
    def toggle(self):
        """切换开关状态"""
        self.state = not self.state
        self._draw()
        
        # 调用回调函数
        if self.command:
            self.command(self.state)
    
    def set_state(self, state: bool):
        """设置开关状态
        
        Args:
            state: True=开启, False=关闭
        """
        if self.state != state:
            self.state = state
            self._draw()
    
    def get_state(self) -> bool:
        """获取当前状态
        
        Returns:
            bool: True=开启, False=关闭
        """
        return self.state
