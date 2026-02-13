"""
转账历史GUI模块
Transfer History GUI Module
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
from typing import Optional


class TransferHistoryGUI:
    """转账历史GUI窗口"""
    
    def __init__(self, parent=None):
        """初始化转账历史GUI
        
        Args:
            parent: 父窗口（可选）
        """
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self.window.title("转账历史记录")
        self.window.geometry("1000x700")
        
        # 先隐藏窗口，避免白屏
        self.window.withdraw()
        
        # 导入转账历史管理器
        from .transfer_history import get_transfer_history
        self.history_manager = get_transfer_history()
        
        self._create_widgets()
        self._load_data()
        
        # 所有内容准备完成后再显示窗口
        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()
    
    def _create_widgets(self):
        """创建界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 创建筛选区域
        self._create_filter_section(main_frame)
        
        # 创建统计信息区域
        self._create_statistics_section(main_frame)
        
        # 创建记录列表区域
        self._create_records_section(main_frame)
        
        # 创建按钮区域
        self._create_button_section(main_frame)
    
    def _create_filter_section(self, parent):
        """创建筛选区域"""
        filter_frame = ttk.LabelFrame(parent, text="筛选条件", padding="5")
        filter_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 发送人筛选
        ttk.Label(filter_frame, text="发送人:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.sender_var = tk.StringVar()
        sender_entry = ttk.Entry(filter_frame, textvariable=self.sender_var, width=15)
        sender_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # 收款人筛选
        ttk.Label(filter_frame, text="收款人:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.recipient_var = tk.StringVar()
        recipient_entry = ttk.Entry(filter_frame, textvariable=self.recipient_var, width=15)
        recipient_entry.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # 管理员筛选（改为下拉选择框）
        ttk.Label(filter_frame, text="管理员:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.owner_var = tk.StringVar(value="全部")
        owner_combo = ttk.Combobox(filter_frame, textvariable=self.owner_var, width=13, state='readonly')
        
        # 获取管理员列表
        owner_list = ["全部"]
        try:
            from .user_manager import UserManager
            user_manager = UserManager()
            users = user_manager.get_all_users()
            owner_list.extend([user.user_name for user in users if user.enabled])
        except Exception as e:
            print(f"获取管理员列表失败: {e}")
        
        owner_combo['values'] = owner_list
        owner_combo.grid(row=0, column=5, sticky=tk.W, padx=5)
        
        # 日期范围筛选
        ttk.Label(filter_frame, text="日期范围:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.days_var = tk.IntVar(value=30)
        days_combo = ttk.Combobox(filter_frame, textvariable=self.days_var, width=12, state='readonly')
        days_combo['values'] = (7, 30, 90, 180, 365)
        days_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(filter_frame, text="天").grid(row=1, column=2, sticky=tk.W)
        
        # 筛选按钮
        filter_btn = ttk.Button(filter_frame, text="应用筛选", command=self._apply_filter)
        filter_btn.grid(row=1, column=3, padx=5, pady=5)
        
        # 重置按钮
        reset_btn = ttk.Button(filter_frame, text="重置", command=self._reset_filter)
        reset_btn.grid(row=1, column=4, padx=5, pady=5)
    
    def _create_statistics_section(self, parent):
        """创建统计信息区域"""
        stats_frame = ttk.LabelFrame(parent, text="统计信息", padding="5")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 统计标签
        self.stats_label = ttk.Label(stats_frame, text="加载中...", justify=tk.LEFT)
        self.stats_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    
    def _create_records_section(self, parent):
        """创建记录列表区域"""
        records_frame = ttk.LabelFrame(parent, text="转账记录", padding="5")
        records_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        records_frame.columnconfigure(0, weight=1)
        records_frame.rowconfigure(0, weight=1)
        
        # 创建Treeview
        columns = ('时间', '发送人', '收款人', '金额', '策略', '状态', '管理员')
        self.tree = ttk.Treeview(records_frame, columns=columns, show='headings', height=15)
        
        # 设置列标题
        self.tree.heading('时间', text='时间')
        self.tree.heading('发送人', text='发送人')
        self.tree.heading('收款人', text='收款人')
        self.tree.heading('金额', text='金额(元)')
        self.tree.heading('策略', text='选择策略')
        self.tree.heading('状态', text='状态')
        self.tree.heading('管理员', text='管理员')
        
        # 设置列宽
        self.tree.column('时间', width=150)
        self.tree.column('发送人', width=120)
        self.tree.column('收款人', width=120)
        self.tree.column('金额', width=80)
        self.tree.column('策略', width=80)
        self.tree.column('状态', width=60)
        self.tree.column('管理员', width=100)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(records_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 绑定双击事件
        self.tree.bind('<Double-1>', self._on_record_double_click)
    
    def _create_button_section(self, parent):
        """创建按钮区域"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, sticky=tk.E, pady=(0, 5))
        
        # 刷新按钮
        refresh_btn = ttk.Button(button_frame, text="刷新", command=self._load_data)
        refresh_btn.grid(row=0, column=0, padx=5)
        
        # 导出按钮
        export_btn = ttk.Button(button_frame, text="导出CSV", command=self._export_csv)
        export_btn.grid(row=0, column=1, padx=5)
        
        # 关闭按钮
        close_btn = ttk.Button(button_frame, text="关闭", command=self.window.destroy)
        close_btn.grid(row=0, column=2, padx=5)
    
    def _load_data(self):
        """加载数据"""
        # 清空现有数据
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 获取筛选条件
        sender = self.sender_var.get().strip() or None
        recipient = self.recipient_var.get().strip() or None
        owner = self.owner_var.get().strip()
        # 如果选择"全部"，则不筛选管理员
        if owner == "全部":
            owner = None
        days = self.days_var.get()
        
        # 计算日期范围
        end_date = datetime.now().isoformat()
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # 获取记录
        records = self.history_manager.get_transfer_records(
            sender_phone=sender,
            recipient_phone=recipient,
            owner=owner,
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
        
        # 填充数据
        for record in records:
            # 格式化时间
            try:
                dt = datetime.fromisoformat(record.timestamp)
                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                time_str = record.timestamp
            
            # 格式化状态
            status = '成功' if record.success else '失败'
            
            # 格式化策略
            strategy_map = {
                'rotation': '轮询',
                'random': '随机'
            }
            strategy = strategy_map.get(record.strategy, record.strategy)
            
            # 插入数据
            self.tree.insert('', tk.END, values=(
                time_str,
                f"{record.sender_name} ({record.sender_phone})",
                f"{record.recipient_name} ({record.recipient_phone})",
                f"{record.amount:.2f}",
                strategy,
                status,
                record.owner or '-'
            ))
        
        # 更新统计信息
        self._update_statistics(sender, recipient, owner, days)
    
    def _update_statistics(self, sender=None, recipient=None, owner=None, days=30):
        """更新统计信息"""
        stats = self.history_manager.get_transfer_statistics(
            sender_phone=sender,
            recipient_phone=recipient,
            owner=owner,
            days=days
        )
        
        # 格式化统计信息
        stats_text = (
            f"统计周期: 最近 {days} 天\n"
            f"总转账次数: {stats['total_count']} 次\n"
            f"成功次数: {stats['success_count']} 次\n"
            f"失败次数: {stats['failed_count']} 次\n"
            f"成功率: {stats['success_rate']:.1f}%\n"
            f"总金额: {stats['total_amount']:.2f} 元"
        )
        
        # 添加收款人统计
        if stats['recipient_stats']:
            stats_text += "\n\n收款人统计 (Top 5):"
            for i, recipient_stat in enumerate(stats['recipient_stats'][:5], 1):
                stats_text += (
                    f"\n{i}. {recipient_stat['name']}: "
                    f"{recipient_stat['count']}次, "
                    f"{recipient_stat['amount']:.2f}元"
                )
        
        self.stats_label.config(text=stats_text)
    
    def _apply_filter(self):
        """应用筛选"""
        self._load_data()
    
    def _reset_filter(self):
        """重置筛选"""
        self.sender_var.set('')
        self.recipient_var.set('')
        self.owner_var.set('全部')
        self.days_var.set(30)
        self._load_data()
    
    def _on_record_double_click(self, event):
        """双击记录事件"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        values = item['values']
        
        # 显示详细信息
        detail_text = (
            f"时间: {values[0]}\n"
            f"发送人: {values[1]}\n"
            f"收款人: {values[2]}\n"
            f"金额: {values[3]} 元\n"
            f"选择策略: {values[4]}\n"
            f"状态: {values[5]}\n"
            f"管理员: {values[6]}"
        )
        
        messagebox.showinfo("转账详情", detail_text)
    
    def _export_csv(self):
        """导出CSV"""
        from tkinter import filedialog
        import csv
        
        # 选择保存路径
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # 获取所有记录
            records = []
            for item in self.tree.get_children():
                records.append(self.tree.item(item)['values'])
            
            # 写入CSV
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(['时间', '发送人', '收款人', '金额(元)', '选择策略', '状态', '管理员'])
                # 写入数据
                writer.writerows(records)
            
            messagebox.showinfo("导出成功", f"已导出 {len(records)} 条记录到:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("导出失败", f"导出CSV失败:\n{e}")
    
    def show(self):
        """显示窗口"""
        self.window.mainloop()


def show_transfer_history(parent=None):
    """显示转账历史窗口
    
    Args:
        parent: 父窗口（可选）
    """
    gui = TransferHistoryGUI(parent)
    if not parent:
        gui.show()
