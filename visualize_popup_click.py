"""
可视化首页广告弹窗点击位置
显示点击位置与底部导航栏的关系
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 创建画布
fig, ax = plt.subplots(figsize=(6, 10))

# 屏幕尺寸 (540x960)
screen_width = 540
screen_height = 960

# 绘制屏幕边框
screen = patches.Rectangle((0, 0), screen_width, screen_height, 
                           linewidth=2, edgecolor='black', facecolor='white')
ax.add_patch(screen)

# 绘制底部导航栏区域 (y=850-960)
nav_bar = patches.Rectangle((0, 850), screen_width, 110, 
                            linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.3)
ax.add_patch(nav_bar)
ax.text(270, 905, '底部导航栏区域', ha='center', va='center', fontsize=10, weight='bold')

# 底部导航按钮
nav_buttons = [
    (90, 920, '首页'),
    (200, 920, '分类'),
    (330, 920, '购物车'),
    (450, 920, '我的')
]

for x, y, label in nav_buttons:
    # 绘制按钮点（半径30像素的圆形区域）
    button = patches.Circle((x, y), 30, linewidth=2, edgecolor='red', facecolor='pink', alpha=0.5)
    ax.add_patch(button)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, weight='bold')

# 绘制首页广告弹窗（假设位置）
popup = patches.Rectangle((70, 300), 400, 400, 
                          linewidth=2, edgecolor='orange', facecolor='lightyellow', alpha=0.3)
ax.add_patch(popup)
ax.text(270, 500, '首页广告弹窗\n(假设区域)', ha='center', va='center', fontsize=11, weight='bold')

# 当前点击位置 (270, 150)
current_click = patches.Circle((270, 150), 15, linewidth=2, edgecolor='green', facecolor='lightgreen')
ax.add_patch(current_click)
ax.text(270, 120, '当前点击位置\n(270, 150)', ha='center', va='center', fontsize=9, color='green', weight='bold')

# 旧的点击位置 (270, 200) 和 (270, 850)
old_click_1 = patches.Circle((270, 200), 15, linewidth=2, edgecolor='gray', facecolor='lightgray', linestyle='--')
ax.add_patch(old_click_1)
ax.text(370, 200, '旧位置1 (270, 200)', ha='left', va='center', fontsize=8, color='gray')

old_click_2 = patches.Circle((270, 850), 15, linewidth=2, edgecolor='red', facecolor='pink', linestyle='--')
ax.add_patch(old_click_2)
ax.text(370, 850, '旧位置2 (270, 850)\n距离导航栏太近!', ha='left', va='center', fontsize=8, color='red', weight='bold')

# 绘制距离线
ax.plot([270, 270], [850, 920], 'r--', linewidth=1)
ax.text(290, 885, '70px', ha='left', va='center', fontsize=8, color='red')

# 绘制分类按钮的影响范围
category_range = patches.Circle((200, 920), 50, linewidth=1, edgecolor='red', facecolor='none', linestyle=':')
ax.add_patch(category_range)
ax.text(200, 980, '分类按钮\n影响范围', ha='center', va='top', fontsize=8, color='red')

# 设置坐标轴
ax.set_xlim(-50, 590)
ax.set_ylim(-50, 1010)
ax.set_aspect('equal')
ax.invert_yaxis()  # Y轴反转，让(0,0)在左上角

# 添加网格
ax.grid(True, alpha=0.3)
ax.set_xlabel('X 坐标 (像素)', fontsize=10)
ax.set_ylabel('Y 坐标 (像素)', fontsize=10)
ax.set_title('首页广告弹窗点击位置分析\n屏幕尺寸: 540x960', fontsize=12, weight='bold')

# 添加图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='当前点击位置'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='旧点击位置', linestyle='--'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10, label='导航按钮'),
    patches.Patch(facecolor='lightblue', alpha=0.3, label='导航栏区域'),
    patches.Patch(facecolor='lightyellow', alpha=0.3, label='弹窗区域')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('popup_click_visualization.png', dpi=150, bbox_inches='tight')
print("✓ 可视化图已保存到: popup_click_visualization.png")
print("\n分析结果:")
print("=" * 60)
print("问题: 旧点击位置 (270, 850) 距离底部导航栏只有 70 像素")
print("风险: 点击时容易误触分类按钮 (200, 920)")
print("解决: 使用新位置 (270, 150) 在顶部，远离所有导航按钮")
print("=" * 60)

plt.show()
