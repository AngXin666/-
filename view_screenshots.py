"""
查看截图并选择首页广告弹窗
"""
import os
from PIL import Image
import glob

def show_screenshots():
    """显示截图供用户选择"""
    screenshot_dirs = [
        'checkin_screenshots/20260211',
        'checkin_screenshots/20260210',
        'checkin_screenshots/20260209',
        'checkin_screenshots/20260208'
    ]
    
    all_screenshots = []
    
    for dir_path in screenshot_dirs:
        if not os.path.exists(dir_path):
            continue
            
        png_files = glob.glob(os.path.join(dir_path, '*.png'))
        
        for img_path in png_files:
            try:
                img = Image.open(img_path)
                width, height = img.size
                
                # 只显示540x960的截图
                if width == 540 and height == 960:
                    all_screenshots.append(img_path)
                    
            except:
                continue
    
    print(f"找到 {len(all_screenshots)} 张截图")
    print("\n显示前20张截图，请查看并告诉我哪张是首页广告弹窗：")
    
    for i, path in enumerate(all_screenshots[:20], 1):
        print(f"{i}. {path}")
        try:
            img = Image.open(path)
            img.show()
            
            choice = input(f"\n这是首页广告弹窗吗？(y/n/q退出): ").strip().lower()
            
            if choice == 'y':
                print(f"\n✓ 已选择: {path}")
                return path
            elif choice == 'q':
                print("退出")
                return None
                
        except Exception as e:
            print(f"无法打开图片: {e}")
            continue
    
    return None

if __name__ == '__main__':
    result = show_screenshots()
    
    if result:
        print(f"\n请使用以下命令可视化点击位置：")
        print(f'python -c "from find_and_visualize_home_popup import visualize_click_position; visualize_click_position(\'{result}\')"')
