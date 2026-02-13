"""可视化学习器数据 - 查看推荐的最佳位置并生成标注图片"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from button_position_learner import ButtonPositionLearner
from ocr_region_learner import OCRRegionLearner
from pathlib import Path
import json

# 创建可视化输出目录
output_dir = Path("learning_visualization")
output_dir.mkdir(exist_ok=True)

def visualize_button_positions():
    """可视化按钮位置学习数据"""
    print("=" * 60)
    print("按钮位置学习器 - 推荐的最佳位置")
    print("=" * 60)
    
    learner = ButtonPositionLearner()
    
    # 读取全局数据
    global_file = Path("runtime_data/button_positions/global.json")
    if not global_file.exists():
        print("⚠️ 没有找到学习数据")
        return
    
    with open(global_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    
    for button_name in data.keys():
        print(f"\n【{button_name}】")
        
        # 获取最佳位置
        best_pos = learner.get_best_position(button_name, min_samples=5)
        if best_pos:
            print(f"  推荐位置: ({best_pos[0]}, {best_pos[1]})")
            results[button_name] = {
                'type': 'button',
                'best_position': best_pos
            }
        else:
            print(f"  ⚠️ 样本不足，无法推荐")
            continue
        
        # 获取统计信息
        stats = learner.get_statistics(button_name)
        if stats:
            print(f"  数据来源: {stats['data_source']}")
            print(f"  样本数量: {stats['sample_count']}")
            print(f"  X坐标: 均值={stats['x_mean']:.1f}, 中位数={stats['x_median']:.1f}, 标准差={stats['x_stdev']:.2f}")
            print(f"  Y坐标: 均值={stats['y_mean']:.1f}, 中位数={stats['y_median']:.1f}, 标准差={stats['y_stdev']:.2f}")
            
            results[button_name]['statistics'] = stats
        
        # 获取合理范围
        default_range = (0, 720, 0, 1280)  # 假设屏幕尺寸
        valid_range = learner.get_valid_range(button_name, default_range)
        print(f"  合理范围: X=[{valid_range[0]}, {valid_range[1]}], Y=[{valid_range[2]}, {valid_range[3]}]")
        
        results[button_name]['valid_range'] = valid_range
    
    return results

def visualize_ocr_regions():
    """可视化OCR区域学习数据"""
    print("\n" + "=" * 60)
    print("OCR区域学习器 - 推荐的最佳区域")
    print("=" * 60)
    
    learner = OCRRegionLearner()
    
    # 读取全局数据
    global_file = Path("runtime_data/ocr_regions/global.json")
    if not global_file.exists():
        print("⚠️ 没有找到学习数据")
        return
    
    with open(global_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    
    for region_name in data.keys():
        print(f"\n【{region_name}】")
        
        # 获取最佳区域
        best_region = learner.get_best_region(region_name, min_samples=5)
        if best_region:
            print(f"  推荐区域: x={best_region[0]}, y={best_region[1]}, w={best_region[2]}, h={best_region[3]}")
            results[region_name] = {
                'type': 'region',
                'best_region': best_region
            }
        else:
            print(f"  ⚠️ 样本不足，无法推荐")
            continue
        
        # 获取统计信息
        stats = learner.get_statistics(region_name)
        if stats:
            print(f"  数据来源: {stats['data_source']}")
            print(f"  样本数量: {stats['sample_count']}")
            print(f"  X坐标: 均值={stats['x_mean']:.1f}, 中位数={stats['x_median']:.1f}, 标准差={stats['x_stdev']:.2f}")
            print(f"  Y坐标: 均值={stats['y_mean']:.1f}, 中位数={stats['y_median']:.1f}, 标准差={stats['y_stdev']:.2f}")
            print(f"  宽度: 均值={stats['width_mean']:.1f}, 中位数={stats['width_median']:.1f}, 标准差={stats['width_stdev']:.2f}")
            print(f"  高度: 均值={stats['height_mean']:.1f}, 中位数={stats['height_median']:.1f}, 标准差={stats['height_stdev']:.2f}")
            
            results[region_name]['statistics'] = stats
        
        # 获取合理范围
        default_range = (0, 720, 0, 1280, 50, 500, 10, 100)  # 假设屏幕尺寸和区域大小
        valid_range = learner.get_valid_range(region_name, default_range)
        print(f"  合理范围:")
        print(f"    X=[{valid_range[0]}, {valid_range[1]}]")
        print(f"    Y=[{valid_range[2]}, {valid_range[3]}]")
        print(f"    宽度=[{valid_range[4]}, {valid_range[5]}]")
        print(f"    高度=[{valid_range[6]}, {valid_range[7]}]")
        
        results[region_name]['valid_range'] = valid_range
    
    return results

def generate_visualization_html(button_results, ocr_results):
    """生成HTML可视化页面"""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>学习器数据可视化</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 8px;
        }
        .item {
            background: white;
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .item h3 {
            color: #2196F3;
            margin-top: 0;
        }
        .canvas-container {
            margin: 20px 0;
            border: 2px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            background: #fff;
        }
        canvas {
            display: block;
            margin: 0 auto;
        }
        .info {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin: 10px 0;
        }
        .info-item {
            background: #f9f9f9;
            padding: 8px;
            border-radius: 4px;
            border-left: 3px solid #4CAF50;
        }
        .label {
            font-weight: bold;
            color: #666;
        }
        .value {
            color: #333;
        }
    </style>
</head>
<body>
    <h1>🎯 学习器数据可视化</h1>
    <p>基于实际运行数据的统计学习结果</p>
"""
    
    # 按钮位置可视化
    if button_results:
        html_content += """
    <h2>📍 按钮位置学习数据</h2>
"""
        for button_name, data in button_results.items():
            best_pos = data['best_position']
            stats = data.get('statistics', {})
            valid_range = data.get('valid_range', (0, 720, 0, 1280))
            
            html_content += f"""
    <div class="item">
        <h3>{button_name}</h3>
        <div class="canvas-container">
            <canvas id="btn_{button_name}" width="720" height="400"></canvas>
        </div>
        <div class="info">
            <div class="info-item">
                <span class="label">推荐位置:</span>
                <span class="value">({best_pos[0]}, {best_pos[1]})</span>
            </div>
            <div class="info-item">
                <span class="label">样本数量:</span>
                <span class="value">{stats.get('sample_count', 0)}</span>
            </div>
            <div class="info-item">
                <span class="label">X坐标统计:</span>
                <span class="value">均值={stats.get('x_mean', 0):.1f}, 标准差={stats.get('x_stdev', 0):.2f}</span>
            </div>
            <div class="info-item">
                <span class="label">Y坐标统计:</span>
                <span class="value">均值={stats.get('y_mean', 0):.1f}, 标准差={stats.get('y_stdev', 0):.2f}</span>
            </div>
        </div>
    </div>
    <script>
        (function() {{
            const canvas = document.getElementById('btn_{button_name}');
            const ctx = canvas.getContext('2d');
            
            // 绘制背景
            ctx.fillStyle = '#f0f0f0';
            ctx.fillRect(0, 0, 720, 400);
            
            // 绘制合理范围
            ctx.fillStyle = 'rgba(76, 175, 80, 0.1)';
            ctx.fillRect({valid_range[0]}, {valid_range[2]}, 
                        {valid_range[1] - valid_range[0]}, {valid_range[3] - valid_range[2]});
            ctx.strokeStyle = 'rgba(76, 175, 80, 0.5)';
            ctx.lineWidth = 2;
            ctx.strokeRect({valid_range[0]}, {valid_range[2]}, 
                          {valid_range[1] - valid_range[0]}, {valid_range[3] - valid_range[2]});
            
            // 绘制推荐位置
            ctx.fillStyle = '#F44336';
            ctx.beginPath();
            ctx.arc({best_pos[0]}, {best_pos[1]}, 8, 0, 2 * Math.PI);
            ctx.fill();
            
            // 绘制标签
            ctx.fillStyle = '#333';
            ctx.font = '14px Arial';
            ctx.fillText('推荐位置: ({best_pos[0]}, {best_pos[1]})', {best_pos[0]} + 15, {best_pos[1]} - 10);
            
            // 绘制坐标轴
            ctx.strokeStyle = '#999';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(0, 400);
            ctx.moveTo(0, 400);
            ctx.lineTo(720, 400);
            ctx.stroke();
        }})();
    </script>
"""
    
    # OCR区域可视化
    if ocr_results:
        html_content += """
    <h2>📐 OCR区域学习数据</h2>
"""
        for region_name, data in ocr_results.items():
            best_region = data['best_region']
            stats = data.get('statistics', {})
            valid_range = data.get('valid_range', (0, 720, 0, 1280, 50, 500, 10, 100))
            
            html_content += f"""
    <div class="item">
        <h3>{region_name}</h3>
        <div class="canvas-container">
            <canvas id="ocr_{region_name}" width="720" height="400"></canvas>
        </div>
        <div class="info">
            <div class="info-item">
                <span class="label">推荐区域:</span>
                <span class="value">x={best_region[0]}, y={best_region[1]}, w={best_region[2]}, h={best_region[3]}</span>
            </div>
            <div class="info-item">
                <span class="label">样本数量:</span>
                <span class="value">{stats.get('sample_count', 0)}</span>
            </div>
            <div class="info-item">
                <span class="label">位置统计:</span>
                <span class="value">X均值={stats.get('x_mean', 0):.1f}, Y均值={stats.get('y_mean', 0):.1f}</span>
            </div>
            <div class="info-item">
                <span class="label">尺寸统计:</span>
                <span class="value">W均值={stats.get('width_mean', 0):.1f}, H均值={stats.get('height_mean', 0):.1f}</span>
            </div>
        </div>
    </div>
    <script>
        (function() {{
            const canvas = document.getElementById('ocr_{region_name}');
            const ctx = canvas.getContext('2d');
            
            // 绘制背景
            ctx.fillStyle = '#f0f0f0';
            ctx.fillRect(0, 0, 720, 400);
            
            // 绘制推荐区域
            ctx.fillStyle = 'rgba(33, 150, 243, 0.3)';
            ctx.fillRect({best_region[0]}, {best_region[1]}, {best_region[2]}, {best_region[3]});
            ctx.strokeStyle = '#2196F3';
            ctx.lineWidth = 2;
            ctx.strokeRect({best_region[0]}, {best_region[1]}, {best_region[2]}, {best_region[3]});
            
            // 绘制标签
            ctx.fillStyle = '#333';
            ctx.font = '12px Arial';
            ctx.fillText('推荐区域', {best_region[0]} + 5, {best_region[1]} + 15);
            ctx.fillText('({best_region[0]}, {best_region[1]}, {best_region[2]}, {best_region[3]})', 
                        {best_region[0]} + 5, {best_region[1]} + 30);
            
            // 绘制坐标轴
            ctx.strokeStyle = '#999';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(0, 400);
            ctx.moveTo(0, 400);
            ctx.lineTo(720, 400);
            ctx.stroke();
        }})();
    </script>
"""
    
    html_content += """
</body>
</html>
"""
    
    # 保存HTML文件
    html_file = output_dir / "visualization.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ 可视化页面已生成: {html_file}")
    return html_file

def main():
    print("\n学习器数据可视化工具")
    print("=" * 60)
    
    # 可视化按钮位置
    button_results = visualize_button_positions()
    
    # 可视化OCR区域
    ocr_results = visualize_ocr_regions()
    
    # 生成HTML可视化
    if button_results or ocr_results:
        html_file = generate_visualization_html(button_results or {}, ocr_results or {})
        
        # 打开文件夹
        print(f"\n正在打开可视化文件夹...")
        import subprocess
        subprocess.run(['explorer', str(output_dir.absolute())])
        
        print(f"\n✅ 完成！请在浏览器中打开 {html_file.name} 查看可视化结果")
    else:
        print("\n⚠️ 没有足够的学习数据生成可视化")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
