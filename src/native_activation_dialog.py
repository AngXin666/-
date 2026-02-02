"""
原生激活对话框（使用 HTA - HTML Application）
不依赖 tkinter，使用 Windows 原生技术
"""

import tempfile
import subprocess
import os
from pathlib import Path


class NativeActivationDialog:
    """原生激活对话框（HTA 实现）"""
    
    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self.result = False
        self.license_key = ""
        
    def show(self) -> tuple[bool, str]:
        """显示激活对话框
        
        Returns:
            (是否成功, 卡密)
        """
        # 创建 HTA 文件
        hta_content = self._create_hta_content()
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.hta', 
            delete=False, 
            encoding='utf-8'
        ) as f:
            hta_file = f.name
            f.write(hta_content)
        
        try:
            # 运行 HTA 对话框
            result = subprocess.run(
                ['mshta.exe', hta_file],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            # 读取结果
            if result.returncode == 0 and result.stdout.strip():
                output = result.stdout.strip()
                if output.startswith("SUCCESS:"):
                    self.license_key = output.replace("SUCCESS:", "").strip()
                    self.result = True
                    return True, self.license_key
            
            return False, ""
            
        finally:
            # 清理临时文件
            try:
                os.unlink(hta_file)
            except:
                pass
    
    def _create_hta_content(self) -> str:
        """创建 HTA 内容（浅色主题 - 最终版本）"""
        return f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>激活卡密</title>
    <HTA:APPLICATION
        BORDER="dialog"
        BORDERSTYLE="normal"
        CAPTION="yes"
        MAXIMIZEBUTTON="no"
        MINIMIZEBUTTON="no"
        SHOWINTASKBAR="yes"
        SYSMENU="yes"
        SCROLL="no"
    />
    <style>
        body {{
            font-family: "Microsoft YaHei UI", Arial, sans-serif;
            background: #f5f7fa;
            margin: 0;
            overflow: hidden;
        }}
        
        .container {{
            padding: 15px 30px;
        }}
        
        h2 {{
            font-size: 22px;
            font-weight: 300;
            color: #2c3e50;
            margin: 0 0 6px 0;
        }}
        
        .subtitle {{
            font-size: 13px;
            color: #7f8c8d;
            margin-bottom: 20px;
        }}
        
        .form-group {{
            margin-bottom: 15px;
        }}
        
        label {{
            display: block;
            margin-bottom: 6px;
            font-size: 13px;
            color: #5a6c7d;
        }}
        
        input[type="text"] {{
            padding: 10px 12px;
            border: 2px solid #d1d9e0;
            border-radius: 8px;
            font-size: 14px;
            font-family: Consolas, "Courier New", monospace;
            background: #ffffff;
            color: #2c3e50;
            box-sizing: border-box;
        }}
        
        input[type="text"]:focus {{
            outline: none;
            border-color: #3498db;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        
        td {{
            vertical-align: middle;
            padding: 0;
        }}
        
        .device-cell {{
            padding: 8px 12px;
            background: #f8f9fa;
            border: 2px solid #d1d9e0;
            border-radius: 8px;
            font-size: 11px;
            font-family: Consolas, "Courier New", monospace;
            color: #5a6c7d;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        .btn-cell {{
            width: 100px;
            padding-left: 10px;
        }}
        
        button {{
            width: 100%;
            padding: 10px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
        }}
        
        button:hover {{
            background: #2980b9;
        }}
        
        .hint {{
            font-size: 11px;
            color: #95a5a6;
            margin-top: 6px;
        }}
        
        .btn-primary {{
            padding: 12px;
            background: #3498db;
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 15px;
            cursor: pointer;
            margin-top: 6px;
            box-sizing: border-box;
        }}
        
        .full-width {{
            width: 100%;
        }}
        
        .license-input {{
            width: 95%;
        }}
        
        .btn-primary:hover {{
            background: #2980b9;
        }}
    </style>
</head>
<body>
    <div class="container">
    <h2>激活卡密</h2>
    <div class="subtitle">请输入您的卡密进行激活</div>
    
    <div class="form-group">
        <label>卡密:</label>
        <input type="text" id="licenseKey" placeholder="KIRO-XXXX-XXXX-XXXX-XXXX" class="license-input" />
    </div>
    
    <div class="form-group">
        <label>设备ID (机器码):</label>
        <table>
            <tr>
                <td class="device-cell">{self.machine_id}</td>
                <td class="btn-cell"><button onclick="copyMachineId()">复制</button></td>
            </tr>
        </table>
        <div class="hint">此设备ID将与卡密绑定</div>
    </div>
    
    <button class="btn-primary full-width" onclick="activate()">激活</button>
    </div>
    
    <script>
        window.resizeTo(700, 400);
        window.moveTo((screen.width - 700) / 2, (screen.height - 400) / 2);
        
        function copyMachineId() {{
            var machineId = "{self.machine_id}";
            var tempInput = document.createElement('input');
            tempInput.value = machineId;
            document.body.appendChild(tempInput);
            tempInput.select();
            document.execCommand('copy');
            document.body.removeChild(tempInput);
            
            alert('已复制设备ID');
        }}
        
        function activate() {{
            var licenseKey = document.getElementById('licenseKey').value.trim();
            if (!licenseKey) {{
                alert('请输入卡密');
                return;
            }}
            
            // 输出结果到标准输出
            var fso = new ActiveXObject("Scripting.FileSystemObject");
            var stdout = fso.GetStandardStream(1);
            stdout.WriteLine("SUCCESS:" + licenseKey);
            
            window.close();
        }}
        
        setTimeout(function() {{
            document.getElementById('licenseKey').focus();
        }}, 100);
    </script>
</body>
</html>
'''
