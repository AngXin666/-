"""
代码冗余分析工具
用于识别项目中的代码冗余设计
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


class RedundancyAnalyzer:
    """代码冗余分析器"""
    
    def __init__(self, src_dir: str = "src"):
        self.src_dir = Path(src_dir)
        self.results = {
            'page_detection_calls': [],
            'hardcoded_timeouts': [],
            'error_handling_patterns': [],
            'resource_management': [],
            'logging_formats': []
        }
    
    def analyze_all(self):
        """执行所有分析"""
        print("=" * 80)
        print("代码冗余分析报告")
        print("=" * 80)
        print()
        
        self.analyze_page_detection_calls()
        self.analyze_hardcoded_timeouts()
        self.analyze_error_handling()
        self.analyze_resource_management()
        self.analyze_logging_formats()
        
        self.generate_report()
    
    def analyze_page_detection_calls(self):
        """分析重复的页面检测调用"""
        print("1. 分析重复的页面检测调用...")
        print("-" * 80)
        
        # 目标文件
        target_files = [
            'daily_checkin.py',
            'navigator.py',
            'balance_transfer.py'
        ]
        
        for filename in target_files:
            filepath = self.src_dir / filename
            if not filepath.exists():
                continue
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # 查找 detect_page 调用
            pattern = r'(await\s+.*\.detect_page\([^)]+\))'
            matches = re.finditer(pattern, content)
            
            calls = []
            for match in matches:
                # 找到匹配的行号
                pos = match.start()
                line_num = content[:pos].count('\n') + 1
                call_text = match.group(1)
                
                # 提取上下文（前后各2行）
                start_line = max(0, line_num - 3)
                end_line = min(len(lines), line_num + 2)
                context = '\n'.join(lines[start_line:end_line])
                
                calls.append({
                    'file': filename,
                    'line': line_num,
                    'call': call_text,
                    'context': context
                })
            
            if calls:
                self.results['page_detection_calls'].extend(calls)
                print(f"  {filename}: 发现 {len(calls)} 次页面检测调用")
        
        print()
    
    def analyze_hardcoded_timeouts(self):
        """分析硬编码的超时时间"""
        print("2. 分析硬编码的超时时间...")
        print("-" * 80)
        
        # 搜索所有Python文件
        timeout_patterns = [
            (r'asyncio\.sleep\((\d+\.?\d*)\)', 'asyncio.sleep'),
            (r'time\.sleep\((\d+\.?\d*)\)', 'time.sleep'),
            (r'timeout\s*=\s*(\d+\.?\d*)', 'timeout参数'),
            (r'await.*\(.*,\s*(\d+\.?\d*)\)', 'await调用'),
        ]
        
        for py_file in self.src_dir.rglob('*.py'):
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for pattern, pattern_name in timeout_patterns:
                matches = re.finditer(pattern, content)
                
                for match in matches:
                    pos = match.start()
                    line_num = content[:pos].count('\n') + 1
                    timeout_value = match.group(1) if match.groups() else match.group(0)
                    
                    # 提取该行内容
                    line_content = lines[line_num - 1].strip()
                    
                    self.results['hardcoded_timeouts'].append({
                        'file': py_file.relative_to(self.src_dir),
                        'line': line_num,
                        'pattern': pattern_name,
                        'value': timeout_value,
                        'code': line_content
                    })
        
        print(f"  发现 {len(self.results['hardcoded_timeouts'])} 处硬编码超时")
        print()
    
    def analyze_error_handling(self):
        """分析重复的错误处理模式"""
        print("3. 分析重复的错误处理模式...")
        print("-" * 80)
        
        # 查找 try-except 块
        pattern = r'try:\s*\n(.*?)\n\s*except\s+(\w+).*?:\s*\n(.*?)(?=\n\s*(?:except|finally|else|\S))'
        
        for py_file in self.src_dir.rglob('*.py'):
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            matches = re.finditer(pattern, content, re.DOTALL)
            
            for match in matches:
                pos = match.start()
                line_num = content[:pos].count('\n') + 1
                exception_type = match.group(2)
                handler_code = match.group(3).strip()
                
                # 识别常见的错误处理模式
                if 'log' in handler_code.lower() or 'print' in handler_code.lower():
                    pattern_type = '日志记录'
                elif 'return' in handler_code:
                    pattern_type = '返回默认值'
                elif 'raise' in handler_code:
                    pattern_type = '重新抛出'
                elif 'pass' in handler_code:
                    pattern_type = '忽略异常'
                else:
                    pattern_type = '其他'
                
                self.results['error_handling_patterns'].append({
                    'file': py_file.relative_to(self.src_dir),
                    'line': line_num,
                    'exception': exception_type,
                    'pattern': pattern_type,
                    'handler': handler_code[:100]  # 只保留前100个字符
                })
        
        print(f"  发现 {len(self.results['error_handling_patterns'])} 个错误处理块")
        print()
    
    def analyze_resource_management(self):
        """分析重复的资源管理代码"""
        print("4. 分析重复的资源管理代码...")
        print("-" * 80)
        
        # 查找资源管理模式
        patterns = [
            (r'conn\s*=.*connect\(', '数据库连接'),
            (r'open\([^)]+\)', '文件操作'),
            (r'Lock\(\)', '线程锁'),
            (r'\.close\(\)', '资源关闭'),
        ]
        
        for py_file in self.src_dir.rglob('*.py'):
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for pattern, resource_type in patterns:
                matches = re.finditer(pattern, content)
                
                for match in matches:
                    pos = match.start()
                    line_num = content[:pos].count('\n') + 1
                    line_content = lines[line_num - 1].strip()
                    
                    # 检查是否使用了上下文管理器
                    uses_context_manager = 'with ' in line_content
                    
                    self.results['resource_management'].append({
                        'file': py_file.relative_to(self.src_dir),
                        'line': line_num,
                        'type': resource_type,
                        'code': line_content,
                        'uses_context_manager': uses_context_manager
                    })
        
        print(f"  发现 {len(self.results['resource_management'])} 处资源管理代码")
        print()
    
    def analyze_logging_formats(self):
        """分析日志格式不统一问题"""
        print("5. 分析日志格式不统一问题...")
        print("-" * 80)
        
        # 查找日志调用
        patterns = [
            r'print\([^)]+\)',
            r'logger\.\w+\([^)]+\)',
            r'log\([^)]+\)',
            r'logging\.\w+\([^)]+\)',
        ]
        
        log_formats = defaultdict(list)
        
        for py_file in self.src_dir.rglob('*.py'):
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for pattern in patterns:
                matches = re.finditer(pattern, content)
                
                for match in matches:
                    pos = match.start()
                    line_num = content[:pos].count('\n') + 1
                    log_call = match.group(0)
                    
                    # 识别日志类型
                    if 'print(' in log_call:
                        log_type = 'print'
                    elif 'logger.' in log_call:
                        log_type = 'logger'
                    elif 'logging.' in log_call:
                        log_type = 'logging'
                    else:
                        log_type = 'custom'
                    
                    log_formats[log_type].append({
                        'file': py_file.relative_to(self.src_dir),
                        'line': line_num,
                        'call': log_call[:80]  # 只保留前80个字符
                    })
        
        self.results['logging_formats'] = dict(log_formats)
        
        for log_type, calls in log_formats.items():
            print(f"  {log_type}: {len(calls)} 次调用")
        
        print()
    
    def generate_report(self):
        """生成分析报告"""
        print("=" * 80)
        print("分析报告生成中...")
        print("=" * 80)
        print()
        
        report_path = Path('.kiro/specs/code-quality-improvement/redundancy_report.md')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 代码冗余分析报告\n\n")
            f.write(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 1. 页面检测调用
            f.write("## 1. 重复的页面检测调用\n\n")
            f.write(f"**发现 {len(self.results['page_detection_calls'])} 次页面检测调用**\n\n")
            
            # 按文件分组
            by_file = defaultdict(list)
            for call in self.results['page_detection_calls']:
                by_file[call['file']].append(call)
            
            for filename, calls in by_file.items():
                f.write(f"### {filename}\n\n")
                f.write(f"发现 {len(calls)} 次调用:\n\n")
                
                for i, call in enumerate(calls[:10], 1):  # 只显示前10个
                    f.write(f"**调用 {i}** (行 {call['line']}):\n")
                    f.write(f"```python\n{call['call']}\n```\n\n")
                
                if len(calls) > 10:
                    f.write(f"... 还有 {len(calls) - 10} 次调用\n\n")
            
            f.write("**优化建议**:\n")
            f.write("- 创建统一的页面检测缓存管理器\n")
            f.write("- 在流程开始时预检测常用页面\n")
            f.write("- 避免在循环中重复检测\n\n")
            
            # 2. 硬编码超时
            f.write("## 2. 硬编码的超时时间\n\n")
            f.write(f"**发现 {len(self.results['hardcoded_timeouts'])} 处硬编码超时**\n\n")
            
            # 按模式分组
            by_pattern = defaultdict(list)
            for timeout in self.results['hardcoded_timeouts']:
                by_pattern[timeout['pattern']].append(timeout)
            
            for pattern, timeouts in by_pattern.items():
                f.write(f"### {pattern}\n\n")
                f.write(f"发现 {len(timeouts)} 处:\n\n")
                
                # 统计超时值分布
                value_counts = defaultdict(int)
                for t in timeouts:
                    value_counts[t['value']] += 1
                
                f.write("**超时值分布**:\n")
                for value, count in sorted(value_counts.items(), key=lambda x: -x[1])[:10]:
                    f.write(f"- {value}秒: {count}次\n")
                f.write("\n")
                
                # 显示示例
                f.write("**示例**:\n\n")
                for i, timeout in enumerate(timeouts[:5], 1):
                    f.write(f"{i}. `{timeout['file']}:{timeout['line']}` - `{timeout['code']}`\n")
                f.write("\n")
            
            f.write("**优化建议**:\n")
            f.write("- 创建统一的超时配置模块 `timeouts_config.py`\n")
            f.write("- 将所有超时时间提取为配置项\n")
            f.write("- 支持运行时调整\n\n")
            
            # 3. 错误处理模式
            f.write("## 3. 重复的错误处理模式\n\n")
            f.write(f"**发现 {len(self.results['error_handling_patterns'])} 个错误处理块**\n\n")
            
            # 按模式分组
            by_pattern = defaultdict(list)
            for pattern in self.results['error_handling_patterns']:
                by_pattern[pattern['pattern']].append(pattern)
            
            for pattern_type, patterns in by_pattern.items():
                f.write(f"### {pattern_type}\n\n")
                f.write(f"发现 {len(patterns)} 个:\n\n")
                
                # 统计异常类型
                exception_counts = defaultdict(int)
                for p in patterns:
                    exception_counts[p['exception']] += 1
                
                f.write("**异常类型分布**:\n")
                for exc, count in sorted(exception_counts.items(), key=lambda x: -x[1])[:10]:
                    f.write(f"- {exc}: {count}次\n")
                f.write("\n")
            
            f.write("**优化建议**:\n")
            f.write("- 创建统一的错误处理装饰器\n")
            f.write("- 标准化日志格式\n")
            f.write("- 使用上下文管理器管理资源\n\n")
            
            # 4. 资源管理
            f.write("## 4. 重复的资源管理代码\n\n")
            f.write(f"**发现 {len(self.results['resource_management'])} 处资源管理代码**\n\n")
            
            # 按类型分组
            by_type = defaultdict(list)
            for resource in self.results['resource_management']:
                by_type[resource['type']].append(resource)
            
            for resource_type, resources in by_type.items():
                f.write(f"### {resource_type}\n\n")
                f.write(f"发现 {len(resources)} 处:\n\n")
                
                # 统计上下文管理器使用情况
                with_context = sum(1 for r in resources if r['uses_context_manager'])
                without_context = len(resources) - with_context
                
                f.write(f"- 使用上下文管理器: {with_context}次\n")
                f.write(f"- 未使用上下文管理器: {without_context}次\n\n")
                
                if without_context > 0:
                    f.write("**需要改进的示例**:\n\n")
                    count = 0
                    for resource in resources:
                        if not resource['uses_context_manager'] and count < 5:
                            f.write(f"- `{resource['file']}:{resource['line']}` - `{resource['code']}`\n")
                            count += 1
                    f.write("\n")
            
            f.write("**优化建议**:\n")
            f.write("- 使用上下文管理器统一管理资源\n")
            f.write("- 创建资源池避免频繁创建/销毁\n")
            f.write("- 实现自动清理机制\n\n")
            
            # 5. 日志格式
            f.write("## 5. 日志格式不统一问题\n\n")
            
            total_logs = sum(len(calls) for calls in self.results['logging_formats'].values())
            f.write(f"**发现 {total_logs} 次日志调用**\n\n")
            
            for log_type, calls in self.results['logging_formats'].items():
                f.write(f"### {log_type}\n\n")
                f.write(f"发现 {len(calls)} 次调用\n\n")
                
                # 显示示例
                f.write("**示例**:\n\n")
                for i, call in enumerate(calls[:5], 1):
                    f.write(f"{i}. `{call['file']}:{call['line']}` - `{call['call']}`\n")
                f.write("\n")
            
            f.write("**优化建议**:\n")
            f.write("- 创建统一的日志配置模块 `logging_config.py`\n")
            f.write("- 定义统一的日志格式\n")
            f.write("- 统一日志级别使用\n\n")
            
            # 总结
            f.write("## 总结\n\n")
            f.write("### 主要发现\n\n")
            f.write(f"1. 页面检测调用: {len(self.results['page_detection_calls'])} 次\n")
            f.write(f"2. 硬编码超时: {len(self.results['hardcoded_timeouts'])} 处\n")
            f.write(f"3. 错误处理块: {len(self.results['error_handling_patterns'])} 个\n")
            f.write(f"4. 资源管理代码: {len(self.results['resource_management'])} 处\n")
            f.write(f"5. 日志调用: {total_logs} 次\n\n")
            
            f.write("### 优先级建议\n\n")
            f.write("**P0 - 高优先级**:\n")
            f.write("- 统一页面检测调用（影响性能）\n")
            f.write("- 提取硬编码超时配置（影响可维护性）\n\n")
            
            f.write("**P1 - 中优先级**:\n")
            f.write("- 统一错误处理模式（提高代码质量）\n")
            f.write("- 改进资源管理（避免泄漏）\n\n")
            
            f.write("**P2 - 低优先级**:\n")
            f.write("- 统一日志格式（提高可读性）\n\n")
        
        print(f"✓ 报告已生成: {report_path}")
        print()


if __name__ == '__main__':
    analyzer = RedundancyAnalyzer()
    analyzer.analyze_all()
