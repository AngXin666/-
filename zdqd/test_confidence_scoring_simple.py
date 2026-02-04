"""
测试置信度评分系统的实现（简化版）
Test the confidence scoring system implementation (simplified)
"""

import re
from typing import Optional, Dict


class ConfidenceScorer:
    """置信度评分器（从ProfileReader提取的方法）"""
    
    def _is_chinese_char(self, char: str) -> bool:
        """检查单个字符是否为中文"""
        return '\u4e00' <= char <= '\u9fff'
    
    def _is_chinese_text(self, text: str) -> bool:
        """检查文本是否包含中文字符"""
        chinese_count = sum(1 for c in text if self._is_chinese_char(c))
        return chinese_count > 0
    
    def _is_pure_number(self, text: str) -> bool:
        """检查文本是否为纯数字"""
        return text.isdigit()
    
    def _is_pure_symbol(self, text: str) -> bool:
        """检查文本是否为纯特殊符号"""
        return all(not c.isalnum() for c in text)
    
    def _calculate_nickname_confidence(
        self, 
        text: str, 
        position_info: Optional[Dict] = None
    ) -> float:
        """计算昵称候选的置信度分数"""
        # 排除关键字列表
        exclude_keywords = [
            "ID", "id", "手机", "余额", "积分", 
            "抵扣券", "优惠券", "抵扣券", "我的", "设置", "首页", "分类",
            "商城", "订单", "查看", "待付款", "待发货", "待收货", "待评价",
            "溪盟", "山泉", "干溪", "汇盟",
            "元", "张", "次"
        ]
        
        # 检查排除关键字(返回0分)
        for kw in exclude_keywords:
            if kw in text:
                return 0.0
        
        # 1. 基础分数
        score = 0.3
        
        # 2. 中文字符加分 (+0.3)
        if self._is_chinese_text(text):
            score += 0.3
        
        # 3. 长度评分
        text_len = len(text)
        if 2 <= text_len <= 10:
            score += 0.2  # 理想长度
        elif 1 <= text_len <= 20:
            score += 0.1  # 可接受长度
        
        # 4. 纯数字惩罚 (-0.3)
        if self._is_pure_number(text) and text_len <= 3:
            score -= 0.3
        
        # 5. 特殊符号惩罚 (-0.1 per symbol, max -0.3)
        symbol_count = sum(1 for c in text if not c.isalnum() and not self._is_chinese_char(c))
        if symbol_count > 0:
            score -= 0.1 * min(symbol_count, 3)
        
        # 6. 位置加分 (+0.2)
        if position_info:
            try:
                text_center_x = position_info.get('center_x')
                text_center_y = position_info.get('center_y')
                region_center_x = position_info.get('region_center_x')
                region_center_y = position_info.get('region_center_y')
                
                if all([text_center_x is not None, text_center_y is not None,
                       region_center_x is not None, region_center_y is not None]):
                    # 计算距离
                    distance = ((text_center_x - region_center_x) ** 2 + 
                               (text_center_y - region_center_y) ** 2) ** 0.5
                    
                    # 如果距离小于50像素,认为靠近中心
                    if distance < 50:
                        score += 0.2
            except Exception:
                pass  # 位置信息处理失败,跳过位置加分
        
        # 确保分数在0.0-1.0范围内
        return max(0.0, min(1.0, score))


def test_confidence_scoring():
    """测试置信度评分功能"""
    
    scorer = ConfidenceScorer()
    
    print("=" * 60)
    print("测试置信度评分系统")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        # (文本, 期望分数范围, 描述)
        ("张三", (0.6, 1.0), "纯中文昵称"),
        ("李四VIP", (0.5, 0.9), "中文+字母昵称"),
        ("123", (0.0, 0.3), "短纯数字"),
        ("1234567", (0.3, 0.6), "长纯数字"),
        ("ID123456", (0.0, 0.0), "包含排除关键字ID"),
        ("余额100", (0.0, 0.0), "包含排除关键字余额"),
        ("积分", (0.0, 0.0), "包含排除关键字积分"),
        ("王五@#", (0.3, 0.7), "包含特殊符号"),
        ("赵六", (0.6, 1.0), "2字中文昵称"),
        ("孙七八九十一二三四五", (0.4, 0.7), "长中文昵称"),
        ("abc", (0.3, 0.6), "纯英文"),
    ]
    
    passed = 0
    failed = 0
    
    for text, (min_score, max_score), description in test_cases:
        score = scorer._calculate_nickname_confidence(text)
        
        # 检查分数是否在0.0-1.0范围内
        if not (0.0 <= score <= 1.0):
            print(f"✗ {description}: '{text}' -> {score:.2f} (超出范围!)")
            failed += 1
            continue
        
        # 检查分数是否在期望范围内
        if min_score <= score <= max_score:
            print(f"✓ {description}: '{text}' -> {score:.2f} (期望: {min_score:.2f}-{max_score:.2f})")
            passed += 1
        else:
            print(f"✗ {description}: '{text}' -> {score:.2f} (期望: {min_score:.2f}-{max_score:.2f})")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    # 测试位置加分
    print("\n测试位置加分功能:")
    print("-" * 60)
    
    # 测试靠近中心的文本
    position_info_near = {
        'center_x': 200,
        'center_y': 100,
        'region_center_x': 210,
        'region_center_y': 105
    }
    
    # 测试远离中心的文本
    position_info_far = {
        'center_x': 200,
        'center_y': 100,
        'region_center_x': 300,
        'region_center_y': 200
    }
    
    score_near = scorer._calculate_nickname_confidence("张三", position_info_near)
    score_far = scorer._calculate_nickname_confidence("张三", position_info_far)
    score_no_pos = scorer._calculate_nickname_confidence("张三", None)
    
    print(f"靠近中心: {score_near:.2f}")
    print(f"远离中心: {score_far:.2f}")
    print(f"无位置信息: {score_no_pos:.2f}")
    
    if score_near > score_far:
        print("✓ 位置加分功能正常")
        passed += 1
    else:
        print("✗ 位置加分功能异常")
        failed += 1
    
    # 测试辅助方法
    print("\n测试辅助方法:")
    print("-" * 60)
    
    # 测试中文检测
    assert scorer._is_chinese_text("张三") == True, "中文检测失败"
    assert scorer._is_chinese_text("123") == False, "中文检测失败"
    assert scorer._is_chinese_text("张三123") == True, "中文检测失败"
    print("✓ _is_chinese_text 正常")
    passed += 1
    
    # 测试纯数字检测
    assert scorer._is_pure_number("123") == True, "纯数字检测失败"
    assert scorer._is_pure_number("12.3") == False, "纯数字检测失败"
    assert scorer._is_pure_number("1a3") == False, "纯数字检测失败"
    print("✓ _is_pure_number 正常")
    passed += 1
    
    # 测试纯符号检测
    assert scorer._is_pure_symbol("@#$") == True, "纯符号检测失败"
    assert scorer._is_pure_symbol("a@#") == False, "纯符号检测失败"
    assert scorer._is_pure_symbol("123") == False, "纯符号检测失败"
    print("✓ _is_pure_symbol 正常")
    passed += 1
    
    # 测试中文字符检测
    assert scorer._is_chinese_char("张") == True, "中文字符检测失败"
    assert scorer._is_chinese_char("a") == False, "中文字符检测失败"
    assert scorer._is_chinese_char("1") == False, "中文字符检测失败"
    print("✓ _is_chinese_char 正常")
    passed += 1
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print(f"总计: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return passed, failed


if __name__ == "__main__":
    import sys
    passed, failed = test_confidence_scoring()
    
    # 如果有失败的测试，退出码为1
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)
