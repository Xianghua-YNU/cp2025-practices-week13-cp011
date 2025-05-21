#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白炽灯温度优化 - 测试代码
"""
import unittest
import numpy as np
from scipy.integrate import quad

# 导入解决方案模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incandescent_lamp_student import (
    planck_law,
    calculate_visible_power_ratio,
    find_optimal_temperature,
    H, C, K_B,
    VISIBLE_LIGHT_MIN, VISIBLE_LIGHT_MAX
)

class TestPlanckLaw(unittest.TestCase):
    """测试普朗克黑体辐射公式"""
    
    def test_planck_law_at_500nm_3000K(self):
        """测试500nm波长在3000K温度下的辐射强度"""
        wavelength = 500e-9  # 500 nm
        temp = 3000  # 3000 K
        expected = 2.0 * H * C**2 / (wavelength**5) / (np.exp(H*C/(wavelength*K_B*temp)) - 1)
        result = planck_law(wavelength, temp)
        self.assertAlmostEqual(result, expected, places=10,
                              msg="普朗克公式计算值与理论值不符")
    
    def test_planck_law_array_input(self):
        """测试数组输入"""
        wavelengths = np.array([400e-9, 500e-9, 600e-9])
        temp = 3000
        results = planck_law(wavelengths, temp)
        self.assertEqual(len(results), 3,
                         "数组输入输出维度不匹配")
        for i, wl in enumerate(wavelengths):
            expected = 2.0 * H * C**2 / (wl**5) / (np.exp(H*C/(wl*K_B*temp)) - 1)
            self.assertAlmostEqual(results[i], expected, places=10,
                                 msg=f"{wl*1e9}nm波长计算结果错误")
    
    def test_planck_law_zero_kelvin(self):
        """测试绝对零度边界条件"""
        with self.assertRaises(ValueError):
            planck_law(500e-9, 0)

class TestVisiblePowerRatio(unittest.TestCase):
    """测试可见光功率比计算"""
    
    def test_visible_power_ratio_at_3000K(self):
        """测试3000K时的可见光功率比"""
        temp = 3000
        result = calculate_visible_power_ratio(temp)
        
        # 手动计算验证
        def intensity(wl):
            return planck_law(wl, temp)
        
        visible_power, _ = quad(intensity, VISIBLE_LIGHT_MIN, VISIBLE_LIGHT_MAX)
        total_power, _ = quad(intensity, 1e-10, 1e-2)  # 更合理的积分范围
        expected = visible_power / total_power
        
        self.assertAlmostEqual(result, expected, places=6,
                             msg="可见光功率比计算错误")
    
    def test_visible_power_ratio_at_6000K(self):
        """测试6000K时的可见光功率比"""
        temp = 6000
        result = calculate_visible_power_ratio(temp)
        
        # 物理合理性验证
        self.assertGreater(result, 0.12,
                         "6000K时效率应大于12%")
        self.assertLess(result, 0.15,
                       "6000K时效率应小于15%")
    
    def test_visible_power_ratio_extreme_temp(self):
        """测试极端温度下的行为"""
        # 超低温测试
        low_temp_eff = calculate_visible_power_ratio(100)
        self.assertAlmostEqual(low_temp_eff, 0, places=4,
                             msg="低温下应有接近0的效率")
        
        # 超高温测试
        high_temp_eff = calculate_visible_power_ratio(15000)
        self.assertLess(high_temp_eff, 0.2,
                      "15000K时效率不应超过理论极限")

class TestOptimalTemperature(unittest.TestCase):
    """测试最优温度查找"""
    
    def test_optimal_temperature_range(self):
        """测试最优温度在合理范围内"""
        opt_temp, opt_eff = find_optimal_temperature()
        
        # 理论范围验证
        self.assertGreaterEqual(opt_temp, 6800,
                              "最优温度应≥6800K")
        self.assertLessEqual(opt_temp, 7100,
                           "最优温度应≤7100K")
        
        # 极值点验证
        for delta in [-100, 100]:
            current_eff = calculate_visible_power_ratio(opt_temp + delta)
            self.assertLessEqual(current_eff, opt_eff,
                               f"{opt_temp+delta}K处效率不应超过最优值")
    
    def test_optimal_efficiency_value(self):
        """测试最优效率值"""
        _, opt_eff = find_optimal_temperature()
        self.assertGreaterEqual(opt_eff, 0.146,
                              "最大效率应≥14.6%")
        self.assertLessEqual(opt_eff, 0.150,
                           "最大效率应≤15.0%")
    
    def test_optimization_precision(self):
        """测试优化精度达到1K"""
        opt_temp, _ = find_optimal_temperature()
        # 检查是否为整数（1K精度）
        self.assertAlmostEqual(opt_temp, round(opt_temp), places=0,
                             msg="温度优化精度未达到1K")

if __name__ == "__main__":
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
