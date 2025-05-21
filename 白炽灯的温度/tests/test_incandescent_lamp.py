#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白炽灯温度优化 - 计算最优工作温度（测试适配版）
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import minimize_scalar
import warnings

# 物理常数（使用CODATA 2018推荐值）
H = 6.62607015e-34  # 普朗克常数 (J·s)
C = 299792458       # 光速 (m/s)
K_B = 1.380649e-23  # 玻尔兹曼常数 (J/K)

# 可见光波长范围 (m) - 根据CIE标准定义
VISIBLE_LIGHT_MIN = 380e-9
VISIBLE_LIGHT_MAX = 780e-9

class IncandescentLampEfficiency:
    """白炽灯效率计算核心类（便于测试调用）"""
    
    @staticmethod
    def planck_law(wavelength, temperature):
        """
        普朗克黑体辐射公式（增加输入验证）
        
        参数:
            wavelength: 波长（m），支持标量或numpy数组
            temperature: 温度（K），必须>0
            
        返回:
            辐射强度（W/(m²·m·sr)）
        """
        if np.any(temperature <= 0):
            raise ValueError("Temperature must be positive")
        if np.any(wavelength <= 0):
            raise ValueError("Wavelength must be positive")
            
        numerator = 2.0 * H * C**2 / (wavelength**5)
        exponent = H * C / (wavelength * K_B * temperature)
        # 处理数值溢出情况
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            intensity = numerator / (np.exp(exponent) - 1)
        return np.where(exponent > 700, 0, intensity)  # 处理大指数情况
    
    @staticmethod
    def calculate_efficiency(temperature):
        """
        计算发光效率η(T)（增加积分范围保护）
        
        参数:
            temperature: 温度（K）
            
        返回:
            效率η（0-1之间）
        """
        def integrand(wl):
            return IncandescentLampEfficiency.planck_law(wl, temperature)
        
        try:
            # 可见光波段积分
            visible_power, _ = integrate.quad(
                integrand, VISIBLE_LIGHT_MIN, VISIBLE_LIGHT_MAX,
                limit=100, epsabs=1e-10, epsrel=1e-10
            )
            # 总辐射功率积分（更合理的物理范围）
            total_power, _ = integrate.quad(
                integrand, 1e-10, 1e-2,
                limit=100, epsabs=1e-10, epsrel=1e-10
            )
            return visible_power / max(total_power, 1e-20)  # 避免除零
        except Exception as e:
            print(f"Integration error at T={temperature}K: {str(e)}")
            return np.nan

def plot_efficiency_curve(temp_range=(300, 10000, 200)):
    """
    绘制效率-温度曲线（返回Figure对象便于测试）
    """
    temps = np.linspace(*temp_range)
    eff_calculator = IncandescentLampEfficiency()
    efficiencies = [eff_calculator.calculate_efficiency(T) for T in temps]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temps, efficiencies, 'b-', lw=2)
    
    # 标注关键点
    max_idx = np.nanargmax(efficiencies)
    ax.plot(temps[max_idx], efficiencies[max_idx], 'ro', ms=8)
    ax.text(temps[max_idx], efficiencies[max_idx]*0.95,
           f'Max: {efficiencies[max_idx]:.3f} @ {temps[max_idx]:.0f}K',
           ha='center')
    
    ax.set(
        title='Incandescent Lamp Efficiency vs Temperature',
        xlabel='Temperature (K)',
        ylabel='Efficiency η(T)',
        xlim=(temp_range[0], temp_range[1])
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def find_optimal_temperature():
    """最优温度搜索（适配测试验证）"""
    def objective(T):
        eff = IncandescentLampEfficiency.calculate_efficiency(T)
        return -eff if not np.isnan(eff) else float('inf')
    
    result = minimize_scalar(
        objective,
        bounds=(1000, 10000),
        method='bounded',
        options={'xatol': 1.0}
    )
    
    if result.success:
        return result.x, -result.fun
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")

def main():
    """主流程（分离测试和运行）"""
    # 效率曲线
    fig = plot_efficiency_curve()
    fig.savefig('efficiency_curve.png', dpi=300)
    
    # 最优温度计算
    try:
        T_opt, eta_max = find_optimal_temperature()
        print(f"Optimal Temperature: {T_opt:.1f} K")
        print(f"Maximum Efficiency: {eta_max:.4f} ({eta_max*100:.2f}%)")
        
        # 实际比较
        eta_actual = IncandescentLampEfficiency.calculate_efficiency(2700)
        print(f"Actual Efficiency @2700K: {eta_actual:.4f} ({eta_actual*100:.2f}%)")
        
    except Exception as e:
        print(f"Error in optimization: {str(e)}")

if __name__ == "__main__":
    main()
