#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白炽灯温度优化 - 学生代码模板

请根据项目说明实现以下函数，完成白炽灯效率与最优温度的计算。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import minimize_scalar

# 物理常数
H = 6.62607015e-34  # 普朗克常数 (J·s)
C = 299792458       # 光速 (m/s)
K_B = 1.380649e-23  # 玻尔兹曼常数 (J/K)

# 可见光波长范围 (m)
VISIBLE_LIGHT_MIN = 380e-9  # 380 nm
VISIBLE_LIGHT_MAX = 780e-9  # 780 nm


def planck_law(wavelength, temperature):
    """
    计算普朗克黑体辐射公式

    参数:
        wavelength (float or numpy.ndarray): 波长，单位为米
        temperature (float): 温度，单位为开尔文

    返回:
        float or numpy.ndarray: 给定波长和温度下的辐射强度 (W/(m²·m))
    """
    # 避免指数溢出
    exponent = H * C / (wavelength * K_B * temperature)
    if isinstance(wavelength, np.ndarray):
        denominator = np.where(exponent > 700, np.inf, np.exp(exponent) - 1)
    else:
        denominator = np.inf if exponent > 700 else np.exp(exponent) - 1
    numerator = 2 * H * C**2 / (wavelength**5)
    intensity = numerator / denominator
    return intensity


def calculate_visible_power_ratio(temperature):
    """
    计算给定温度下可见光功率与总辐射功率的比值

    参数:
        temperature (float): 温度，单位为开尔文

    返回:
        float: 可见光效率（可见光功率/总功率）
    """
    # 调整总辐射功率积分范围，避免不必要的计算
    total_power, _ = integrate.quad(planck_law, 1e-9, 1e-3, args=(temperature,))
    visible_power, _ = integrate.quad(
        planck_law, VISIBLE_LIGHT_MIN, VISIBLE_LIGHT_MAX, args=(temperature,))
    visible_power_ratio = visible_power / total_power
    return visible_power_ratio


def plot_efficiency_vs_temperature(temp_range):
    """
    绘制效率-温度关系曲线

    参数:
        temp_range (numpy.ndarray): 温度范围，单位为开尔文

    返回:
        tuple: (matplotlib.figure.Figure, numpy.ndarray, numpy.ndarray)
        图形对象、温度数组、效率数组
    """
    efficiencies = []
    for temp in temp_range:
        efficiency = calculate_visible_power_ratio(temp)
        efficiencies.append(efficiency)
    efficiencies = np.array(efficiencies)

    fig = plt.figure()
    plt.plot(temp_range, efficiencies)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Visible Light Efficiency')
    plt.title('Incandescent Lamp Efficiency vs Temperature')
    plt.grid(True)

    return fig, temp_range, efficiencies


def find_optimal_temperature():
    """
    寻找使白炽灯效率最大的最优温度

    返回:
        tuple: (float, float) 最优温度和对应的效率
    """
    def negative_efficiency(temperature):
        return -calculate_visible_power_ratio(temperature)

    result = minimize_scalar(negative_efficiency, bounds=(
        1000, 10000), options={'xatol': 1.0})
    optimal_temp = result.x
    optimal_efficiency = -result.fun

    return optimal_temp, optimal_efficiency


def main():
    """
    主函数，计算并可视化最优温度
    """
    # 绘制效率-温度曲线 (1000K-10000K)
    temp_range = np.linspace(1000, 10000, 100)
    fig_efficiency, temps, effs = plot_efficiency_vs_temperature(temp_range)
    plt.savefig('efficiency_vs_temperature.png', dpi=300)
    plt.show()

    # 计算最优温度
    optimal_temp, optimal_efficiency = find_optimal_temperature()
    print(f"\n最优温度: {optimal_temp:.1f} K")
    print(f"最大效率: {optimal_efficiency:.4f} ({optimal_efficiency*100:.2f}%)")

    # 与实际白炽灯温度比较
    actual_temp = 2700
    actual_efficiency = calculate_visible_power_ratio(actual_temp)
    print(f"\n实际灯丝温度: {actual_temp} K")
    print(f"实际效率: {actual_efficiency:.4f} ({actual_efficiency*100:.2f}%)")
    print(f"效率差异: {(optimal_efficiency - actual_efficiency)*100:.2f}%")

    # 标记最优和实际温度点
    plt.figure(figsize=(10, 6))
    plt.plot(temps, effs, 'b-')
    plt.plot(optimal_temp, optimal_efficiency, 'ro',
             markersize=8, label=f'Optimal: {optimal_temp:.1f} K')
    plt.plot(actual_temp, actual_efficiency, 'go',
             markersize=8, label=f'Actual: {actual_temp} K')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Visible Light Efficiency')
    plt.title('Incandescent Lamp Efficiency vs Temperature')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('optimal_temperature.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
