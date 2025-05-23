#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薛定谔方程 - 方势阱能级计算（参考答案）

本模块实现了一维方势阱中粒子能级的计算方法。
"""

import numpy as np
import matplotlib.pyplot as plt

# 物理常数
HBAR = 1.0545718e-34  # 约化普朗克常数 (J·s)
ELECTRON_MASS = 9.1094e-31  # 电子质量 (kg)
EV_TO_JOULE = 1.6021766208e-19  # 电子伏转换为焦耳的系数


def calculate_y_values(E_values, V, w, m):
    """
    计算方势阱能级方程中的三个函数值
    
    参数:
        E_values (numpy.ndarray): 能量值数组 (eV)
        V (float): 势阱高度 (eV)
        w (float): 势阱宽度 (m)
        m (float): 粒子质量 (kg)
    
    返回:
        tuple: 包含三个numpy数组 (y1, y2, y3)，分别对应三个函数在给定能量值下的函数值
    """
    # 将能量从eV转换为J
    E_joules = E_values * EV_TO_JOULE
    V_joule = V * EV_TO_JOULE
    
    # 计算参数，避免使用过小的数值
    # 使用 (w^2 * m) / (2 * hbar^2) 作为一个整体计算
    factor = (w**2 * m) / (2 * HBAR**2)
    
    # 计算三个函数值
    y1 = np.tan(np.sqrt(factor * E_joules))
    
    # 对于y2和y3，需要处理可能的除零错误
    with np.errstate(divide='ignore', invalid='ignore'):
        y2 = np.sqrt((V_joule - E_joules) / E_joules)
        y3 = -np.sqrt(E_joules / (V_joule - E_joules))
    
    # 处理无穷大和NaN值
    y1 = np.where(np.isfinite(y1), y1, np.nan)
    y2 = np.where(np.isfinite(y2), y2, np.nan)
    y3 = np.where(np.isfinite(y3), y3, np.nan)
    
    return y1, y2, y3


def plot_energy_functions(E_values, y1, y2, y3):
    """
    绘制能级方程的三个函数曲线
    
    参数:
        E_values (numpy.ndarray): 能量值数组 (eV)
        y1 (numpy.ndarray): 函数y1的值
        y2 (numpy.ndarray): 函数y2的值
        y3 (numpy.ndarray): 函数y3的值
    
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制三个函数曲线
    ax.plot(E_values, y1, 'b-', label=r'$y_1 = \tan\sqrt{w^2mE/2\hbar^2}$')
    ax.plot(E_values, y2, 'r-', label=r'$y_2 = \sqrt{\frac{V-E}{E}}$ (偶宇称)')
    ax.plot(E_values, y3, 'g-', label=r'$y_3 = -\sqrt{\frac{E}{V-E}}$ (奇宇称)')
    
    # 添加水平和垂直参考线
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # 设置坐标轴范围，限制y轴范围以便更清晰地看到交点
    ax.set_xlim(0, 20)
    ax.set_ylim(-10, 10)
    
    # 添加标签和标题
    ax.set_xlabel('Energy E (eV)')
    ax.set_ylabel('Function value')
    ax.set_title('Square Potential Well Energy Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def energy_equation_even(E, V, w, m):
    """
    偶宇称能级方程: tan(sqrt(w^2*m*E/(2*hbar^2))) = sqrt((V-E)/E)
    返回两边的差值，用于求根
    """
    E_joule = E * EV_TO_JOULE
    V_joule = V * EV_TO_JOULE
    factor = (w**2 * m) / (2 * HBAR**2)
    
    left = np.tan(np.sqrt(factor * E_joule))
    right = np.sqrt((V_joule - E_joule) / E_joule)
    
    return left - right


def energy_equation_odd(E, V, w, m):
    """
    奇宇称能级方程: tan(sqrt(w^2*m*E/(2*hbar^2))) = -sqrt(E/(V-E))
    返回两边的差值，用于求根
    """
    E_joule = E * EV_TO_JOULE
    V_joule = V * EV_TO_JOULE
    factor = (w**2 * m) / (2 * HBAR**2)
    
    left = np.tan(np.sqrt(factor * E_joule))
    right = -np.sqrt(E_joule / (V_joule - E_joule))
    
    return left - right


def find_energy_level_bisection(n, V, w, m, precision=0.001, E_min=0.001, E_max=None):
    """
    使用二分法求解方势阱中的第n个能级
    
    参数:
        n (int): 能级序号 (0表示基态，1表示第一激发态，以此类推)
        V (float): 势阱高度 (eV)
        w (float): 势阱宽度 (m)
        m (float): 粒子质量 (kg)
        precision (float): 求解精度 (eV)
        E_min (float): 能量搜索下限 (eV)
        E_max (float): 能量搜索上限 (eV)，默认为V
    
    返回:
        float: 第n个能级的能量值 (eV)
    """
    if E_max is None:
        E_max = V - 0.001  # 避免在V处的奇点
    
    # 根据能级序号n选择合适的方程
    if n % 2 == 0:  # 偶数能级 (0, 2, 4, ...)
        equation = lambda E: energy_equation_even(E, V, w, m)
    else:  # 奇数能级 (1, 3, 5, ...)
        equation = lambda E: energy_equation_odd(E, V, w, m)
    
    # 初始化搜索区间
    a, b = E_min, E_max
    
    # 检查区间端点的函数值符号是否相反
    fa, fb = equation(a), equation(b)
    if fa * fb > 0:
        # 如果端点函数值符号相同，需要调整搜索区间
        # 这里简化处理，实际应用中可能需要更复杂的策略
        # 例如，可以在区间内采样多个点，寻找函数值符号变化的区间
        raise ValueError(f"无法在给定区间 [{a}, {b}] 内找到第 {n} 个能级")
    
    # 二分法迭代
    while (b - a) > precision:
        c = (a + b) / 2  # 区间中点
        fc = equation(c)
        
        if abs(fc) < 1e-10:  # 如果中点非常接近根
            return c
        
        if fa * fc < 0:  # 如果根在左半区间
            b = c
            fb = fc
        else:  # 如果根在右半区间
            a = c
            fa = fc
    
    # 返回区间中点作为近似解
    return (a + b) / 2


def main():
    """
    主函数，执行方势阱能级的计算和可视化
    """
    # 参数设置
    V = 20.0  # 势阱高度 (eV)
    w = 1e-9  # 势阱宽度 (m)
    m = ELECTRON_MASS  # 粒子质量 (kg)
    
    # 1. 计算并绘制函数曲线
    E_values = np.linspace(0.001, 19.999, 1000)  # 能量范围 (eV)
    y1, y2, y3 = calculate_y_values(E_values, V, w, m)
    fig = plot_energy_functions(E_values, y1, y2, y3)
    plt.savefig('energy_functions.png', dpi=300)
    plt.show()
    
    # 2. 使用二分法计算前6个能级
    energy_levels = []
    for n in range(6):
        energy = find_energy_level_bisection(n, V, w, m)
        energy_levels.append(energy)
        print(f"能级 {n}: {energy:.3f} eV")
    
    # 与参考值比较
    reference_levels = [0.318, 1.270, 2.851, 5.050, 7.850, 11.215]
    print("\n参考能级值:")
    for n, ref in enumerate(reference_levels):
        print(f"能级 {n}: {ref:.3f} eV")
    
    # 计算相对误差
    print("\n相对误差:")
    for n, (calc, ref) in enumerate(zip(energy_levels, reference_levels)):
        rel_error = abs(calc - ref) / ref * 100
        print(f"能级 {n}: {rel_error:.2f}%")


if __name__ == "__main__":
    main()
