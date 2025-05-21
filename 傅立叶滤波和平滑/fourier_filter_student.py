#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
傅立叶滤波和平滑 - 道琼斯工业平均指数分析解决方案

本模块实现了对道Jones工业平均指数数据的傅立叶分析和滤波处理。
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """
    加载道Jones工业平均指数数据
    
    参数:
        filename (str): 数据文件路径
    
    返回:
        numpy.ndarray: 指数数组
    """
    try:
        filepath = "dow.txt"
        return np.loadtxt(filepath)
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise

def plot_data(data, title="Dow Jones Industrial Average"):
    """
    绘制时间序列数据
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Trading Day")
    plt.ylabel("Index Value")
    plt.grid(True, alpha=0.3)
    plt.show()
    return fig  # 确保返回Figure对象


def fourier_filter(data, keep_fraction=0.1):
    """
    执行傅立叶变换并滤波
    
    参数:
        data (numpy.ndarray): 输入数据数组
        keep_fraction (float): 保留的傅立叶系数比例
    
    返回:
        tuple: (滤波后的数据数组, 原始傅立叶系数数组)
    """
    # 计算实数信号的傅立叶变换
    fft_coeff = np.fft.rfft(data)
    
    # 计算保留的系数数量
    cutoff = int(len(fft_coeff) * keep_fraction)
    
    # 创建滤波后的系数数组
    filtered_coeff = fft_coeff.copy()
    filtered_coeff[cutoff:] = 0
    
    # 计算逆变换
    filtered_data = np.fft.irfft(filtered_coeff, n=len(data))
    
    return filtered_data, fft_coeff

def plot_comparison(original, filtered, title="Fourier Filter Result"):
    """
    绘制原始数据和滤波结果的比较
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(original, 'g-', linewidth=1, label="Original Data")
    plt.plot(filtered, 'r-', linewidth=2, label="Filtered Result")
    plt.title(title)
    plt.xlabel("Trading Day")
    plt.ylabel("Index Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    return fig  # 确保返回Figure对象


def main():
    # 任务1：数据加载与可视化
    data = load_data('dow.txt')
    plot_data(data, "Dow Jones Industrial Average - Original Data")
    
    # 任务2：傅立叶变换与滤波（保留前10%系数）
    filtered_10, coeff = fourier_filter(data, 0.1)
    plot_comparison(data, filtered_10, "Fourier Filter (Keep Top 10% Coefficients)")

    # 任务3：修改滤波参数（保留前2%系数）
    filtered_2, _ = fourier_filter(data, 0.02)
    plot_comparison(data, filtered_2, "Fourier Filter (Keep Top 2% Coefficients)")
    

if __name__ == "__main__":
    main()
