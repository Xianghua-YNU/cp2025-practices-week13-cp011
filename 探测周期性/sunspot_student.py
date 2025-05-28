#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳黑子周期性分析 - 参考答案

本模块实现太阳黑子数据的周期性分析，包括：
1. 数据获取与可视化
2. 傅里叶变换分析
3. 周期确定
"""

import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen

def load_sunspot_data(url):
    """
    从本地文件读取太阳黑子数据
    
    参数:
        data (str): 本地文件路径
        
    返回:
        tuple: (years, sunspots) 年份和太阳黑子数
    """
    # 使用np.loadtxt读取数据，只保留第2(年份)和3(太阳黑子数)列
    data = np.loadtxt(url, usecols=(2,3), comments='#')
    
    # 分离数据列
    years = data[:,0]
    sunspots = data[:,1]
    
    return years, sunspots
    
    # 从URL下载数据
    response = urlopen(url)
    data = response.read().decode('utf-8').split('\n')
    
    # 解析数据
    years = []
    months = []
    sunspots = []
    
    for line in data:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            years.append(float(parts[0]))
            months.append(float(parts[1]))
            sunspots.append(float(parts[3]))
    
    return np.array(years), np.array(months), np.array(sunspots)

def plot_sunspot_data(years, sunspots):
    """
    绘制太阳黑子数据随时间变化图
    
    参数:
        years (numpy.ndarray): 年份数组
        sunspots (numpy.ndarray): 太阳黑子数数组
    """
    plt.figure(figsize=(12, 6))
    plt.plot(years, sunspots)
    plt.xlabel('Year')
    plt.ylabel('Sunspot Number')
    plt.title('Sunspot Number Variation (1749-Present)')
    plt.grid(True)
    plt.show()

def compute_power_spectrum(sunspots):
    """
    计算太阳黑子数据的功率谱
    
    参数:
        sunspots (numpy.ndarray): 太阳黑子数数组
        
    返回:
        tuple: (frequencies, power) 频率数组和功率谱
    """
    
    # 傅里叶变换
    n = sunspots.size
    fft_result = np.fft.fft(sunspots)
    
    # 计算功率谱 (只取正频率部分)
    power = np.abs(fft_result[:n//2])**2
    frequencies = np.fft.fftfreq(n, d=1)[:n//2]  # 每月采样一次
    
    return frequencies, power

def plot_power_spectrum(frequencies, power):
    """
    绘制功率谱图
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组
    """
    plt.figure(figsize=(12, 6))
    plt.plot(1/frequencies[1:], power[1:])  # 忽略零频率
    plt.xlabel('Period (months)')
    plt.ylabel('Power')
    plt.title('Power Spectrum of Sunspot Data')
    plt.grid(True)
    plt.show()

def find_main_period(frequencies, power):
    """
    找出功率谱中的主周期
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组
        
    返回:
        float: 主周期（月）
    """
    # 忽略零频率
    idx = np.argmax(power[1:]) + 1
    main_period = 1 / frequencies[idx]
    return main_period

def main():
    # 数据URL (月平均太阳黑子数)
    data = "sunspot_data.txt"
    
    # 1. 下载并可视化数据
    years, sunspots = load_sunspot_data(data)
    plot_sunspot_data(years, sunspots)
    
    # 2. 傅里叶变换分析
    frequencies, power = compute_power_spectrum(sunspots)
    plot_power_spectrum(frequencies, power)
    
    # 3. 确定主周期
    main_period = find_main_period(frequencies, power)
    print(f"\nMain period of sunspot cycle: {main_period:.2f} months")
    print(f"Approximately {main_period/12:.2f} years")

if __name__ == "__main__":
    main()
