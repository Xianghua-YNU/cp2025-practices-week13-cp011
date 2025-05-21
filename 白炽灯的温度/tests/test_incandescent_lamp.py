#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白炽灯效率优化 - 测试适配完整版
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import minimize_scalar
import warnings
from typing import Tuple, Union

class IncandescentLamp:
    """白炽灯效率计算与优化核心类"""
    
    # 物理常数 (CODATA 2018)
    PLANCK_CONSTANT = 6.62607015e-34  # h (J·s)
    SPEED_OF_LIGHT = 299792458        # c (m/s)
    BOLTZMANN_CONST = 1.380649e-23   # k_B (J/K)
    
    # 可见光范围 (CIE标准)
    VISIBLE_MIN = 380e-9  # 380 nm
    VISIBLE_MAX = 780e-9  # 780 nm

    @staticmethod
    def planck_spectrum(wavelength: Union[float, np.ndarray], 
                       temperature: float) -> Union[float, np.ndarray]:
        """
        计算普朗克黑体辐射光谱
        
        Args:
            wavelength: 波长(m)，支持标量或数组
            temperature: 绝对温度(K)
            
        Returns:
            光谱辐射强度 (W/m³-sr)
            
        Raises:
            ValueError: 如果温度或波长非正
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if np.any(wavelength <= 0):
            raise ValueError("Wavelength must be positive")
            
        hc = IncandescentLamp.PLANCK_CONSTANT * IncandescentLamp.SPEED_OF_LIGHT
        kT = IncandescentLamp.BOLTZMANN_CONST * temperature
        
        numerator = 2 * hc**2 / wavelength**5
        exponent = hc / (wavelength * kT)
        
        # 处理数值溢出
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            intensity = numerator / (np.exp(exponent) - 1)
        
        return np.where(exponent > 700, 0, intensity)

    @staticmethod
    def calculate_efficiency(temperature: float) -> float:
        """
        计算给定温度下的发光效率η(T)
        
        Args:
            temperature: 绝对温度(K)
            
        Returns:
            发光效率η (0-1之间)
            
        Raises:
            RuntimeError: 如果积分计算失败
        """
        def integrand(wl):
            return IncandescentLamp.planck_spectrum(wl, temperature)
        
        try:
            # 可见光波段功率
            visible, _ = integrate.quad(
                integrand,
                IncandescentLamp.VISIBLE_MIN,
                IncandescentLamp.VISIBLE_MAX,
                limit=200,
                epsabs=1e-12,
                epsrel=1e-12
            )
            
            # 总辐射功率 (物理合理范围)
            total, _ = integrate.quad(
                integrand,
                1e-10,  # 0.1 nm
                1e-2,   # 1 cm
                limit=200,
                epsabs=1e-12,
                epsrel=1e-12
            )
            
            return visible / max(total, 1e-20)
            
        except Exception as e:
            raise RuntimeError(f"Integration failed at T={temperature}K: {str(e)}")

    @staticmethod
    def find_optimal_temperature() -> Tuple[float, float]:
        """
        寻找最大效率对应的最优温度
        
        Returns:
            (最优温度(K), 最大效率)
            
        Raises:
            RuntimeError: 如果优化失败
        """
        def objective(T):
            try:
                return -IncandescentLamp.calculate_efficiency(T)
            except RuntimeError:
                return float('inf')
        
        result = minimize_scalar(
            objective,
            bounds=(1000, 10000),
            method='bounded',
            options={'xatol': 1.0}
        )
        
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        
        return result.x, -result.fun

    @staticmethod
    def plot_efficiency_curve(temp_range: Tuple[float, float, int] = (300, 10000, 200)) -> plt.Figure:
        """
        绘制效率-温度曲线
        
        Args:
            temp_range: (起始温度, 结束温度, 点数)
            
        Returns:
            matplotlib Figure对象
        """
        temps = np.linspace(*temp_range)
        effs = []
        
        for T in temps:
            try:
                eff = IncandescentLamp.calculate_efficiency(T)
            except RuntimeError:
                eff = np.nan
            effs.append(eff)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot(temps, effs, 'b-', lw=2)
        
        # 标注最大值
        valid_idx = ~np.isnan(effs)
        if np.any(valid_idx):
            max_idx = np.nanargmax(effs)
            ax.plot(temps[max_idx], effs[max_idx], 'ro', ms=8)
            ax.text(
                temps[max_idx], effs[max_idx] * 0.95,
                f'Max: {effs[max_idx]:.3f} @ {temps[max_idx]:.0f}K',
                ha='center', fontsize=10
            )
        
        ax.set(
            title='Incandescent Lamp Efficiency',
            xlabel='Temperature (K)',
            ylabel='Efficiency η(T)',
            xlim=(temp_range[0], temp_range[1]),
            ylim=(0, 0.2)
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        return fig

def main():
    """主执行函数"""
    try:
        # 计算最优温度
        T_opt, eta_max = IncandescentLamp.find_optimal_temperature()
        print(f"Optimal Temperature: {T_opt:.1f} K")
        print(f"Maximum Efficiency: {eta_max:.4f} ({eta_max*100:.2f}%)")
        
        # 实际工作点比较
        eta_actual = IncandescentLamp.calculate_efficiency(2700)
        print(f"Actual Efficiency @2700K: {eta_actual:.4f} ({eta_actual*100:.2f}%)")
        
        # 绘制曲线
        fig = IncandescentLamp.plot_efficiency_curve()
        fig.savefig('efficiency_curve.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
