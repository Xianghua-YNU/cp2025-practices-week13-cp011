import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import minimize_scalar

# 物理常数定义 - 用于普朗克黑体辐射定律计算
H = 6.62607015e-34  # 普朗克常数 (J·s)
C = 299792458       # 光速 (m/s)
K_B = 1.380649e-23  # 玻尔兹曼常数 (J/K)

# 可见光波长范围 (m) - 用于计算可见光效率
VISIBLE_LIGHT_MIN = 380e-9  # 380 nm (紫光)
VISIBLE_LIGHT_MAX = 780e-9  # 780 nm (红光)


def planck_law(wavelength, temperature):
    """
    计算普朗克黑体辐射公式
    描述了黑体在特定温度下在不同波长处的电磁辐射分布
    
    参数:
        wavelength (float or numpy.ndarray): 波长，单位为米
        temperature (float): 温度，单位为开尔文
    
    返回:
        float or numpy.ndarray: 给定波长和温度下的辐射强度 (W/(m²·m·sr))
    """
    # 普朗克定律公式: I(λ,T) = (2hc²/λ⁵) / (e^(hc/λkT) - 1)
    numerator = 2.0 * H * C**2 / (wavelength**5)  # 分子部分: 2hc²/λ⁵
    exponent = np.exp(H * C / (wavelength * K_B * temperature))  # 指数部分: e^(hc/λkT)
    intensity = numerator / (exponent - 1.0)  # 完整公式
    return intensity


def calculate_visible_power_ratio(temperature):
    """
    计算给定温度下可见光功率与总辐射功率的比值
    该比值代表了白炽灯将电能转化为可见光的效率
    
    参数:
        temperature (float): 温度，单位为开尔文
    
    返回:
        float: 可见光效率（可见光功率/总功率）
    """
    # 定义被积函数 - 普朗克定律描述的是光谱辐射率
    def intensity_function(wavelength):
        return planck_law(wavelength, temperature)
    
    # 计算可见光波段(380-780nm)的积分 - 得到可见光功率
    visible_power, _ = integrate.quad(intensity_function, VISIBLE_LIGHT_MIN, VISIBLE_LIGHT_MAX)
    
    # 计算全波段(1nm-10000nm)的积分 - 得到总辐射功率
    # 注意：积分范围选择1nm-10000nm是为了覆盖绝大部分辐射能量
    total_power, _ = integrate.quad(intensity_function, 1e-9, 10000e-9)
    
    # 计算效率：可见光功率占总功率的比例
    visible_power_ratio = visible_power / total_power
    return visible_power_ratio


def plot_efficiency_vs_temperature(temp_range):
    """
    绘制效率-温度关系曲线
    展示白炽灯效率如何随温度变化，帮助找到最优工作点
    
    参数:
        temp_range (numpy.ndarray): 温度范围，单位为开尔文
    
    返回:
        tuple: (matplotlib.figure.Figure, numpy.ndarray, numpy.ndarray) 
               图形对象、温度数组、效率数组
    """
    # 计算每个温度点对应的可见光效率
  

# 使用列表推导式遍历温度范围并计算效率
    efficiencies = np.array([calculate_visible_power_ratio(temp) for temp in temp_range])
    
    # 创建图形并绘制效率曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temp_range, efficiencies, 'b-')
    
    # 找到最大效率对应的温度点
    max_idx = np.argmax(efficiencies)
    max_temp = temp_range[max_idx]
    max_efficiency = efficiencies[max_idx]
    
    # 标记峰值点并添加文本说明
    ax.plot(max_temp, max_efficiency, 'ro', markersize=8)  # 红色圆点标记峰值
    ax.text(max_temp, max_efficiency * 0.95, 
            f'Max efficiency: {max_efficiency:.4f}\nTemperature: {max_temp:.1f} K', 
            ha='center')  # 添加文本说明
    
    # 设置图表属性
    ax.set_title('Incandescent Lamp Efficiency vs Temperature')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Visible Light Efficiency')
    ax.grid(True, alpha=0.3)  # 添加网格线，透明度0.3
    fig.tight_layout()  # 自动调整布局
    
    return fig, temp_range, efficiencies


def find_optimal_temperature():
    """
    寻找使白炽灯效率最大的最优温度
    使用优化算法在给定温度范围内找到效率最高点
    
    返回:
        tuple: (float, float) 最优温度和对应的效率
    """
    # 定义目标函数 - 由于要找最大值，所以返回负值用于最小化
    def objective(temperature):
        return -calculate_visible_power_ratio(temperature)
    
    # 使用scipy的minimize_scalar函数进行优化
    # 方法选择'bounded'，因为我们知道温度范围在1000-10000K之间
    result = minimize_scalar(
        objective,
        bounds=(1000, 10000),
        method='bounded',
        options={'xatol': 1.0}  # 温度精度设置为1K
    )
    
    optimal_temp = result.x  # 最优温度
    optimal_efficiency = -result.fun  # 最大效率（取负值还原）
    return optimal_temp, optimal_efficiency


def main():
    """
    主函数，计算并可视化最优温度
    整合所有功能，生成结果并展示
    """
    # 第一部分：绘制效率-温度曲线
    print("正在计算并绘制效率-温度曲线...")
    temp_range = np.linspace(1000, 10000, 100)  # 创建1000-10000K的温度范围，共100个点
    fig_efficiency, temps, effs = plot_efficiency_vs_temperature(temp_range)
    plt.savefig('efficiency_vs_temperature.png', dpi=300)  # 保存图表为PNG文件
    plt.show()  # 显示图表
    
    # 第二部分：计算最优温度
    print("\n正在计算最优工作温度...")
    optimal_temp, optimal_efficiency = find_optimal_temperature()
    print(f"最优温度: {optimal_temp:.1f} K")
    print(f"最大效率: {optimal_efficiency:.4f} ({optimal_efficiency*100:.2f}%)")
    
    # 第三部分：与实际白炽灯温度比较
    print("\n与实际白炽灯性能比较:")
    actual_temp = 2700  # 典型白炽灯灯丝温度
    actual_efficiency = calculate_visible_power_ratio(actual_temp)
    print(f"实际灯丝温度: {actual_temp} K")
    print(f"实际效率: {actual_efficiency:.4f} ({actual_efficiency*100:.2f}%)")
    print(f"效率差异: {(optimal_efficiency - actual_efficiency)*100:.2f}%")
    
    # 第四部分：绘制对比图，标记最优和实际温度点
    plt.figure(figsize=(10, 6))
    plt.plot(temps, effs, 'b-')  # 绘制效率曲线
    plt.plot(optimal_temp, optimal_efficiency, 'ro', markersize=8, label=f'Optimal: {optimal_temp:.1f} K')
    plt.plot(actual_temp, actual_efficiency, 'go', markersize=8, label=f'Actual: {actual_temp} K')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Visible Light Efficiency')
    plt.title('Incandescent Lamp Efficiency vs Temperature')
    plt.grid(True, alpha=0.3)  # 添加网格线
    plt.legend()  # 显示图例
    plt.savefig('optimal_temperature.png', dpi=300)  # 保存对比图
    plt.show()  # 显示对比图


if __name__ == "__main__":
    main()  # 程序入口点
