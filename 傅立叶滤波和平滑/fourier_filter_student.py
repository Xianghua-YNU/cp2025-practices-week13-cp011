import numpy as np
import matplotlib.pyplot as plt

# 读取数据
def load_data(filename):
    with open(filename, 'r') as file:
        data = np.loadtxt(file)
    return data

# 绘制数据
def plot_data(data, title="原始数据"):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('时间')
    plt.ylabel('道琼斯工业平均指数')
    plt.grid(True)
    plt.show()

# 傅立叶变换与滤波
def fourier_transform_and_filter(data, cutoff_percent=0.1):
    # 计算傅立叶变换
    coeff = np.fft.rfft(data)
    
    # 计算截止频率
    cutoff = int(len(coeff) * cutoff_percent)
    
    # 将后90%系数设为0
    coeff[cutoff:] = 0
    
    # 计算逆傅立叶变换
    filtered_data = np.fft.irfft(coeff)
    
    return filtered_data

# 主函数
def main():
    # 加载数据
    data = load_data('dow.txt')
    
    # 绘制原始数据
    plot_data(data, "原始数据")
    
    # 进行傅立叶变换和滤波
    filtered_data = fourier_transform_and_filter(data)
    
    # 绘制滤波后的数据
    plot_data(filtered_data, "滤波后的数据")

if __name__ == "__main__":
    main()
