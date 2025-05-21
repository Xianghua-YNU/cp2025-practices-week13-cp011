import numpy as np
import matplotlib.pyplot as plt

# 任务1: 数据加载与可视化
def load_data(file_path):
    data = np.loadtxt(file_path)
    return data

def plot_time_series(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Dow Jones Index')
    plt.title('Dow Jones Index Over Time')
    plt.xlabel('Time')
    plt.ylabel('Index Value')
    plt.legend()
    plt.show()

# 任务2: 傅立叶变换与滤波
def fourier_transform(data):
    coeff = np.fft.rfft(data)
    return coeff

def filter_data(coeff, cutoff_percent=0.1):
    cutoff = int(len(coeff) * cutoff_percent)
    coeff[cutoff:-cutoff] = 0  # 保留前10%系数，其余设为0
    filtered = np.fft.irfft(coeff)
    return filtered

# 主函数
def main():
    file_path = 'dow.txt'  # 假设数据文件名为dow.txt
    data = load_data(file_path)
    plot_time_series(data)
    
    coeff = fourier_transform(data)
    filtered_data = filter_data(coeff)
    
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Original Data')
    plt.plot(filtered_data, label='Filtered Data', linestyle='--')
    plt.title('Original vs Filtered Data')
    plt.xlabel('Time')
    plt.ylabel('Index Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

