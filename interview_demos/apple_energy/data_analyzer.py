"""
示例 2：数据处理与自动化报告 (Pandas + Matplotlib)
硬件测试会产生大量的 CSV 数据。你的优势是能用 Python 快速找出功耗峰值，而不是去翻 Excel。
"""

import pandas as pd
import matplotlib.pyplot as plt


def analyze_power_log(file_path):
    # 读取实验室仪器生成的原始 CSV 数据
    df = pd.read_csv(file_path)

    # 计算总能量消耗 (Wh) = 平均功率 * 时间
    avg_power = df['Power(W)'].mean()
    duration_hours = len(df) / 3600  # 假设采样率为 1Hz
    total_energy = avg_power * duration_hours

    # 自动识别异常功耗峰值 (Spikes)
    # 比如功耗超过平均值的 2 倍定义为异常
    threshold = avg_power * 2
    anomalies = df[df['Power(W)'] > threshold]

    # 可视化输出 (面试时可以展示你如何一键生成专业报告)
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time(s)'], df['Power(W)'], label='Power Profile')
    plt.axhline(y=avg_power, color='r', linestyle='--', label='Average Power')
    plt.title("Product Energy Efficiency Profile")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.savefig('test_report.png')

    print(f"Analysis Complete: Total Energy = {total_energy:.4f} Wh")
    print(f"Found {len(anomalies)} potential power anomalies to investigate.")

# analyze_power_log('lab_data.csv')