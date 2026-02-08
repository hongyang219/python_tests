"""
示例 1：使用 PyVISA 自动化控制功率分析仪 (Power Analyzer)
这个脚本模拟了最核心的工作：设置仪器参数、触发测量、获取数据。
"""

import pyvisa
import time


def energy_test_automation():
    # 1. 初始化资源管理器 (VISA standard)
    rm = pyvisa.ResourceManager()

    # 2. 连接仪器 (假设通过 USB 或 TCP/IP)
    # 这里的地址在实验室中通常是具体的设备 ID
    try:
        instrument = rm.open_resource('USB0::0x0B21::0x0025::INSTR')
        instrument.write("*RST")  # 重置仪器到初始状态
        print(f"Connected to: {instrument.query('*IDN?')}")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # 3. 配置测量参数 (针对能效测试)
    instrument.write("VOLTage:RANGe 240")  # 设置电压量程
    instrument.write("CURRent:RANGe 5")  # 设置电流量程
    instrument.write("CONFigure:ENERgy:TIME 3600")  # 预设测试时长 1 小时

    # 4. 模拟在特定负载下开始记录
    print("Starting data collection...")
    instrument.write("INITiate")

    # 模拟读取 5 次瞬时功率
    for i in range(5):
        time.sleep(1)
        power = instrument.query("MEASure:POWer:AC?")
        print(f"Sample {i + 1}: Current Power = {power.strip()} Watts")

    # 5. 保存并断开
    instrument.write("STOP")
    rm.close()


if __name__ == "__main__":
    energy_test_automation()