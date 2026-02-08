"""
场景 1：串口测试 (Serial Port) —— 类似于“本地 API”
这是嵌入式最基础的通信方式。你可以把它想象成 web 测试里的 HTTP 请求，只不过我们要指定的是 COM 口（Windows）或 ttyUSB（Linux），而不是 URL。
"""

import serial
import time


def test_device_via_serial():
    # 1. 配置串口参数 (类似于配置 Base URL 和 Header)
    # 医疗设备通常会有说明书：波特率 9600, 数据位 8, 无校验位
    ser = serial.Serial(
        port='COM3',  # 端口号 (Windows 是 COMx, Linux 是 /dev/ttyUSBx)
        baudrate=9600,  # 传输速度
        timeout=2  # 读数据超时时间
    )

    if ser.is_open:
        print(f"✅ 成功连接到设备: {ser.port}")

    try:
        # 2. 发送指令 (Action)
        # Web 测试发 JSON，嵌入式通常发 十六进制(Hex) 或 字节流(Bytes)
        # 假设：0x01 是“查询”，0xAA 是“结束符”
        command = b'\x01\x00\x00\xAA'
        print(f"发送指令: {command.hex()}")
        ser.write(command)

        # 3. 接收响应 (Response)
        time.sleep(0.5)  # 给设备一点处理时间
        response = ser.read(1024)  # 读取缓冲区的数据

        print(f"收到响应: {response.hex()}")

        # 4. 断言 (Assert)
        # 假设设备正常应该回复以 0x02 开头
        if response.startswith(b'\x02'):
            print("✅ 测试通过：设备状态正常")
        else:
            print("❌ 测试失败：设备返回异常")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        ser.close()  # 记得像关闭文件一样关闭串口

# 运行 (需要实际连接设备，或者使用虚拟串口工具)
# test_device_via_serial()