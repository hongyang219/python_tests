"""
场景 3：解析二进制数据 (struct 库) —— 嵌入式的“JSON 解析”
这是 Web 转嵌入式最大的痛点。Web 的返回值是 { "temperature": 36.5 }，一看就懂。 嵌入式返回的是 0x01 0x24 0x01 0x6D，这谁看得懂？
这就需要用到 Python 的 struct 库来解码。
假设协议文档规定：
第 1 个字节：设备 ID
第 2-3 个字节：体温 (整数，需除以 10)
第 4 个字节：电量百分比
"""

import struct

raw_data_valid = b'\x01\x01\x6d\x5a'
raw_data_invalid = b'\x01\x01\x67\x5a'


def parse_medical_data(raw_data):
    # 模拟从串口或网络收到的一串“乱码” (二进制数据)
    # Hex: 01 01 6D 5A -> 对应十进制：ID=1, Temp=365, Battery=90

    print(f"原始数据 (Hex): {raw_data.hex()}")

    try:
        # 这里的 '>H B' 是核心魔法 (Format String)
        # > : 大端序 (Big Endian)，嵌入式常用网络字节序
        # B : Unsigned Char (1字节) -> 取出 ID
        # H : Unsigned Short (2字节) -> 取出体温
        # B : Unsigned Char (1字节) -> 取出电量

        device_id, temp_raw, battery = struct.unpack('>B H B', raw_data)

        # 业务逻辑转换
        temperature = temp_raw / 10.0

        print("------ 解析结果 ------")
        print(f"设备 ID: {device_id}")
        print(f"当前体温: {temperature} °C")
        print(f"剩余电量: {battery} %")

        # 断言
        # if 36.0 <= temperature <= 37.5:
        #     print("✅ 体温正常")
        # else:
        #     print("⚠️ 体温异常")
        #
        assert 36.0 <= temperature <= 37.5, "⚠️ 体温异常"
        print("✅ 体温正常")

    except struct.error as e:
        print(f"数据包格式错误: {e}")


if __name__ == "__main__":
    parse_medical_data(raw_data_valid)
    parse_medical_data(raw_data_invalid)