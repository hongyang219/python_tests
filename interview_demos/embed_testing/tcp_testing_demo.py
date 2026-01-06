"""
场景 2：TCP Socket 测试 —— 类似于“不带 HTTP 头的 API”
JD 里提到的 TCP/IP 知识就用在这里。医疗设备往往通过网线连到服务器。这里我们用 Python 原生的 socket 库。
"""

import socket

def test_heartbeat_tcp():
    # 模拟连接一台心电监护仪
    host = '192.168.1.50'
    port = 8080

    try:
        # 1. 建立连接 (类似于 TCP 三次握手)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(3) # 设置超时，防止卡死
        client.connect((host, port))
        print(f"✅ 已连接到设备 {host}:{port}")

        # 2. 发送握手包/心跳包
        # 假设协议规定：发送 "HELLO" 字符串
        msg = "HELLO".encode('utf-8')
        client.sendall(msg)

        # 3. 接收数据
        data = client.recv(1024)
        print(f"收到数据 (Raw): {data}")
        print(f"收到数据 (Decoded): {data.decode('utf-8', errors='ignore')}")

        # 4. 断言
        if b"OK" in data:
            print("✅ 心跳检测通过")
        else:
            print("❌ 响应不符合预期")

    except socket.timeout:
        print("❌ 连接或读取超时 - 这是一个典型的弱网或设备死机 Bug")
    except ConnectionRefusedError:
        print("❌ 无法连接 - 端口可能未开启")
    finally:
        client.close() # 四次挥手

