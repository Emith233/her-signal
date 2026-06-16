#!/usr/bin/env python3
"""LSB 提取器 - 从视频帧中还原隐藏的文字"""
import struct, subprocess, sys

def extract(video_path):
    # 用 ffmpeg 取第一帧的原始数据
    cmd = ["ffmpeg", "-y", "-i", video_path, "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw, _ = proc.communicate()
    if not raw:
        print("读取视频失败")
        return

    # 提取所有 RGB 通道的 LSB
    bits = []
    for i in range(0, len(raw), 3):
        bits.append(raw[i] & 1)       # R
        bits.append(raw[i+1] & 1)     # G
        bits.append(raw[i+2] & 1)     # B

    # 还原字節
    data = bytearray()
    for j in range(0, len(bits) - 7, 8):
        byte = 0
        for k in range(8):
            byte = (byte << 1) | bits[j + k]
        data.append(byte)

    # 解析长度前綴 + 内容
    length = struct.unpack('>I', data[:4])[0]
    if length <= 0 or length > len(data) - 4:
        print("未检测到有效数据")
        return

    text = data[4:4+length].decode('utf-8')
    print(text)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"用法: python3 {sys.argv[0]} <视频文件>")
    else:
        extract(sys.argv[1])
