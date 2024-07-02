import pyaudio
import wave
import keyboard
from datetime import datetime
# 控制麦克风录音, 并保存成wav文件
def Init():
    global recording_flag, FORMAT, CHANNELS, RATE, CHUNK
    # 定义一个全局标志来控制录音循环
    recording_flag = True
    # 录音参数
    FORMAT = pyaudio.paInt16  # 采样位数
    CHANNELS = 1  # 声道数
    RATE = 44100  # 采样频率
    CHUNK = 1024  # 缓冲区大小 

# 检查Caps Lock按键状态的回调函数
def check_caps_lock():
    return keyboard.is_pressed('caps lock') 

# 录音函数
def record_audio() -> str:
    global recording_flag
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []
    while check_caps_lock() and recording_flag:
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    audio.terminate()
    # 获取当前时间
    current_time = datetime.now()
    # 格式化时间戳
    timestamp = current_time.strftime("%Y%m%d%H%M%S%f")
    # 生成文件名
    filename = f"{timestamp}.wav" 
    # 保存录音结果到音频文件
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    return filename

def Start():
    global recording_flag  
    Init()
    try:
        while True:
            keyboard.wait('caps lock')
            yield record_audio()
    except KeyboardInterrupt:
        print("Ctrl+C detected")
    finally:
        recording_flag = False
        print("Exiting...")