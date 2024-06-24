import requests
from pydub import AudioSegment
from pydub.playback import play
import pyaudio
# 文字转语音
def txt_to_audio(text):
    res = requests.post('http://127.0.0.1:9966/tts', data={
    "text": text,
    "prompt": "",
    "voice": 11,
    "speed": 5,
    "temperature": 0.3,
    "top_p": 0.7,
    "top_k": 20,
    "refine_max_new_token": 384,
    "infer_max_new_token": 2048,
    "text_seed": 42,
    "skip_refine": 0,
    "custom_voice": 0
    })
    data = res.json()
    url = data["audio_files"][0]["url"]
    res = requests.get(url)
    with open('audio/audio.wav', 'wb') as f:
        f.write(res.content)
    # 加载音频文件
    audio = AudioSegment.from_wav("audio.wav")

    # 使用pyaudio播放音频并等待播放完毕
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(audio.sample_width),
                    channels=audio.channels,
                    rate=audio.frame_rate,
                    output=True)

    # 播放音频数据
    stream.write(audio.raw_data)

    # 等待播放完毕
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("播放完毕")