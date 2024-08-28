from pydub import AudioSegment

# 读取音频文件
audio = AudioSegment.from_file("/home/yanxl/Aigc/ER-nerf/data/test_audio/twgx.wav", format="wav")

# 定义裁剪的起始时间和结束时间（单位：毫秒）
start_time = 0 #0秒
end_time = 10000   # 10秒

# 裁剪音频
cropped_audio = audio[start_time:end_time]

# 保存裁剪后的音频文件
cropped_audio.export("/home/yanxl/Aigc/ER-nerf/data/test_audio/aud.wav", format="wav")
