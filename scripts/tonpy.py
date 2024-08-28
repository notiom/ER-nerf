import numpy as np
from scipy.io import wavfile

# 读取.wav文件
sample_rate, data = wavfile.read('/home/yanxl/Aigc/ER-nerf/data/test_audio/twgx.wav')

# 将数据保存为.npy文件
np.save('/home/yanxl/Aigc/ER-nerf/data/test_audio/scripts_gen.npy', data)
