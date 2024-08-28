import os
import argparse
from ffmpy import FFmpeg
 
 
# 视频添加音频
def video_add_audio(video_path: str, audio_path: str):
    _ext_video = os.path.basename(video_path).strip().split('.')[-1]
    _ext_audio = os.path.basename(audio_path).strip().split('.')[-1]
    if _ext_audio not in ['mp3', 'wav']:
        raise Exception('audio format not support')
    _codec = 'copy'
    if _ext_audio == 'wav':
        _codec = 'aac'
        
    i = 0
    while True:
        result = os.path.join(
        audio_path, '../', 'output_{}.{}'.format(i,_ext_video))
        # 通常你会想要规范化路径，消除路径中的 '../'
        result = os.path.normpath(result) 
        if(os.path.exists(result) is False):
            break
        i +=1      
    ff = FFmpeg(
        inputs={video_path: None, audio_path: None},
        outputs={result: '-map 0:v -map 1:a -c:v copy -c:a {} -shortest'.format(_codec)})
    print(ff.cmd)
    ff.run()
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=str, default="/home/yanxl/Aigc/ER-nerf/Myworkspace/trial_torso_shangnan/results/ngp_ep0073.mp4", help="input .mp4"
    )
    parser.add_argument("--wav", type=str, default='/home/yanxl/Aigc/ER-nerf/data/test_audio/shangnan.wav', help="input .wav")
    args = parser.parse_args()
    print(video_add_audio(args.video, 
                          args.wav))