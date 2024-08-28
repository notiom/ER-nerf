
import cv2
import subprocess
import dlib
import argparse
import os
args = parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
opt = parser.parse_args()

#参数列表
inputFile = opt.path

#最终生成的video的路径
outputFolder = os.path.join(inputFile,'../')
# 通常你会想要规范化路径，消除路径中的 '../'
outputFolder = os.path.normpath(outputFolder)

#最终生成的视频帧的路径
outputFamesFolder=os.path.join(outputFolder,'frames')
if(os.path.exists(outputFamesFolder) == False):
    os.makedirs(outputFamesFolder)

#video_name
name = inputFile.split('/')[-1].split('.')[0]
#裁剪的音频的输出
audio_output=os.path.join(outputFolder,'val_wavs')
if(os.path.exists(audio_output)== False):
    os.makedirs(audio_output)

# 初始化dlib的人脸检测器
detector = dlib.get_frontal_face_detector()
def get_face_coordinates(image):
    # cv读取的图片转为RGB格式
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 使用dlib的人脸检测器检测人脸
    detections = detector(rgb_image)
    
    if len(detections) > 0:
        face = detections[0]
        # 计算并返回人脸中心点
        center_x = (face.left() + face.right()) // 2
        center_y = (face.top() + face.bottom()) // 2
        return center_x, center_y
    else:
        return None

# 读取视频
video_capture = cv2.VideoCapture(inputFile)
# 获取第一帧的人脸坐标
ret, first_frame = video_capture.read()
face_coords = get_face_coordinates(first_frame)

if face_coords is not None:
    center_x, center_y = face_coords
    print("Center coordinates of the first detected face:", center_x, center_y)
else:
    print("No face detected in the first frame.")


targetWH=512
crop_size = targetWH//2  
#crop_size = 512 
start_x = max(center_x - crop_size, 0)
start_y = max(center_y - crop_size, 0)

#根据第一帧的人脸坐标信息,逐帧进行裁剪
frame_number = 0
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    cropped_image = frame[start_y:start_y + targetWH, start_x:start_x + targetWH]
    frame_filename = f"{outputFamesFolder}/frame_{frame_number:04d}.png"
    cv2.imwrite(frame_filename, cropped_image)
    frame_number += 1

video_capture.release()

# 提取视频的音频
cmd = [
    "ffmpeg",
    "-i",inputFile,
    '-f','wav',
    '-ar','16000',
    '-y',f"{audio_output}/{name}_audio.wav"
]

out = subprocess.run(cmd,\
        stdout=subprocess.PIPE,\
        stderr=subprocess.STDOUT,\
        text=True)
targetFps=25
frame_pattern = f"{outputFamesFolder}/frame_%04d.png"

#把裁剪人脸后的视频帧和音频无损合并为 最终的视频
cmd =[
    "ffmpeg",
    "-i",frame_pattern,
    "-i",f"{audio_output}/{name}_audio.wav",
    "-c:v","libx264",
    "-framerate", str(targetFps),
    '-pix_fmt', 'yuv444p',
    "-y",f"{outputFolder}/{name}_face_crop.mp4"
]

out = subprocess.run(cmd)