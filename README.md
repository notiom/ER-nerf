# ER-nerf
主要写er-nerf从零到一所有部署过程

### 原github链接 

```bash
git clone https://github.com/Fictionarry/ER-NeRF.git
```
### 安装依赖
1.git本项目源码
```bash
git clone https://github.com/notiom/ER-nerf.git
```
2.进入项目目录
```bash
cd ER-nerf
```
3.创建到自己服务器下的链接
```bash
conda create -p /home/xxx/.conda/envs/ernerf python=3.10 
```
4.使用conda安装cuda环境，这里不能使用conda安装torch，conda默认安装torch+cpu
```bash
conda install cudatoolkit=11.3 -c pytorch
```
5.使用pip安装torch1.12.1+cu113
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra -index-url https://download.pytorch.org/whl/cu113
```
6.安装其他依赖
```bash
pip install -r requirements.txt
```
7.安装pytorch3d
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
下载pytorch3d源码,如果下载不，按上面的百度网盘下载：链接：<a href = "https://pan.baidu.com/s/1xPFo-MQPWzkDMpLHhaloCQ">pytorch3d</a>
<br>
提取码：1422
<br>
若报g++版本问题，运行


8.安装tensorflow-gpu
```bash
pip install tensorflow-gpu==2.8.0
```
### 预训练准备
1.下载人脸解析模型79999_iter.pth放到data_utils/face_parsing/这个目录
```bash
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O data_utils/face_parsing/79999_iter.pth
```
2.
在data_utils/face_tracking目录下新建一个3DMM文件夹，并下载头部姿态估计模型到data_utils/face_tracking/3DMM/
```bash
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/exp_info.npy?raw=true -O data_utils/face_tracking/3DMM/exp_info.npy
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/keys_info.npy?raw=true -O data_utils/face_tracking/3DMM/keys_info.npy
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/sub_mesh.obj?raw=true -O data_utils/face_tracking/3DMM/sub_mesh.obj
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/topology_info.npy?raw=true -O data_utils/face_tracking/3DMM/topology_info.npy
```
3.下载01_MorphableModel.mat模型到data_utils/face_trackong/3DMM/目录
```bash
wget https://github.com/notiom/ER-nerf/releases/download/mat/01_MorphableModel.mat
```
在下载模型之后，运行
```bash
cd data_utils/face_tracking
python convert_BFM.py
```
若网络下载缓慢，->模型链接<a href = "https://pan.baidu.com/s/1z83r_2r4_5tsHDbC0ZlWeA">百度网盘下载链接</a>
<br>
4.这一步可以省略，因为在代码运行过程中也会自动下载，但是在运行时下载会很慢，建议下载
```bash
cd .cache/torch/hub/checkpoints
wget https://download.pytorch.org/models/resnet18-5c106cde.pth
wget https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
wget https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip
wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
```

### 数据集准备
1.获取一个5分钟左右的视屏，这里获取奥巴马的视屏
```bash
mkdir -p data/obama
wget https://github.com/YudongGuo/AD-NeRF/blob/master/dataset/vids/Obama.mp4?raw=true -O data/obama/obama.mp4
```
2.使用openface制作csv文件，openface推荐使用windows，openface网盘链接如下->
<br>
<a href = "https://pan.baidu.com/s/12MGt2mFpd6-zmiHL4BG0SQ">openface</a>
<br>
提取码为g105

打开OpenFace目录下的OpenFaceOffline.exe
 <div align=center>
<img src="/figs/fig1.png" height="400px" width="600px"/> 
</div>
选择要获取眨眼数据的视频
 <div align=center>
<img src="/figs/fig2.png" height="400px" width="600px"/> 
</div>
运行完之后，在processed目录下就有与视频名相同的csv文件
 <div align=center>
<img src="/figs/fig3.png" height="400px" width="600px"/> 
</div>

3.原始图片，此处我准备了一个ori_imgs(这一步可以省略，由process生成)
```bash

```
若想下载完整的数据集，可以去<a href ="https://tianchi.aliyun.com/dataset/155924">阿里云</a>找到rad-nerf-data.zip
<br>
4.生成其他的图片路径文件夹
```bash
python data_utils/process.py data/obama/obama.mp4
```
其中task的作用为:<br>
--task 1  #分离音频

--task 2  #生成aud_eo.npy

--task 3  #把视频拆分成图像

--task 4  #分割人像

--task 5  #提取背景图像

--task 6 #分割出身体部分

--task 7 #获取人脸landmarks lms文件 

--task 8 #获取人脸跟踪数据，这步要训练一个追踪模型，会很慢

--task 9 #保存所有数据
<br><br>
5.重命名
处理完成之后，把OpenFace处理出来的眨眼数据复制到当前目录，重新命名成au.csv，把原本的aud.npy重新命名成aud_ds.npy。
### 测试
```bash
python main.py data/obama/ --workspace trial_obama/ -O --test --ckpt trial_obama/checkpoints/ngp.pth   # head
python main.py data/obama/ --workspace trial_obama_torso/ -O --test --torso --ckpt trial_obama_torso/checkpoints/ngp.pth   # head+torso
```


测试结果为:
| setting    | PSNR   | LPIPS  | LMD   |
| ---------- | ------ | ------ | ----- |
| head       | 35.607 | 0.0178 | 2.525 |
| head+torso | 26.594 | 0.0446 | 2.550 |

### 其他问题解决
1.若报错 libopenh264.so.5的问题，则将conda环境下的lib文件夹下的libopenh264.so改名为libopenh264.so.5


### 参考文档
<a href = "https://blog.csdn.net/matt45m/article/details/133700108">数字人解决方案</a>

### 有其他问题，欢迎联系作者 
->邮箱
meimeidemeng@outlook.com


