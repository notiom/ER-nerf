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


### 其他问题解决
1.若报错 libopenh264.so.5的问题，则将conda环境下的lib文件夹下的libopenh264.so改名为libopenh264.so.5


### 参考文档
<a href = "https://blog.csdn.net/matt45m/article/details/133700108">数字人解决方案</a>

### 有其他问题，欢迎联系作者 
->邮箱
meimeidemeng@outlook.com


