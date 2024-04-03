# ER-nerf
主要写er-nerf从零到一所有部署过程

### 原github链接 

```bash
git clone https://github.com/Fictionarry/ER-NeRF.git
```
### 安装依赖
1.it本项目源码
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
<div>
<span>下载pytorch3d源码,如果下载不，按上面的百度网盘下载：链接：<a href = "https://pan.baidu.com/s/1xPFo-MQPWzkDMpLHhaloCQ">pytorch3d</a>
<br>
提取码：1422</span>
</div>
<br>
8.安装tensorflow-gpu
```bash
pip install tensorflow-gpu==2.8.0
```


