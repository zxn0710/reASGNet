# ASGNet (ASGNet_26_TCSVT) 复现文档

## 一、环境与硬件适配说明

**硬件差异：**

源代码默认使用 4 张 4090 显卡运行，因此源训练文件中的 `device_ids = [0, 1, 2, 3]`。

此处为单卡环境，已将训练文件修改为 `device_ids = [0]`，测试文件中也修改为 `os.environ["CUDA_VISIBLE_DEVICES"] = '0'`。

*（注：以上二者需按实际显卡数量调整）*

## 二、数据集准备

在 `train` 同级目录下新建 `data` 文件夹，并将下面下载好的 `traindataset` 和 `testdataset` 直接放入即可。

**traindataset下载链接：**
https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing

包含内容：Kvasir-SEG (900 张训练样本) 和 CVC-ClinicDB (550 张训练样本)。
文件大小：399.5MB

**testdataset下载链接：**
https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing

包含内容：CVC-300 (60 张样本)、CVC-ClinicDB (62 张样本)、CVC-ColonDB (380 张样本)、ETIS-LaribPolypDB (196 张样本)、Kvasir (100 张样本)。
文件大小：约 327.2 MB

## 三、环境配置

```bash
conda create -n asgnet_ python=3.8.12
conda activate asgnet_
cd /root/ASGNet-main/ASGNet_26_TCSVT
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt --no-deps
```

## 四、下载预训练文件

文件中有下载的相关代码，但速度较慢，此处使用 autodl 的学术加速下载：

```bash
source /etc/network_turbo
wget -O /root/.cache/torch/hub/checkpoints/resnet50_a1_0-14fe96d1.pth \
https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth
```

## 五、运行步骤

**1. 生成训练数据的图像边缘文件：**

```bash
python make_edges.py
```
*（更改训练数据集路径）*

**2. 训练：**

```bash
python Train_PS_ResNet50_edge1_new1.py
```
*（更改训练、验证集数据集路径）*

**3. 测试：**

```bash
python test.py
```
*（更改测试集路径、训练权重pth文件路径）*

> **注：** 原始 `test` 文件（注释部分）只生成分割图片，未输出指标；当前的指标输出部分，是通过导入了源文件里面的 `eval_functions` 评估函数来实现的。