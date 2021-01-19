# 基于Pytorch_Retinaface的车牌定位及关键点检测

本项目基于[Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)修改，完成车牌的定位及四个关键点检测（车牌的左上，右上，右下及左下角点），以此通过透视变换完成车牌的对齐，可使用mobilenet0.25或resnet50作为骨干网络进行实现。

## 使用
说明：本项目训练用数据集格式参考widerface进行制作

### 克隆工程
1. git clone https://github.com/Fanghc95/Plate-keypoints-detection.git

2. Pytorch version 1.1.0+ and torchvision 0.3.0+ are needed.

3. Codes are based on Python 3

### 准备数据

该项目数据集基于widerface制作，其目录格式如下

```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```

train为训练数据文件夹，val为验证数据文件夹，文件夹下images目录放入图像数据，txt文件为标签数据

txt标签格式如下：

```Shell
  # [image name]
  x y w h x1 y1 0.0 x2 y2 0.0 x3 y3 0.0 x4 y4 0.0 
```

每张图像数据占两行，第一行标识图像文件命，第二行(x,y)为车牌检测框左上角坐标，(w,h)为车牌检测框的宽，高。(x1,y1)-(x4,y4)依次表示车牌左上，右上，右下，左下四个角点的坐标

### 训练
本项目提供基于restnet50和mobilenet0.25为骨干网络的模型训练。这里提供原项目预训练的Mobilenet0.25模型[百度](https://pan.baidu.com/s/12h97Fy1RYuqMMIV-RpzdPg) 密码：fstq。下载的模型放在weights目录下。

1. 训练之前，可在 ``data/config.py and train.py``中对训练的一些参数进行修改，例如GPU数量，batch_size等参数。

2. 训练模型（例子图如下）:
  ```Shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50 or
  CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25
  ```
3. 模型将保存在``./weights``下


### 测试
```Shell
python detect.py --trained_model [weight_file] --network [mobile0.25 or resnet50] --input [path to test_image]
```
其他输入参数具体查看``./detect.py``，检测结果保存到``./res.jpg`
<p align="center"><img src="res/test1.jpg" width="570"\></p>
<p align="center"><img src="res/test.jpg" width="570"\></p>
<p align="center"><img src="res/test3.jpg" width="570"\></p>

### TensorRT
-[TensorRT](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)

## References
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
