# 基于Pytorch_Retinaface的车牌定位及关键点检测

本项目基于[Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)修改，完成车牌的定位及四个关键点检测（车牌的左上，右上，右下及左下角点），以此通过透视变换完成车牌的对齐，可使用mobilenet0.25或resnet50作为骨干网络进行实现。

## 使用

说明：本项目训练用数据集格式参考widerface进行制作

###克隆工程

1. git clone https://github.com/Fanghc95/Plate-keypoints-detection.git

2. Pytorch version 1.1.0+ and torchvision 0.3.0+ are needed.

3. Codes are based on Python 3

###准备数据

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
##### Data
1. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.

2. Download annotations (face bounding boxes & five facial landmarks) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

3. Organise the dataset directory as follows:


ps: wider_val.txt only include val file names but not label information.

##### Data1
We also provide the organized dataset we used as in the above directory structure.

Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

## Training
We provide restnet50 and mobilenet0.25 as backbone network to train model.
We trained Mobilenet0.25 on imagenet dataset and get 46.58%  in top 1. If you do not wish to train the model, we also provide trained model. Pretrain model  and trained model are put in [google cloud](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) and [baidu cloud](https://pan.baidu.com/s/12h97Fy1RYuqMMIV-RpzdPg) Password: fstq . The model could be put as follows:
```Shell
  ./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
```
1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in ``data/config.py and train.py``.

2. Train the model using WIDER FACE:
  ```Shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50 or
  CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25
  ```


## Evaluation
### Evaluation widerface val
1. Generate txt file
```Shell
python test_widerface.py --trained_model weight_file --network mobile0.25 or resnet50
```
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)
```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```
3. You can also use widerface official Matlab evaluate demo in [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)
### Evaluation FDDB

1. Download the images [FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) to:
```Shell
./data/FDDB/images/
```

2. Evaluate the trained model using:
```Shell
python test_fddb.py --trained_model weight_file --network mobile0.25 or resnet50
```

3. Download [eval_tool](https://bitbucket.org/marcopede/face-eval) to evaluate the performance.

<p align="center"><img src="curve/1.jpg" width="640"\></p>

## TensorRT
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
