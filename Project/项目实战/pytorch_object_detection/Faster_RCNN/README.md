# Faster R-CNN

## 该项目主要是来自pytorch官方torchvision模块中的源码
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection

## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.7.1(注意：必须是 1.6.0 或以上，因为使用官方提供的混合精度训练 1.6.0 后才支持)
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`)
* Ubuntu 或 Centos (不建议 Windows )
* 最好使用 GPU 训练
* 详细环境配置见`requirements.txt`

## 文件结构：
```
  ├── backbone: 特征提取网络，根据自己的要求选择。
  ├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
  ├── train_utils: 训练验证相关模块（包括cocotools）
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train_mobilenet.py: 以MobileNetV2做为backbone进行训练
  ├── train_resnet50_fpn.py: 以resnet50+FPN做为backbone进行训练
  ├── train_multi_GPU.py: 多GPU训练
  ├── predict.py: 使用训练好的权重进行预测
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── pascal_voc_classes.json: pascal_voc标签文件
```

## 预训练权重下载地址（下载后放入backbone文件夹中）：
* MobileNetV2 weights(下载后重命名为 `mobilenet_v2.pth` ，然后放到 `bakcbone` 文件夹下): https://download.pytorch.org/models/mobilenet_v2-b0353104.pth

* Resnet50 weights(下载后重命名为 `resnet50.pth` ，然后放到 `bakcbone` 文件夹下): https://download.pytorch.org/models/resnet50-0676ba61.pth

* ResNet50+FPN weights(下载后重命名为 `fasterrcnn_resnet50_fpn` ，然后放到 `bakcbone` 文件夹下): https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

* 注意，下载的预训练权重记得要重命名，比如在 train_resnet50_fpn.py 中读取的是 `fasterrcnn_resnet50_fpn.pth` 文件，不是 `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` 。
 
 
## PASCAL VOC2012数据集
* Pascal VOC2012 train/val 数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

* 如果不了解数据集或者想使用自己的数据集进行训练，请参考 bilibili：https://b23.tv/F1kSCK

* 使用 ResNet50+FPN 以及迁移学习在 VOC2012 数据集上得到的权重: 链接:https://pan.baidu.com/s/1ifilndFRtAV5RDZINSHj5w 提取码:dsz8

* PASCAL VOC2012 数据集结构
```
VOCdevkit
    └── VOC2012
         ├── Annotations               所有的图像标注信息(XML文件)
         ├── ImageSets    
         │   ├── Action                人的行为动作图像信息
         │   ├── Layout                人的各个部位图像信息
         │   │
         │   ├── Main                  目标检测分类图像信息
         │   │     ├── train.txt       训练集(5717)
         │   │     ├── val.txt         验证集(5823)
         │   │     └── trainval.txt    训练集+验证集(11540)
         │   │
         │   └── Segmentation          目标分割图像信息
         │         ├── train.txt       训练集(1464)
         │         ├── val.txt         验证集(1449)
         │         └── trainval.txt    训练集+验证集(2913)
         │ 
         ├── JPEGImages                所有图像文件
         ├── SegmentationClass         语义分割png图（基于类别）
         └── SegmentationObject        实例分割png图（基于目标）
```

## 训练方法
* 确保提前准备好数据集

* 确保提前下载好对应预训练模型权重

* 若要训练 mobilenetv2+fasterrcnn ，直接使用 train_mobilenet.py 训练脚本

* 若要训练 resnet50+fpn+fasterrcnn ，直接使用 train_resnet50_fpn.py 训练脚本

* 若要使用多 GPU 训练，使用 `python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py` 指令, `nproc_per_node` 参数为使用GPU数量

* 如果想指定使用哪些GPU设备可在指令前加上 `CUDA_VISIBLE_DEVICES=0,3` (例如我只要使用设备中的第1块和第4块GPU设备)

* `CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py`

## 注意事项1
- **pycocotools 模块可能没有安装**，需要到官网安装所使用环境对应的版本，我安装的是 pycocotools-2.0.8-cp311-cp311-win_amd64.whl，因为所使用的环境中是 python3.11 ，系统是 Windows 。然后激活所使用的环境，使用"pip install D:\Edge下载\pycocotools-2.0.8-cp311-cp311-win_amd64.whl"将其安装到使用的环境中，**安装路径根据实际情况更改**。  

- train_utils\coco_eval.py 中的 `torch._six` 模块可能因为 torchvision 的版本不一样导致无法加载。解决办法是将 `import torch._six` 注释掉，添加以下代码：  
```  
int_classes = int
string_classes = str
```
然后将 `if isinstance(resFile, torch._six.string_classes):` 中的 `torch._six.string_classes` 改为 `str`。

- 在使用`predict.py`进行预测时，可能会出现报错：  
`AttributeError: 'FreeTypeFont' object has no attribute 'getsize' `，  
可能原因是环境中安装的 Pillow 版本太高，没有 `getsize` 这个属性，激活对应的环境后使用 `pip install Pillow==9.5` 安装具有该属性的版本即可。

- 由于训练比较耗时，最开始尝试训练时尽量将 `epochs` 设置得小一些，比如 1 或 2 ，看代码能否跑通。也可不训练，直接下载已经训练好的模型，使用 `predict_test.py` 进行预测，观察效果。
## 注意事项2
* 在使用训练脚本时，注意要将`--data-path`(VOC_root)设置为自己存放 `VOCdevkit` 文件夹所在的 **根目录** 。路径设置不是固定的，根据实际设置即可。

* 由于带有 FPN 结构的 Faster RCNN 很吃显存，如果 GPU 的显存不够(如果 batch_size 小于8的话)建议在 create_model 函数中使用默认的 norm_layer ，  
即不传递 norm_layer 变量，默认去使用 FrozenBatchNorm2d (即不会去更新参数的bn层),使用中发现效果也很好。

* 训练过程中保存的 `results.txt` 是每个 epoch 在验证集上的 COCO 指标，前12个值是 COCO 指标，后面两个值是训练平均损失以及学习率
* 在使用预测脚本时，要将 `train_weights` 设置为你自己生成的权重路径。

* 使用 validation 文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改 `--num-classes`、`--data-path` 和 `--weights-path` 即可，其他代码尽量不要改动

## 如果对 Faster RCNN 原理不是很理解可参考 bilibili
* https://b23.tv/sXcBSP

## 对 Faster RCNN 代码的分析可参考 bilibili
* https://b23.tv/HvMiDy

## Faster RCNN 框架图
![Faster R-CNN](fasterRCNN.png) 
