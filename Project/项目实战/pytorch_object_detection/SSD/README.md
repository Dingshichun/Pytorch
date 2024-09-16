# SSD: Single Shot MultiBox Detector

## 环境配置：
* Python 3.6/3.7/3.8
* Pytorch 1.7.1
* pycocotools(Linux:```pip install pycocotools```; Windows:```pip install pycocotools-windows```)
* 最好使用 GPU 训练

## 文件结构：
```
├── src: 实现SSD模型的相关模块    
│     ├── resnet50_backbone.py   使用 resnet50 网络作为 SSD 的 backbone  
│     ├── ssd_model.py           SSD 网络结构文件 
│     ├── utils.py         训练过程中使用到的一些功能实现
│     └── nvidia_ssdpyt_fp32.pt  预训练权重，要先下载。 
├── train_utils: 训练验证相关模块（包括cocotools）  
├── my_dataset.py: 自定义 dataset 用于读取 VOC 数据集    
├── train_ssd300.py: 以 resnet50 做为 backbone 的 SSD 网络进行训练的脚本    
├── train_multi_GPU.py: 多 GPU 训练脚本    
├── predict_test.py: 使用训练好的模型进行预测    
├── pascal_voc_classes.json: pascal_voc 标签文件    
├── plot_curve.py: 用于绘制训练损失及验证集的 mAP
└── validation.py: 利用训练好的权重验证/测试数据的 COCO 指标，并生成 record_mAP.txt 文件
```

## 预训练权重下载地址（下载后放入src文件夹）：
* ResNet50+SSD: https://ngc.nvidia.com/catalog/models  
 `搜索ssd -> 找到SSD for PyTorch(FP32) -> download FP32 -> 解压文件`

* 如果找不到可通过百度网盘下载，链接：https://pan.baidu.com/s/1RXQ2pw0DLn9jFUNWxYkgfA ，提取码：cpdd 

## 数据集下载，使用的是PASCAL VOC2012数据集
- 下载解压，我的代码是将 VOCdevkit 文件夹复制到该项目的上一级目录。实际情况根据数据集位置更改代码。

* Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar  
百度网盘下载链接：https://pan.baidu.com/s/1VCTJUZy_-k2WzXQZXhT3MA 
提取码：cpdd

* Pascal VOC2007 test数据集请参考：http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

* 如果不了解数据集或者想使用自己的数据集进行训练，可参考 bilibili：https://b23.tv/F1kSCK

## 训练方法
* 确保提前准备好数据集 PASCAL VOC2012 以及 pycocotools 模块。

* 确保提前下载好对应预训练模型权重 nvidia_ssdpyt_fp32.pt

* 单 GPU 训练或 CPU ，直接使用 train_ssd300.py 训练脚本。

* 若要使用多 GPU 训练，使用 "python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py" 指令,nproc_per_node 参数为使用 GPU 数量

* 训练过程中保存的`results.txt`是每个 epoch 在验证集上的 COCO 指标，前 12个 值是 COCO 指标，后面两个值是训练平均损失以及学习率

## SSD 算法原理参考 bilibili
* https://www.bilibili.com/video/BV1fT4y1L7Gi

## 对 SSD 算法代码的分析参考 bilibili
* https://www.bilibili.com/video/BV1vK411H771/

## Resnet50 + SSD 算法框架图
![Resnet50 SSD](res50_ssd.png) 

## 注意事项
- **pycocotools 模块可能没有安装**，需要到官网安装所使用环境对应的版本，我安装的是 pycocotools-2.0.8-cp311-cp311-win_amd64.whl，因为所使用的环境中是 python3.11 ，系统是 Windows 。然后激活所使用的环境，使用"pip install D:\Edge下载\pycocotools-2.0.8-cp311-cp311-win_amd64.whl"将其安装到使用的环境中，**安装路径根据实际情况更改**。  

- train_utils\coco_eval.py 中的 `torch._six` 模块可能因为torchvision的版本不一样导致无法加载。解决办法是将 `import torch._six` 注释掉，添加以下代码：  
```  
int_classes = int
string_classes = str
```
然后将 `if isinstance(resFile, torch._six.string_classes):` 中的 `torch._six.string_classes` 改为 `str`。

- 在使用`predict.py`进行预测时，可能会出现报错：  
`AttributeError: 'FreeTypeFont' object has no attribute 'getsize' `，  
可能原因是环境中安装的 Pillow 版本太高，没有 `getsize` 这个属性，激活对应的环境后使用 `pip install Pillow==9.5` 安装具有该属性的版本即可。

- 由于训练比较耗时，最开始尝试训练时尽量将 `epochs` 设置得小一些，比如 1 或 2 ，看代码能否跑通。也可不训练，直接下载已经训练好的模型，使用 `predict_test.py` 进行预测，观察效果。