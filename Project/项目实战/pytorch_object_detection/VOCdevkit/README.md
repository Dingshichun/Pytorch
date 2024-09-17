
## PASCAL VOC2012 数据集下载
- 下载后解压，复制 VOCdevkit 文件夹到想要的位置，然后根据数据集实际位置更改代码。

* Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar  
百度网盘下载链接：https://pan.baidu.com/s/1VCTJUZy_-k2WzXQZXhT3MA 
提取码：cpdd

* Pascal VOC2007 test数据集请参考：http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

* 如果不了解数据集或者想使用自己的数据集进行训练，可参考 bilibili：https://b23.tv/F1kSCK

## PASCAL VOC2012 数据集结构

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