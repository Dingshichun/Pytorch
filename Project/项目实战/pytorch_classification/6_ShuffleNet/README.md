# 代码使用简介

1. 下载花分类数据集，地址: [https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)

2. 下载预训练权重。这里用到的是 [shufflenetv2_x1.0下载链接](https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth)，在`train.py`脚本中将`--weights`参数设成下载好的预训练权重路径。

3. 设置好数据集的路径。由于已经将数据集下载到了data_set文件夹中，并且已经划分为训练集和验证集，所以直接使用datasets.ImageFolder()加载即可。

4. 在`predict.py`脚本中导入和训练脚本中同样的模型，并将`model_weight_path`设置成训练好的模型权重路径(默认保存在weights文件夹下)

5. 在`predict.py`脚本中将`img_path`设置成你自己需要预测的图片绝对路径。

6. 设置好权重路径`model_weight_path`和预测的图片路径`img_path`就能使用`predict.py`脚本进行预测了。

7. 如果要使用自己的数据集，请按照花分类数据集的文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的`num_classes`设置成你自己数据的类别数。
