## 代码使用简介

1. 下载好数据集，代码中默认使用的是花分类数据集，下载地址: [https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)

2. 在`train.py`脚本中将`--data-path`设置成解压后的`flower_photos`文件夹绝对路径。

3. 下载预训练权重，根据自己使用的模型下载对应预训练权重: https://pan.baidu.com/s/19hWNri5eOGI4Sfil29jn3A ，
提取码：cpdd

4. 在`train.py`脚本中将`--weights`参数设成下载好的预训练权重路径。

6. 在`predict.py`脚本中导入和训练脚本中同样的模型，并将`model_weight_path`设置成训练好的模型权重路径(默认保存在weights文件夹下)

7. 在`predict.py`脚本中将`img_path`设置成你自己需要预测的图片绝对路径。

8. 设置好权重路径`model_weight_path`和预测的图片路径`img_path`就能使用`predict.py`脚本进行预测了。

9. 如果要使用自己的数据集，请按照花分类数据集的文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的`num_classes`设置成你自己数据的类别数。
