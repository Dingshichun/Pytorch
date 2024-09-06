# LeNet实现CIFAR10图像分类
#### （1）文件简介
**model.py**：模型文件  
**train.py**：训练文件   
**predict.py**：预测文件  
**save_cifar_image**：将CIFAR10中的图像保存用来进行测试  
**Lenet.pth**：模型训练后保存的权重文件  
**代码练习.ipynb**：重新实现模型编写、训练、预测的notebook文件  
**test_image**：用来预测的图像  
**data**：CIFAR10数据集所在的文件夹  
#### （2）注意事项
- 要使用该脚本需要先下载数据集，并划分训练集和验证集，最后正确设置加载路径。 [CIFAR10数据集下载地址](https://pan.baidu.com/s/14lbav5J6qKCi7nzjAdOs7A
)，提取码：cpdd   
 
- 组织模型结构时，输入经过卷积层后需要将一些连续的维度进行展平之后才能输入全连接层，可在组织网络结构时使用torch.nn.Flatten()，也可在前向传播函数中使用view()函数。
