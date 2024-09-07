# MobileNet-V2 训练和测试
### （1）训练步骤
1. 首先下载mobilenet-v2的预训练权重。链接：<https://download.pytorch.org/models/mobilenet_v2-b0353104.pth>  

2. 冻结预训练权重的特征提取层，只训练全连接层的参数。

### （2）注意事项
- 数据集的路径问题，结合自己数据集所在位置修改路径。  

- 需要先运行train.py得到用于测试的模型之后才能运行predict.py进行测试。