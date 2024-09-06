# InceptionNet训练 

### （1）文件简介
**model.py**：模型文件  
**train.py**：训练文件  
**predict.py**：预测文件    
**class_indices.json**：图像类别字典，用数字代表实际类别。   
**split_data.py**：将下载的数据划分训练集和验证集   
**InceptionNet.pth**：训练后保存的模型权重  
**test_img**：用于测试模型的图像，predict.py文件使用。

### （2）数据集处理注意事项
* （1）点击链接下载花分类数据集 [https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)

* （2）在模型所在文件夹的上一级目录中创建data_set文件夹，然后在data_set文件夹下创建"flower_data"文件夹。
* （3）解压数据集，复制到flower_data文件夹下
* （4）执行"split_data.py"脚本自动将数据集划分成训练集train和验证集val 。
* **（5）划分训练、验证数据集时一定要注意各个文件中的路径是否正确。主要是split.py和train.py，不然无法运行，路径不对应的话需要进行修改。**  

生成的数据集目录如下：
```
├── flower_data   
       ├── flower_photos（解压的数据集文件夹，3670个样本）  
       ├── train（生成的训练集，3306个样本）  
       └── val（生成的验证集，364个样本） 
```
