# pytorch训练的模型部署到本地终端  

### （1）**文件简介**  
- **"素材文件"文件夹**：包含测试的图片banana1.jpg、视频video_4.mp4、ImageNet1000的类别索引素材文件imagenet_class_index.csv、中文字体SimHei.ttf。  
由于文件较占空间，[百度网盘自行下载](https://pan.baidu.com/s/1v9pFvc6-YJ7YZYInK7SrXA)，
提取码：cpdd

- **1_pytorch图像分类模型转ONNX.ipynb**：将官方预训练模型或者自己训练的模型转换为ONNX（open neural network exchange）格式，以供部署。

- **2_ONNX_Runtime预测单张图片的部署.ipynb**：部署ONNX模型预测单张图片。

- **3_ONNX_Runtime部署-摄像头和视频-英文.ipynb**：部署ONNX模型分类摄像头实时采集的图像以及分类保存的视频图像，在图像上以英文显示类别。文件4功能一样，只是以中文显示。

**ResNet18_ImageNet.onnx**：转换好的ONNX文件，可直接用于部署。

### （2）部署步骤
1. 下载好素材文件，保存在“素材文件”文件夹中。[百度网盘自行下载](https://pan.baidu.com/s/1v9pFvc6-YJ7YZYInK7SrXA)，
提取码：cpdd  

2. 执行"1_pytorch图像分类模型转ONNX.ipynb"文件将pytorch保存的模型转换为ONNX格式，得到的文件名为"ResNet18_ImageNet.onnx"。得到ONNX格式文件后可进行部署。

3. 执行"3_ONNX_Runtime部署-摄像头和视频-英文.ipynb"，可实时分类摄像头采集到的图像，也可分类保存的视频中的图像，并将类别及概率以英文显示在图像上。

4. "4_ONNX_Runtime部署-摄像头和视频-中文.ipynb"文件功能和步骤3的一样，区别只是以中文进行显示。