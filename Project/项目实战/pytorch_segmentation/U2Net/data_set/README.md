## DUTS数据集下载
链接：https://pan.baidu.com/s/1cop-GSOkWRDSjsDNkgW9rA ，提取码：cpdd

- 其中DUTS-TR为训练集，DUTS-TE是测试（验证）集，数据集目录结构应如下设置：
```
data_set
├── DUTS-TR
│      ├── DUTS-TR-Image: 该文件夹存放所有训练集的图片
│      └── DUTS-TR-Mask: 该文件夹存放对应训练图片的GT标签（Mask蒙板形式）
│
└── DUTS-TE
       ├── DUTS-TE-Image: 该文件夹存放所有测试（验证）集的图片
       └── DUTS-TE-Mask: 该文件夹存放对应测试（验证）图片的GT标签（Mask蒙板形式）
```

- 注意训练或者验证过程中，将`--data-path`指向`DUTS-TR`所在根目录，也就是上一级目录。