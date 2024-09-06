# 此代码保存CIFAR的图片。

import numpy as np
import pickle
import imageio
import matplotlib.pyplot as plt
import cv2


# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, "rb")
    dict = pickle.load(fo, encoding="latin1")
    fo.close()
    return dict


test_file = "./data/cifar-10-batches-py/test_batch"
# 显示测试集图片
dict_test = unpickle(test_file)
x_test = dict_test.get("data")
y_test = dict_test.get("labels")
dict_test = unpickle(test_file)
x_test = dict_test.get("data")
y_test = dict_test.get("labels")
image_m = np.reshape(x_test[1], (3, 32, 32))
r = image_m[0, :, :]
g = image_m[1, :, :]
b = image_m[2, :, :]
img23 = cv2.merge([r, g, b])
plt.figure()
plt.imshow(img23)
plt.show()

# 保存测试集图片
testXtr = unpickle(test_file)
for i in range(1, 10):  # 保存全部的使用for i in range(1, 10000):
    img = np.reshape(testXtr["data"][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = "./test_image/" + str(testXtr["labels"][i]) + "_" + str(i) + ".jpg"
    imageio.imsave(picName, img)  # , dpi=(600.0,600.0))
print("test_batch loaded.")
