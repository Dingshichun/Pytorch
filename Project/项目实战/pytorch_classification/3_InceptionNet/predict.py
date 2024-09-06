import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import GoogLeNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # load image
    img_path = "./test_img/rose_test.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    img = data_transform(img)
    # 增加图像数量维度N，原来是[C, H, W],增加后是[N, C, H, W]
    # 因为模型要求的输入数据维度是四维[N, C, H, W]
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = GoogLeNet(num_classes=5, aux_logits=False).to(device)

    # load model weights
    weights_path = "./InceptionNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(
        weights_path
    )
    missing_keys, unexpected_keys = model.load_state_dict(
        torch.load(weights_path, map_location=device), strict=False
    )

    # 预测时使用模型的验证模式。
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(
        class_indict[str(predict_cla)], predict[predict_cla].numpy()
    )
    plt.title(print_res)
    for i in range(len(predict)):
        print(
            "class: {:10}   prob: {:.3}".format(
                class_indict[str(i)], predict[i].numpy()
            )
        )
    plt.show()


if __name__ == "__main__":
    main()
