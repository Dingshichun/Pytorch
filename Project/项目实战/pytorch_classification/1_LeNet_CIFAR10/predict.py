import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    net = LeNet()
    net.load_state_dict(torch.load("Lenet.pth"))

    im = Image.open("./test_image/0_3.jpg")
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        # predict = torch.max(outputs, dim=1)[1].numpy()
        # 这里应该是直接自动得到输出最大值的索引indices，
        # 而不是每次预测不同图片都要将索引改为图片对应的值。
        predict = torch.max(outputs, dim=1).indices.numpy()
    print("这张图像是：", classes[int(predict)])


if __name__ == "__main__":
    main()
