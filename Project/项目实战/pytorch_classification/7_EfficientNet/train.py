import os
import math
import argparse
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torch.optim.lr_scheduler as lr_scheduler

from model import efficientnet_b0 as create_model
from utils import train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print(
        'Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/'
    )
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    img_size = {
        "B0": 224,
        "B1": 240,
        "B2": 260,
        "B3": 300,
        "B4": 380,
        "B5": 456,
        "B6": 528,
        "B7": 600,
    }
    num_model = "B0"

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size[num_model]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(img_size[num_model]),
                transforms.CenterCrop(img_size[num_model]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # get data root path
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./.."))
    # flower data set path
    image_path = os.path.join(data_root, "data_set", "flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "train"), transform=data_transform["train"]
    )
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    batch_size = args.batch_size
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
    )  # number of workers
    print("Using {} dataloader workers every process".format(nw))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw
    )

    validate_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "val"), transform=data_transform["val"]
    )
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw
    )

    print(
        "using {} images for training, {} images for validation.".format(
            train_num, val_num
        )
    )

    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {
                k: v
                for k, v in weights_dict.items()
                if model.state_dict()[k].numel() == v.numel()
            }
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = (
        lambda x: ((1 + math.cos(x * math.pi / args.epoches)) / 2) * (1 - args.lrf)
        + args.lrf
    )  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # epoch范围是[1, args.epoches]的整数
    for epoch in range(1, args.epoches + 1):
        # train
        mean_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
        )

        scheduler.step()

        # validate
        acc = evaluate(model=model, data_loader=validate_loader, device=device)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # 只保存靠近args.epoches中间位置的和最后一个epoch训练的模型。
        if epoch % (args.epoches / 2) == 0:
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--epoches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)

    # download model weights
    # 链接：https://pan.baidu.com/s/1s7F-8XBXS73RqvDFdH1eww ，提取码：cpdd
    parser.add_argument(
        "--weights",
        type=str,
        default="./efficientnetb0.pth",
        help="initial weights path",
    )
    parser.add_argument("--freeze-layers", type=bool, default=False)
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu)"
    )

    opt = parser.parse_args()

    main(opt)
