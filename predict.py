# encoding: utf-8
# @author kwp
# @created 2020-3-6

from torchvision.transforms import ToTensor, ToPILImage
import torch
from PIL import Image
from UNet import UNet


def predict(load_path, image_path, scal=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(in_channels=3, classes=1)
    net.load_state_dict(
        torch.load(load_path, map_location=device)
    )
    img = Image.open(image_path)
    w, h = img.size

    newW, newH = w // scal, h // scal
    img = img.resize((newW, newH))
    img = ToTensor()(img)
    img = img.unsqueeze(0)

    masks = net(img)
    masks = (masks >= 0.5)
    out_img = img * masks  # 添加遮罩
    out_img = out_img.squeeze()
    out_img = ToPILImage()(out_img)
    return out_img


if __name__ == '__main__':
    load_path = "CP_epoch3.pth"
    img_path = "data/kaggle/train/0cdf5b5d0ce1_04.jpg"
    predict(load_path, img_path, 4).show()
