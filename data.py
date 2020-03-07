# encoding: utf-8
# @author kwp
# @created 2020-3-6

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms.functional import to_pil_image, to_tensor, to_grayscale
import torch
import numpy as np
from PIL import Image
import os

class ImageSet(Dataset):
    def __init__(self, ids_file, input_dir, output_dir):
        with open(ids_file) as f:
            self.ids = [name.strip() for name in f.readlines()]
        self.input_dir = input_dir
        self.output_dir = output_dir

    def process_img(self, img, out=False):
        w, h = img.size
        scal = 5
        newW, newH = w // scal, h // scal
        img = img.resize((newW, newH))
        if out:
            img = np.array(img, dtype='float32')   # 关键点！
        else:
            img = np.array(img)
        return ToTensor()(img)

    def __getitem__(self, item):
        id = self.ids[item]
        input_path = os.path.join(self.input_dir, id + ".jpg")
        output_path = os.path.join(self.output_dir, id + "_mask.gif")
        input = Image.open(input_path)
        output = Image.open(output_path)
        # print(output)
        # input = ToTensor()(self.process_img(input))
        # output = to_tensor(to_grayscale(self.process_img(output)))
        input = self.process_img(input)
        output = self.process_img(output, out=True)
        return input, output

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    in_dir = "data/kaggle/train"
    out_dir = "data/kaggle/train_masks/"
    train_id_file = "data/kaggle/train.txt"
    train_data = ImageSet(train_id_file, in_dir, out_dir)
    i, o = train_data[0]
    print(i)
    print(i.size())
    img = ToPILImage()(i)

    # BATCH_SIZE = 4
    # loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


