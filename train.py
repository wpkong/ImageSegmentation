# encoding: utf-8
# @author kwp
# @created 2020-3-6

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import ImageSet
from UNet import UNet
import logging
import os
import random

def train(net: UNet, train_ids_file_path: str, val_ids_file_path: str,
          in_dir_path: str, mask_dir_path: str, check_points: str,
          epochs=10, batch_size=4, learning_rate=0.1, device=torch.device("cpu")):
    train_data_set = ImageSet(train_ids_file_path, in_dir_path, mask_dir_path)

    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, num_workers=1)

    net = net.to(device)

    loss_func = nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.99)
    writer = SummaryWriter("tensorboard")
    g_step = 0

    for epoch in range(epochs):
        net.train()
        total_loss = 0

        with tqdm(total=len(train_data_set), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for step, (imgs, masks) in tqdm(enumerate(train_data_loader)):
                imgs = imgs.to(device)
                masks = masks.to(device)

                outputs = net(imgs)
                loss = loss_func(outputs, masks)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # record
                writer.add_scalar("Loss/Train", loss.item(), g_step)
                writer.flush()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])
                g_step += 1

                if g_step % 10 == 0:
                    writer.add_images('masks/origin', imgs, g_step)
                    writer.add_images('masks/true', masks, g_step)
                    writer.add_images('masks/pred', outputs > 0.5, g_step)
                    writer.flush()

        try:
            os.mkdir(check_points)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save(net.state_dict(),
                   check_points + f'CP_epoch{epoch + 1}.pth')
        logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


if __name__ == '__main__':
    in_dir = "data/kaggle/train"
    out_dir = "data/kaggle/train_masks"
    train_lst_path = "data/kaggle/train.txt"
    val_lst_path = "data/kaggle/val.txt"
    check_points = "check_points/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(F"Using {device}")
    net = UNet(in_channels=5, classes=1)

    load_path = ""
    if load_path != "":
        net.load_state_dict(
            torch.load(load_path, map_location=device)
        )

    try:
        train(net=net, train_ids_file_path=train_lst_path, val_ids_file_path=val_lst_path,
              in_dir_path=in_dir, mask_dir_path=out_dir, check_points=check_points,
              epochs=2, batch_size=1,learning_rate=0.1, device=device
              )
    except Exception as e:
        logging.error(e)
        torch.save(net.state_dict(), 'savednet.pth')
        logging.info("saved")

