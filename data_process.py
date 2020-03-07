# encoding: utf-8
# @author kwp
# @created 2020-3-6

import os
import random

def make_train_and_validation_set_list():
    in_dir = "data/kaggle/train/"
    train_lst_path = "data/kaggle/train.txt"
    val_lst_path = "data/kaggle/val.txt"
    train_val_lst_path = "data/kaggle/trainval.txt"

    lst = [name.split(".")[0] for name in os.listdir(in_dir)]
    random.shuffle(lst)
    wall = int(len(lst) * 0.90)
    with open(train_lst_path, "w") as f:
        for name in lst[0: wall]:
            f.write(name + "\n")

    with open(val_lst_path, "w") as f:
        for name in lst[wall:]:
            f.write(name + "\n")

    with open(train_val_lst_path, "w") as f:
        for name in lst:
            f.write(name + "\n")

if __name__ == '__main__':
    make_train_and_validation_set_list()