import os

import imageio
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image

"""
需要将hr图像为2的倍数
"""


def imread(dir):
    x = imageio.imread(dir)
    img = torch.from_numpy(x.transpose((2, 0, 1)))
    return img


def interpolate(scale_factor, batch):
    if scale_factor != 1:
        batch = nn.functional.interpolate(batch, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    batch = batch.permute(0, 2, 3, 1)
    return batch


def getImgList(data_dir):
    img_list = []
    for i in os.listdir(data_dir):
        img_list.append(data_dir + "/" + i)
    return img_list


def creatDataset(data_dir):
    """1. 先修改尺寸，然后在下采样"""

    hr_dir = 'HDT_train_HR'
    lr_dir = 'HDT_train_LR_bicubic'
    source_dir = 'HDT_source'

    ext = '.png'

    '''step-1 对source中的图像进行筛选与裁剪，存储到hr中'''
    size = (100, 100)
    img_dir_list = getImgList(data_dir + '/' + source_dir)

    hr_img_list = []
    for i in img_dir_list:
        img = Image.open(i)
        # 筛选图像
        if img.size[0] >= size[0] and img.size[1] >= size[1]:
            crop_img = F.center_crop(img, size)
            hr_img_list.append(crop_img)

    for i, hr in enumerate(hr_img_list):
        filename = '{:0>4}'.format(i) + ext
        imageio.imsave(data_dir + "/" + hr_dir + "/" + filename, hr)

    '''step-2 将hr文件进行下采样并存储'''
    img_dir_list = getImgList(data_dir + '/' + hr_dir)

    img_list = []
    for i in img_dir_list:
        img_list.append(imread(i))
    # 整合为batch
    batch = torch.stack(img_list).float()
    # 进行下采样
    batch_x2 = interpolate(0.5, batch)
    batch_x4 = interpolate(0.25, batch)
    # 存储
    for i in range(batch_x2.shape[0]):
        filename = '{:0>4}'.format(i)
        filename = 'X{}/{}x{}{}'.format(2, filename, 2, ext)
        imageio.imsave(data_dir + "/" + lr_dir + "/" + filename, batch_x2[i].int())

    for i in range(batch_x4.shape[0]):
        filename = '{:0>4}'.format(i)
        filename = 'X{}/{}x{}{}'.format(4, filename, 4, ext)
        imageio.imsave(data_dir + "/" + lr_dir + "/" + filename, batch_x4[i].int())


creatDataset("../data/HDT")
