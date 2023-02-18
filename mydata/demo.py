import os

import scipy.misc as misc
import torch.utils.data as data

import mydata.common as common


class Demo(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.name = 'Demo'
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = False

        self.filelist = []  # 存储文件名
        for f in os.listdir(args.dir_demo):
            if f.find('.png') >= 0 or f.find('.jpg') >= 0:
                self.filelist.append(os.path.join(args.dir_demo, f))
        self.filelist.sort()

    def __getitem__(self, idx):
        filename = os.path.split(self.filelist[idx])[-1]  # 'IMG_1177.jpg'
        filename, _ = os.path.splitext(filename)  # MG_1177
        lr = misc.imread(self.filelist[idx])  # 读取数据
        lr = common.set_channel([lr], self.args.n_colors)[0]  # 矫正维度
        lr = common.np2Tensor([lr], self.args.rgb_range)[0]  # 调整rgb，标准化
        return lr, -1, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
