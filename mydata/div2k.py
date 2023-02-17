import os

import srdata


class DIV2K(srdata.SRData):
    def __init__(self, args, train=True):
        super(DIV2K, self).__init__(args, train)

        # self.repeat = args.test_every // (args.n_train // args.batch_size)
        self.repeat = 6  # 重复次数

    def _scan(self):
        """获取hr 和 lr的路径列表"""
        list_hr = []
        list_lr = [[] for _ in self.scale]

        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>4}'.format(i)  # 将编号转化为4位的数字
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        """配置文件系统 apaph、dir_hr、dir_lr、ext"""
        self.apath = dir_data + '/DIV2K'
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        self.ext = '.png'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx
