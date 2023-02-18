from importlib import import_module

from torch.utils.data.dataloader import default_collate

from mydata.myDataLoader import MSDataLoader


class Data:
    def __init__(self, args):
        """ 直接构造loader并返回
        :param args: args
        """

        kwargs = {}
        if not args.cpu:  # 使用cpu进行计算
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        '''导入trainSet，并构造trainLoader返回'''
        self.loader_train = None
        # if not test only
        if not args.test_only:
            # 动态导入对应的dataset模块
            module_train = import_module('mydata.' + args.data_train.lower())
            # 生成trainset
            trainset = getattr(module_train, args.data_train)(args)
            # load trainset
            self.loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs
            )

        '''导入testSet，并构造testLoader并返回'''
        if args.data_test in ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']:
            if not args.benchmark_noise:
                module_test = import_module('mydata.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False)
            else:
                module_test = import_module('mydata.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(
                    args,
                    train=False
                )
        else:
            module_test = import_module('mydata.' + args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)
        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            **kwargs
        )
