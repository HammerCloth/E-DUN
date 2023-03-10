import gc

import torch

import loss
import models
import mydata
import utility
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,2,3,5'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gc.collect()
torch.cuda.empty_cache()
if checkpoint.ok:
    loader = mydata.Data(args)
    model = models.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()
    checkpoint.done()
