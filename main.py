import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import os
from thop import profile
from thop import clever_format
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)       ## setting the log and the train information
    if checkpoint.ok:
        loader = data.Data(args)                ## data loader
        model = model.Model(args, checkpoint)

        dummy_input = torch.randn(1, 2, 128, 128).to('cuda:0')
        
        flops, params = profile(model, (dummy_input,dummy_input,dummy_input,None,1))
        
        print('FLOPs: ', flops, 'params: ', params)
        
        print('FLOPs: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
     
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()


