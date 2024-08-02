import os
import math
import time
import datetime
from functools import reduce
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
import cv2
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from skimage.metrics import structural_similarity as ssimcalcu
from math import log10

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = './experiment/' + args.save
        else:
            self.dir = './experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            misc.imsave('{}{}.png'.format(filename, p), ndarr)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def save_fig(x, fig_name,srresult=None):
        # f, ax = plt.plots(1, 3, figsize=(30, 10))
        # ax[0].imshow(x, cmap=plt.cm.gray)
        # ax[0].set_title('LR', fontsize=30)
       
        # ax[1].imshow(pred, cmap=plt.cm.gray)
        # ax[1].set_title('SR', fontsize=30)
        # ax[1].set_xlabel("PSNR:{:.4f}\nSSIM:{:.4f}\nMSE:{:.4f}".format(srresult[0],srresult[1],srresult[2]),fontsize=20)

        # ax[2].imshow(y, cmap=plt.cm.gray)
        # ax[2].set_title('HR', fontsize=30)
        # f.savefig(fig_name)
        # plt.close()
        if srresult is not None:
            plt.figure()
            plt.imshow(x, cmap='gray')
            plt.xlabel("PSNR:{:.4f}\nSSIM:{:.4f}\nNMSE:{:.4f}".format(srresult[0],srresult[1],srresult[2]))

            plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.0)  # 保存图像到指定路径
            plt.close()
        else:
            plt.figure(figsize=(0.5, 0.5),dpi=600)
            plt.imshow(x, cmap='gray')
            plt.axis('off')
            plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.0)  # 保存图像到指定路径
            plt.close()

def calc_psnr(id, lr, sr, hr, FSsr=None, img_name=None, save=False, scale=1, savefile=None, ref=None):
    
    if FSsr is not None:
        FSsr = torch.fft.ifftshift((FSsr), dim=[2,3])
        sr = math.sqrt(FSsr.shape[2]*FSsr.shape[3]) * (torch.fft.ifftn(FSsr, dim=[2,3]))
        srmagnitude = torch.abs(sr) 
    else:
        srmagnitude = (sr[:, 0:1, :, :] ** 2 + sr[:, 1:2, :, :] ** 2).sqrt()
    
    lrmagnitude = (lr[:, 0:1, :, :] ** 2 + lr[:, 1:2, :, :] ** 2).sqrt()
    hrmagnitude = (hr[:, 0:1, :, :] ** 2 + hr[:, 1:2, :, :] ** 2).sqrt()
    lrcpu = lrmagnitude[0,0,:,:].cpu().numpy()
    hrcpu = hrmagnitude[0,0,:,:].cpu().numpy()
    srcpu = srmagnitude[0,0,:,:].cpu().numpy()
    if ref is not None:
        refmagnitude = (ref[:, 0:1, :, :] ** 2 + ref[:, 1:2, :, :] ** 2).sqrt()
        refcpu = refmagnitude[0,0,:,:].cpu().numpy()

    peak_signal = (hrmagnitude.max()-hrmagnitude.min()).item()
    mse = (srmagnitude - hrmagnitude).pow(2).mean().item()
    nmse = np.linalg.norm(hrcpu - srcpu) ** 2 / np.linalg.norm(srcpu) ** 2
    errormap = torch.abs(srmagnitude - hrmagnitude).cpu().numpy()
    errormap = errormap[0,0,:,:]
    psnr = 10*log10(peak_signal**2/mse)
    ssim = ssimcalcu(srcpu,hrcpu)

    if save and id%10 == 0:     
        pthroot_lr = os.path.join('./savefigresult/cartesian25','{:s}'.format('lr'))
        pthroot_sr = os.path.join('./savefigresult/cartesian25','{:s}'.format('sr'))
        pthroot_hr = os.path.join('./savefigresult/cartesian25','{:s}'.format('hr'))
        pthroot_metric = os.path.join('./savefigresult/cartesian25','{:s}'.format('metric'))       
        if not os.path.exists(pthroot_lr):
            os.makedirs(pthroot_lr)
        if not os.path.exists(pthroot_sr):
            os.makedirs(pthroot_sr)
        if not os.path.exists(pthroot_hr):
            os.makedirs(pthroot_hr)
        if not os.path.exists(pthroot_metric):
            os.makedirs(pthroot_metric)




        img_path_lr = os.path.join(pthroot_lr, 'results_{:s}.png'.format(img_name))
        img_path_sr = os.path.join(pthroot_sr, 'results_{:s}.png'.format(img_name))
        img_path_hr = os.path.join(pthroot_hr, 'results_{:s}.png'.format(img_name))
        img_path = os.path.join(pthroot_metric, 'results_{:s}.png'.format(img_name))


        srresult = [psnr,ssim,nmse]
        save_fig(lrcpu*255, img_path_lr)
        save_fig(srcpu*255, img_path_sr)
        save_fig(hrcpu*255, img_path_hr)

        save_fig(srcpu*255, img_path,srresult)

    return psnr,ssim,nmse

def save_map(feature_map, map_name):

        feature_map = feature_map[0].cpu().numpy()
        # 设置要显示的通道数，最多显示64个
        num_channels = feature_map.shape[0]
        cols = 8
        rows = num_channels // cols if num_channels % cols == 0 else num_channels // cols + 1
        # 创建子图
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

        for i in range(num_channels):
            ax = axes[i // cols, i % cols]
            ax.imshow(feature_map[i], cmap='jet')
            ax.axis('off')
            ax.set_title(f'Channel {i+1}')

        # 保存为 PNG 格式
        plt.tight_layout()
        plt.savefig(map_name, bbox_inches="tight", pad_inches=0.0)
        plt.close()

def save_map_mean(feature_map, map_name):

        feature_map = feature_map[0].cpu().numpy()
        # 设置要显示的通道数，最多显示64个
        feature_map = feature_map[18]    
        # average_feature_map = np.mean(feature_map, axis=0)

        feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))

        # 可视化平均特征图
        plt.figure(figsize=(8, 8))
        plt.imshow(feature_map,vmin=0.0, vmax=1.0, cmap='jet')
        # plt.colorbar()
        plt.axis('off')
        # 保存为 PNG 格式
        plt.savefig(map_name, bbox_inches="tight", pad_inches=0.0)
        plt.close()




def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma,
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    scheduler.step(args.start_epoch - 1)

    return scheduler

