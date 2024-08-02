import random
import numpy as np
import skimage.color as sc
import torch
from torchvision import transforms


def get_patch(*args, patch_size=96, scale=1, scale2=1):
    ih, iw = args[0].shape[:2]  ## LR image
 
    tp = int(round(scale * patch_size))
    tp2 = int(round(scale2 * patch_size))
    ip = patch_size
    
    if scale==int(scale):  #2,3,4
        step = 1
    elif (scale*2)== int(scale*2):  #1.5,2.5,3.5
        step = 2
    elif (scale*5) == int(scale*5): #1.2,1.4,1.6,1.8,2.2,2.4,2.6,2.8,3.2,3.4,3.6,3.8
        step = 5
    else:       #1.1,1.3,1.7,1.9,2.1,2.3,2.7,2.9,3.1,3.3,3.5,3.9
        step = 10
    if scale2==int(scale2):
        step2 = 1
    elif (scale2*2)== int(scale2*2):
        step2 = 2
    elif (scale2*5) == int(scale2*5):
        step2 = 5
    else:
        step2 = 10
    if (ih-ip)//step==0:
        iy = 0
    else:
        iy = random.randrange(0, (ih-ip)//step) * step
    if (iw-ip)//step==0:
        ix = 0
    else:
        ix = random.randrange(0, (iw-ip)//step2) * step2
        
    tx, ty = int(round(scale2 * ix)), int(round(scale * iy))

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp2, :] for a in args[1:]]
    ]

    # ret = [
    #     args[0][:, :, :],
    #     *[a[:, :, :] for a in args[1:]]
    # ]

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        #tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img, rot=True):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot:
            if rot90: img = img.transpose(1, 0, 2)

        return img

    out = []

    if args[1].shape[0] == args[1].shape[1]:
        for arg in args:
            out.append(_augment(arg))
    else:
        for arg in args:
            out.append(_augment(arg, rot=False))

    return out

