# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# modified from: https://github.com/thstkdgus35/EDSR-PyTorch
import math
from argparse import Namespace
import numbers
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)

class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim,bimamba_type="v3")
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    def forward(self,ms,ms_resi,pan):
        ms_resi = ms+ms_resi
        ms = self.norm1(ms_resi)
        pan = self.norm2(pan)
        global_f = self.cross_mamba(self.norm1(ms),extra_emb=self.norm2(pan))
        B,HW,C = global_f.shape
        ms = global_f.transpose(1, 2).view(B, C, int(math.sqrt(HW)), int(math.sqrt(HW)))
        ms =  (self.dwconv(ms)+ms).flatten(2).transpose(1, 2)
        return ms,ms_resi


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x

class Refine(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            # CALayer(n_feat,4),
            # CALayer(n_feat,4),
            ChannelAttention(n_feat, 4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        # out = self.process(out)
        out = self.conv_last(out)

        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.process(x)
        y = self.avg_pool(res)
        z = self.conv_du(y)
        return z *res + x

class mambablock(nn.Module):
    def __init__(self, args):
        super(mambablock, self).__init__()
        self.args = args
        G0 = args.G0
        kSize = args.RDNkSize

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.inp_channel, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        self.SFENet1_ref = nn.Conv2d(args.inp_channel, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2_ref = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)


        self.pred_to_token = PatchEmbed(in_chans=G0,embed_dim=G0,patch_size=1,stride=1)
        self.ref_to_token = PatchEmbed(in_chans=G0,embed_dim=G0,patch_size=1,stride=1)


        self.deep_fusion1_pref= CrossMamba(G0)
        self.deep_fusion2_pref = CrossMamba(G0)

        self.deep_fusion1_ref= CrossMamba(G0)
        self.deep_fusion2_ref = CrossMamba(G0)

        self.patchunembe = PatchUnEmbed(G0)

        self.output_pred = Refine(G0,G0)
        self.output_ref = Refine(G0,G0)

    def forward(self, pred, ref):
        B,C,H,W = pred.shape
        pred_basic = self.SFENet1(pred)
        pred_basic  = self.SFENet2(pred_basic)

        ref_basic = self.SFENet1_ref(ref)
        ref_basic  = self.SFENet2_ref(ref_basic)

        pred_f = self.pred_to_token(pred_basic)
        ref_f = self.ref_to_token(ref_basic)

        residual_pred_f = 0
        residual_ref_f = 0

        pred_f_cross,residual_pred_f = self.deep_fusion1_pref(pred_f,residual_pred_f,ref_f)
        pred_f_cross,residual_pred_f = self.deep_fusion2_pref(pred_f_cross,residual_pred_f,ref_f)

        ref_f_cross,residual_ref_f = self.deep_fusion1_ref(ref_f,residual_ref_f,pred_f)
        ref_f_cross,residual_ref_f = self.deep_fusion2_ref(ref_f_cross,residual_ref_f,pred_f)


        pred = self.patchunembe(pred_f_cross,(H,W))
        ref = self.patchunembe(ref_f_cross,(H,W))


        pred = self.output_pred(pred)+pred_basic
        ref = self.output_ref(pred)+ref_basic

        return pred, ref

def make_mamba(G0=64, RDNkSize=3):
    args = Namespace()
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.inp_channel = 2
    return mambablock(args)

def make_mamba_after(G0=64, RDNkSize=3):
    args = Namespace()
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.inp_channel = 64

    return mambablock(args)