import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
from argparse import Namespace
import random 
import math
from model.mambablock import make_mamba, make_mamba_after
from model.resblock import ResBlock
from mamba_ssm.modules.mamba_simple import Mamba
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmengine.model.weight_init import constant_init
from .flownet import FlowNetC
from .attnet import FSAS


def make_model(args, parent=False):
    return DUALRef(args)

class SineAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)

def patch_norm_2d(x, kernel_size=3):
    mean = F.avg_pool2d(x, kernel_size=kernel_size, padding=kernel_size//2)
    mean_sq = F.avg_pool2d(x**2, kernel_size=kernel_size, padding=kernel_size//2)
    var = mean_sq - mean**2
    return (x-mean)/(var + 1e-6)

class ImplicitDecoder(nn.Module):
    def __init__(self, in_channels=64, hidden_dims=[128, 128, 128, 128, 64]):
        super().__init__()

        last_dim_K = in_channels * 9 
        
        last_dim_Q = 3

        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()
        
        for hidden_dim in hidden_dims:
            self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                        nn.ReLU(),
                                        ResBlock(channels = hidden_dim, nConvLayers = 4)          #融合模块
                                        ))    

            self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                        SineAct()))
            last_dim_K = hidden_dim
            last_dim_Q = hidden_dim
            
        self.last_layer = nn.Conv2d(hidden_dims[-1], 64, 1)


        self.in_branch = nn.Sequential(nn.Conv2d(64 * 9, hidden_dims[-2], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-2],hidden_dims[-1], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-1],hidden_dims[-1], 1),
                            nn.ReLU())
        
    def _make_pos_encoding(self, x, size): 
        B, C, H, W = x.shape
        H_up, W_up = size
       
        h_idx = -1 + 1/H + 2/H * torch.arange(H, device=x.device).float()
        w_idx = -1 + 1/W + 2/W * torch.arange(W, device=x.device).float()
        in_grid = torch.stack(torch.meshgrid(h_idx, w_idx), dim=0)

        h_idx_up = -1 + 1/H_up + 2/H_up * torch.arange(H_up, device=x.device).float()
        w_idx_up = -1 + 1/W_up + 2/W_up * torch.arange(W_up, device=x.device).float()
        up_grid = torch.stack(torch.meshgrid(h_idx_up, w_idx_up), dim=0)
        
        rel_grid = (up_grid - F.interpolate(in_grid.unsqueeze(0), size=(H_up, W_up), mode='nearest'))
        rel_grid[:,0,:,:] *= H
        rel_grid[:,1,:,:] *= W

        return rel_grid.contiguous().detach()

    def step(self, x, syn_inp):
              
        q = syn_inp
        k = x

        for i in range(len(self.K)):
            k = self.K[i](k)
            q = k*self.Q[i](q)
        q = self.last_layer(q)

        return q + self.in_branch(x) 


    def forward(self, x, size):
        B, C, H_in, W_in = x.shape
        # print(size) #放大后的size
        rel_coord = (self._make_pos_encoding(x, size).expand(B, -1, *size))
        ratio = (x.new_tensor([math.sqrt((H_in*W_in)/(size[0]*size[1]))]).view(1, -1, 1, 1).expand(B, -1, *size))
      
        syn_inp = torch.cat([rel_coord, ratio], dim=1)
        x = F.interpolate(F.unfold(x, 3, padding=1).view(B, C*9, H_in, W_in), size=syn_inp.shape[-2:], mode='bilinear')

        pred = self.step(x, syn_inp)
        return pred
    
class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = 10 

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.in_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.flownet = FlowNetC()
        self.attnet = FSAS()



        self.fusion = nn.Conv2d(128, 64, 1)

        self.init_offset()

    def init_offset(self):
        """Init constant offset."""
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def flow_warp(self, x,
                flow,
                interpolation='bilinear',
                padding_mode='zeros',
                align_corners=True):
        """Warp an image or a feature map with optical flow.

        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
                a two-channel, denoting the width and height relative offsets.
                Note that the values are not normalized to [-1, 1].
            interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
                Default: 'bilinear'.
            padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Whether align corners. Default: True.

        Returns:
            Tensor: Warped image or feature map.
        """     
        if x.size()[-2:] != flow.size()[1:3]:
            raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                            f'flow ({flow.size()[1:3]}) are not the same.')
        _, _, h, w = x.size()
        # create mesh grid
        device = flow.device
        # torch.meshgrid has been modified in 1.10.0 (compatibility with previous
        # versions), and will be further modified in 1.12 (Breaking Change)
        if 'indexing' in torch.meshgrid.__code__.co_varnames:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, h, device=device, dtype=x.dtype),
                torch.arange(0, w, device=device, dtype=x.dtype),
                indexing='ij')
        else:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, h, device=device, dtype=x.dtype),
                torch.arange(0, w, device=device, dtype=x.dtype))
        grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
        grid.requires_grad = False

        grid_flow = grid + flow
        # scale grid_flow to [-1,1]
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        grid_flow = grid_flow.type(x.type())
        output = F.grid_sample(
            x,
            grid_flow,
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners)
        return output  


    def forward(self, pred, ref):
        """Forward function."""

        flow = self.compute_flow(pred, ref)
        output_after_attn = self.compute_attention(pred, ref)

        feature_wrap = self.flow_warp(ref, flow.permute(0, 2, 3, 1))

        # extra_feat = torch.cat([pred, feature_wrap], dim=1)
        extra_feat = torch.cat([output_after_attn , feature_wrap], dim=1)


        # out = self.conv_offset(extra_feat)
        out = self.conv_offset(extra_feat)


        oh, ow, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = torch.tanh(torch.cat((oh, ow), dim=1))
        # offset = torch.cat((oh, ow), dim=1)
        # offset = self.max_residue_magnitude * torch.tanh(off)

        # offset = offset + flow.repeat(1, offset.size(1) // 2, 1, 1)
        # mask
        mask = torch.sigmoid(mask)


        alingn_feature = modulated_deform_conv2d(ref, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

        final = torch.cat([pred, alingn_feature], dim=1)

        final = self.fusion(final)

        return final


    def compute_flow(self, pred, ref):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """
        flow = self.flownet(pred, ref)
 
        return flow
    
    def compute_attention(self, pred, ref):

        attention = self.attnet(pred, ref)
 
        return attention

class DUALRef(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = make_mamba()
        self.encoder_1 = make_mamba_after()
        self.encoder_2 = make_mamba_after()

        self.decoder = ImplicitDecoder()
        self.decoder_1 = ImplicitDecoder()
        self.decoder_2 = ImplicitDecoder()

        self.multicontrastaliign = SecondOrderDeformableAlignment(
                64,
                64,
                3,
                padding=1,
                deform_groups=16)
        self.multicontrastaliign_1 = SecondOrderDeformableAlignment(
                64,
                64,
                3,
                padding=1,
                deform_groups=16)
        self.multicontrastaliign_2 = SecondOrderDeformableAlignment(
                64,
                64,
                3,
                padding=1,
                deform_groups=16) 
           
        # self.impenc_2 = nn.Conv2d(128, 64, 1)
        # self.impenc_1 = nn.Conv2d(192, 64, 1)
        # self.impenc = nn.Conv2d(192, 64, 1)
        self.fusion = nn.Conv2d(64, 2, 1)
                                       
    def set_scale(self, scale, scale2):
        self.scale = scale
        self.scale2 = scale2

    def forward(self, inp):
        if len(inp)==5:
            epoch = inp[4]
        else:
            epoch = None
        ref_type = inp[3]
        if ref_type == None:
            ref_type = random.randint(1,2) 
            if epoch is not None and epoch < 10:
                ref_type = 1
        ref = inp[ref_type] 
        inp = inp[0]

        #encoder
        feat, ref = self.encoder((inp-0.5)/0.5, (ref-0.5)/0.5)
        B_0,C_0,H_0,W_0 = feat.shape
        feat_1 = F.avg_pool2d(feat, kernel_size=2, stride=2, padding=0)
        ref_1 = F.avg_pool2d(ref, kernel_size=2, stride=2, padding=0)

        feat_1, ref_1 = self.encoder_1(feat_1, ref_1)
        B_1,C_1,H_1,W_1 = feat_1.shape 
        feat_2 = F.avg_pool2d(feat_1, kernel_size=2, stride=2, padding=0)
        ref_2 = F.avg_pool2d(ref_1, kernel_size=2, stride=2, padding=0)

        feat_2, ref_2 = self.encoder_2(feat_2, ref_2)
        B_2,C_2,H_2,W_2 = feat_2.shape 

        fusionfeat = self.multicontrastaliign(feat, ref)
        fusionfeat_1 = self.multicontrastaliign(feat_1, ref_1)
        fusionfeat_2 = self.multicontrastaliign(feat_2, ref_2)



        # pred_2 = self.decoder_2(self.impenc_2(torch.cat((feat_2,fusionfeat_2), dim=1)), [H_1,W_1])

        # pred_1 = self.decoder_1(self.impenc_1(torch.cat((pred_2,feat_1,fusionfeat_1), dim=1)), [H_0,W_0])

        # pred_0 = self.decoder(self.impenc(torch.cat((pred_1,feat,fusionfeat), dim=1)), [H_0,W_0])

        pred_2 = self.decoder_2(feat_2 + fusionfeat_2, [H_1,W_1])

        pred_1 = self.decoder_1(pred_2 + fusionfeat_1, [H_0,W_0])

        pred_0 = self.decoder(pred_1 + fusionfeat, [H_0,W_0])

        pred = self.fusion(pred_0)

        return pred*0.5+0.5
