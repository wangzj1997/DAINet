import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from data.transforms import fft2c, ifft2c


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def complex_to_chan_dim(x):
    b, c, h, w, two = x.shape
    assert two == 2
    return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)

def chan_complex_to_last_dim(x):
    b, c2, h, w = x.shape
    assert c2 % 2 == 0
    c = c2 // 2
    return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()


class FSAS(nn.Module):
    def __init__(self, dim=64, bias=False):
        super(FSAS, self).__init__()

        # self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        
        # self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        # self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        # self.norm = LayerNorm(dim * 2)

        # self.patch_size = 32

        self.to_hidden_q = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        
        self.to_hidden_k = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.to_hidden_v = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        # self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2)

    def forward(self, pred, ref):
        # hidden = self.to_hidden(x)
    

        # q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        hidden_q = self.to_hidden_q(pred)
    
        hidden_k = self.to_hidden_k(ref)
    
        hidden_v = self.to_hidden_v(ref)

        # q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)


        # q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        # k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        q_patch = hidden_q
        k_patch = hidden_k        
        
        q_fft = fft2c(chan_complex_to_last_dim(q_patch))
        k_fft = fft2c(chan_complex_to_last_dim(k_patch))
        
        q_fft = q_fft[..., 0] + 1j * q_fft[..., 1]
        k_fft = k_fft[..., 0] + 1j * k_fft[..., 1]                
                
                
        # q_fft = torch.fft.rfft2(q_patch.float())
        # k_fft = torch.fft.rfft2(k_patch.float())
        out = q_fft * k_fft
        b, c, h, w = out.shape
        out_two = torch.empty(b, c, h, w, 2, device='cuda')
        out_two[..., 0] = out.real
        out_two[..., 1] = out.imag       
        
        out = ifft2c(out_two)
        out = complex_to_chan_dim(out)
        # out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        # out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
        #                 patch2=self.patch_size)

        out_attention = self.norm(out)

        output = hidden_v * out_attention
        output = self.project_out(output)

        return output
    
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias