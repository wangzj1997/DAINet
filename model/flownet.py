import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
import torch.nn as nn
import torch.nn.functional as F

try:
    from spatial_correlation_sampler import spatial_correlation_sample
except ImportError as e:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn(
            "failed to load custom correlation module" "which is needed for FlowNetC",
            ImportWarning,
        )

__all__ = ["flownetc", "flownetc_bn"]


class FlowNetC(nn.Module):
    expansion = 1

    def __init__(self, instanceNorm=False):
        super(FlowNetC, self).__init__()

        self.instanceNorm = instanceNorm
        self.conv1 = conv(self.instanceNorm, 64, 64, kernel_size=3, stride=1)
        self.conv2 = conv(self.instanceNorm, 64, 128, kernel_size=3, stride=1)

        self.conv_redir = conv(self.instanceNorm, 128, 32, kernel_size=1, stride=1)

        self.finalpred = predict_flow(473)



        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, pred, ref):
        x1 = pred
        x2 = ref

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)


        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)

        out_conv_redir = self.conv_redir(out_conv2a)
        out_correlation = correlate(out_conv2a, out_conv2b)

        in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)

        flow = self.finalpred(in_conv3_1)

        return flow



def conv(instanceNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if instanceNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.InstanceNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)



def correlate(input1, input2):
    out_corr = spatial_correlation_sample(
        input1,
        input2,
        kernel_size=1,
        patch_size=21,
        stride=1,
        padding=0,
        dilation_patch=2,
    )
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w) / input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)
