import torch
from torch import nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.nn import init as init
import numbers
from einops import rearrange
from basicsr.models.archs.arch_util import (DCNv2Pack, ResidualBlockNoBN,
                                            make_layer)

from basicsr.models.archs.restormer_arch import ResT

from basicsr.models.archs.octconv import OctaveConv, OctaveConv2
from basicsr.models.archs.basic_modules import get_norm, get_act, ConvNormAct, LayerScale2D, MSPatchEmb
inplace = True

class GetWeight(nn.Module):  
    def __init__(self, channel=64):
        super(GetWeight, self).__init__()
        self.downsample = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 4, channel * 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, size, _ = x.size()  # torch.Size([1, 32, 24, 24])
        #c = c
        x = self.downsample(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y  # [16, 32, 1, 1]


class ImplicitTrans(nn.Module): 
    def __init__(self, in_channels):
        super(ImplicitTrans, self).__init__()
        self.table = torch.tensor([
            16, 16, 16, 16, 17, 18, 21, 24,
            16, 16, 16, 16, 17, 19, 22, 25,
            16, 16, 17, 18, 20, 22, 25, 29,
            16, 16, 18, 21, 24, 27, 31, 36,
            17, 17, 20, 24, 30, 35, 41, 47,
            18, 19, 22, 27, 35, 44, 54, 65,
            21, 22, 25, 31, 41, 54, 70, 88,
            24, 25, 29, 36, 47, 65, 88, 115]) / 255.0  # .reshape(8, 8)
        self.table = self.table.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  

        temp = torch.empty(256, 1, 1, 1)
        self.factor = nn.Parameter(torch.ones_like(temp))
        self.bias = nn.Parameter(torch.zeros_like(temp))
        self.table = self.table.cuda()

        conv_shape = (256, 64, 1, 1)
        kernel = np.zeros(conv_shape, dtype='float32')
        r1 = math.sqrt(1.0 / 8)
        r2 = math.sqrt(2.0 / 8)
        for i in np.arange(0.0, 8.0, 0.5):  
            _u = 2 * i + 1  
            for j in np.arange(0.0, 8.0, 0.5):
                _v = 2 * j + 1  
                index1 = int(_u * _v - 1)
                for u in range(8):  
                    for v in range(8):
                        index2 = u * 8 + v
                        t = math.cos(_u * u * math.pi / 16) * math.cos(_v * v * math.pi / 16)
                        t = t * r1 if u == 0 else t * r2  # if u=0, t=t*r1
                        t = t * r1 if v == 0 else t * r2
                        kernel[index1, index2, 0, 0] = t
        self.kernel = torch.from_numpy(kernel)
        self.kernel = self.kernel.cuda()

    def forward(self, x, weight):  
        new_table = torch.repeat_interleave(self.table, repeats=4, dim=0)
        _table = new_table * self.factor + self.bias
        _kernel = self.kernel * _table
        x = x * weight  #  tenosr x is [4,64,16,16]
        
        
        y = F.conv2d(input=x, weight=_kernel, stride=1)  # y size:torch.Size([1, 256, 24, 24])
       
        pixel_shuffle = nn.PixelShuffle(2)
        y = pixel_shuffle(y) # [1,64,48,48]
        
        return y
        
        
class ImplicitTrans_org(nn.Module):
    def __init__(self, in_channels):
        super(ImplicitTrans_org, self).__init__()
        self.table = torch.tensor([
            16, 16, 16, 16, 17, 18, 21, 24,
            16, 16, 16, 16, 17, 19, 22, 25,
            16, 16, 17, 18, 20, 22, 25, 29,
            16, 16, 18, 21, 24, 27, 31, 36,
            17, 17, 20, 24, 30, 35, 41, 47,
            18, 19, 22, 27, 35, 44, 54, 65,
            21, 22, 25, 31, 41, 54, 70, 88,
            24, 25, 29, 36, 47, 65, 88, 115]) / 255.0  # .reshape(8, 8)
        self.table = self.table.unsqueeze(-1)
        self.table = self.table.unsqueeze(-1)
        self.table = self.table.unsqueeze(-1)

        self.factor = nn.Parameter(torch.ones_like(self.table))
        self.bias = nn.Parameter(torch.zeros_like(self.table))
        self.table = self.table.cuda()

        conv_shape = (64, 64, 1, 1)
        kernel = np.zeros(conv_shape, dtype='float32')
        r1 = math.sqrt(1.0 / 8)
        r2 = math.sqrt(2.0 / 8)
        for i in range(8):
            _u = 2 * i + 1
            for j in range(8):
                _v = 2 * j + 1
                index = i * 8 + j
                for u in range(8):
                    for v in range(8):
                        index2 = u * 8 + v
                        t = math.cos(_u * u * math.pi / 16) * math.cos(_v * v * math.pi / 16)

                        t = t * r1 if u == 0 else t * r2
                        t = t * r1 if v == 0 else t * r2
                        kernel[index, index2, 0, 0] = t
        self.kernel = torch.from_numpy(kernel)
        self.kernel = self.kernel.cuda()

    def forward(self, x, weight):
        _table = self.table * self.factor + self.bias
        _kernel = self.kernel * _table
        x = x * weight  # [4, 64, 16, 16])
        y = F.conv2d(input=x, weight=_kernel, stride=1)
       
        return y




class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3,
                                                     1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            self.dcn_pack[level] = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                padding=1,
                deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                  1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(
            num_feat,
            num_feat,
            3,
            padding=1,
            deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat



class PyramidCell(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(PyramidCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation_rates = dilation_rates
        self.dilation_rate = 0
        # (3, 2, 1, 1, 1, 1)
        self.conv_relu_1 = ConvRelu(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel=3, padding=3,
                                    dilation_rate=dilation_rates[0])
        self.conv_relu_2 = ConvRelu(in_channels=self.in_channels * 2, out_channels=self.out_channels,
                                    kernel=3, padding=2,
                                    dilation_rate=dilation_rates[1])
        self.conv_relu_3 = ConvRelu(in_channels=self.in_channels * 3, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=dilation_rates[2])
        self.conv_relu_4 = ConvRelu(in_channels=self.in_channels * 4, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=dilation_rates[2])
        self.conv_relu_5 = ConvRelu(in_channels=self.in_channels * 5, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=dilation_rates[2])
        self.conv_relu_6 = ConvRelu(in_channels=self.in_channels * 6, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=dilation_rates[2])

    def forward(self, x):
        t = self.conv_relu_1(x)  # 64
        _t = torch.cat([x, t], dim=1)  # 128

        t = self.conv_relu_2(_t)
        _t = torch.cat([_t, t], dim=1)  #

        t = self.conv_relu_3(_t)
        _t = torch.cat([_t, t], dim=1)

        t = self.conv_relu_4(_t)
        _t = torch.cat([_t, t], dim=1)

        t = self.conv_relu_5(_t)
        _t = torch.cat([_t, t], dim=1)

        t = self.conv_relu_6(_t)
        _t = torch.cat([_t, t], dim=1)
        return _t

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# Define Cross Attention Block
class blockNL(torch.nn.Module):
    def __init__(self, channels):
        super(blockNL, self).__init__()
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        
        self.norm_x = LayerNorm(1, 'WithBias')
        self.norm_z = LayerNorm(64, 'WithBias')

        self.t = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.p = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        self.v = nn.Conv2d(in_channels=self.channels+1, out_channels=self.channels+1, kernel_size=1, stride=1, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
        )

    def forward(self, x, z):
        
        x0 = self.norm_x(x)
        z0 = self.norm_z(z)
        
        z1 = self.t(z0)
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1) # b, c, hw
        x1 = self.p(x0) # b, c, hw
        x1 = x1.view(b, c, -1)
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        x1 = torch.nn.functional.normalize(x1, dim=-1)
        x_t = x1.permute(0, 2, 1) # b, hw, c
        att = torch.matmul(z1, x_t)
        att = self.softmax(att) # b, c, c
        
        z2 = self.g(z0)
        z_v = z2.view(b, c, -1)
        out_x = torch.matmul(att, z_v)
        out_x = out_x.view(b, c, h, w)
        out_x = self.w(out_x) + self.pos_emb(z2) + z
        y = self.v(torch.cat([x, out_x], 1))

        return y

# Define ISCA block
class Atten(torch.nn.Module):
    def __init__(self, channels):
        super(Atten, self).__init__()
               
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = LayerNorm(self.channels, 'WithBias')
        self.norm2 = LayerNorm(self.channels, 'WithBias')
        self.conv_q = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.conv_kv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, stride=1, padding=1, groups=self.channels*2, bias=True)
        )
        self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        
    def forward(self, pre, cur):
        
        b, c, h, w = pre.shape
        pre_ln = self.norm1(pre)
        cur_ln = self.norm2(cur)
        q = self.conv_q(cur_ln)
        q = q.view(b, c, -1)
        k,v = self.conv_kv(pre_ln).chunk(2, dim=1)
        k = k.view(b, c, -1)
        v = v.view(b, c, -1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        att = torch.matmul(q, k.permute(0, 2, 1))
        att = self.softmax(att)
        out = torch.matmul(att, v).view(b, c, h, w)
        out = self.conv_out(out) + cur
        
        return out


class ImpFreqUp(nn.Module):  
    def __init__(self, n_channels, n_pyramid_cells, n_pyramid_channels): 
        super(ImpFreqUp, self).__init__()
        #self.pyramid = PyramidCell(in_channels=n_channels, out_channels=n_pyramid_channels,
        #                           dilation_rates=n_pyramid_cells)
        self.pyramid = ResT(num_blocks = 4 )
        self.conv_1 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_2 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3,
                           padding=2, dilation_rate=2)
        #self.channel_squeeze = Conv(in_channels=n_channels * 7, out_channels=n_channels,
        #                            kernel=1, padding=0)
        self.get_weight_y = GetWeight()
        self.get_weight_c = GetWeight()
        self.implicit_trans_1 = ImplicitTrans(in_channels=n_channels)
        self.implicit_trans_2 = ImplicitTrans(in_channels=n_channels)

        self.attention = Atten(64)
        
        self.pixel_restoration = ResT(num_blocks = 4 )

        self.conv_3 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_4 = Conv(in_channels=n_channels*2, out_channels=n_channels, kernel=3, padding=1)  
        
        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

        self.upscale2 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):  # x size = [1,64,n,n]
        _t = self.pyramid(x)
        #_t = self.channel_squeeze(_t)  # _t torch.Size([1, 64, 24, 24])

        _ty = self.conv_1(_t)
        _tc = self.conv_2(_t)
        _ty = torch.clamp(_ty, -0.5, 0.5) # [4,64,16,16]  _ty torch.Size([1, 64, 24, 24])

        ty_weight = self.get_weight_y(_t)
        _ty = self.implicit_trans_1(_ty, ty_weight)
        tc_weight = self.get_weight_c(_t)
        _tc = self.implicit_trans_2(_tc, tc_weight)  # [4,16,16,16]--->[4,256,16,16]  [1,64,48,48]  
        _tp = self.pixel_restoration(_t)  # _tp:torch.Size([1, 64, 24, 24])
        _tp = self.conv_3(_tp)  # _tp:torch.Size([1, 64, 24, 24])
        _tp = self.upscale(_tp)

        _td = torch.cat([_ty, _tc], dim=1)  # [4,128,16,16]
        _td = self.conv_4(_td)  # _td:torch.Size([1, 64, 48, 48])

	y = torch.add(_td, _tp)  
      #  y = y.mul(0.1)
      #  x = self.upscale2(x)
      #  y = torch.add(x, y)
        return y


class ImpFreqUpx1(nn.Module):
    def __init__(self, n_channels, n_pyramid_cells, n_pyramid_channels):
        super(ImpFreqUpx1, self).__init__()
        #self.pyramid = PyramidCell(in_channels=n_channels, out_channels=n_pyramid_channels,
         #                          dilation_rates=n_pyramid_cells)
        self.pyramid = ResT(num_blocks = 4 )
        self.conv_1 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_2 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3,
                           padding=2, dilation_rate=2)
        ##self.channel_squeeze = Conv(in_channels=n_channels * 7, out_channels=n_channels,
         #                           kernel=1, padding=0)
        self.get_weight_y = GetWeight()
        self.get_weight_c = GetWeight()
        self.implicit_trans_1 = ImplicitTrans_org(in_channels=n_channels)
        self.implicit_trans_2 = ImplicitTrans_org(in_channels=n_channels)


        self.pixel_restoration = ResT(num_blocks = 4 )

        self.conv_3 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_4 = Conv(in_channels=n_channels * 2, out_channels=n_channels, kernel=3,
                           padding=1)

        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x): 
        # x size = [1,64,n,n]
        _t = self.pyramid(x)
       # _t = self.channel_squeeze(_t)  # _t torch.Size([1, 64, 24, 24])
        _ty = self.conv_1(_t)
        _tc = self.conv_2(_t)
        _ty = torch.clamp(_ty, -0.5, 0.5)  # [4,64,16,16]  _ty torch.Size([1, 64, 24, 24])

        ty_weight = self.get_weight_y(_t)
        _ty = self.implicit_trans_1(_ty, ty_weight)
        tc_weight = self.get_weight_c(_t)
        _tc = self.implicit_trans_2(_tc, tc_weight)  

        _tp = self.pixel_restoration(_t)  # _tp:torch.Size([1, 64, 24, 24])
        _tp = self.conv_3(_tp)  # _tp:torch.Size([1, 64, 24, 24])
        # _tp = self.upscale(_tp)

        _td = torch.cat([_ty, _tc], dim=1)  # [4,128,16,16]

        _td = self.conv_4(_td)  # _td:torch.Size([1, 64, 48, 48])

        y = torch.add(_td, _tp)  
        y = y.mul(0.1)

        # x = self.upscale(x)

        y = torch.add(x, y)
        return y


def pixel_unshuffle(input, downscale_factor):
    """
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    """
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)


class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        """
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        """

        return pixel_unshuffle(input, self.downscale_factor)


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=1, use_bias=True, dilation_rate=1):
        super(ConvRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding,
                              bias=use_bias,
                              dilation=dilation_rate)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        output = self.relu(self.conv(x))
        return output


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=0, use_bias=True, dilation_rate=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding,
                              bias=use_bias,
                              dilation=dilation_rate)

    def forward(self, x):
        output = self.conv(x)
        return output


def make_layer_conv(basic_block, num_basic_block):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block)
    return nn.Sequential(*layers)



class iRMB(nn.Module):
	
	def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
				 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=64, window_size=7,
				 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
		super().__init__()
		self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
		dim_mid = int(dim_in * exp_ratio)
		self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
		self.attn_s = attn_s
		if self.attn_s:
			assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
			self.dim_head = dim_head
			self.window_size = window_size
			self.num_head = dim_in // dim_head
			self.scale = self.dim_head ** -0.5
			self.attn_pre = attn_pre
			self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none', act_layer='none')
			self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias, norm_layer='none', act_layer=act_layer, inplace=inplace)
			self.attn_drop = nn.Dropout(attn_drop)
		else:
			if v_proj:
				self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none', act_layer=act_layer, inplace=inplace)
			else:
				self.v = nn.Identity()
		self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation, groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
		self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()
		
		self.proj_drop = nn.Dropout(drop)
		self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)
		self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
	
	def forward(self, x):
		shortcut = x
		x = self.norm(x)
		B, C, H, W = x.shape
		if self.attn_s:
			# padding
			if self.window_size <= 0:
				window_size_W, window_size_H = W, H
			else:
				window_size_W, window_size_H = self.window_size, self.window_size
			pad_l, pad_t = 0, 0
			pad_r = (window_size_W - W % window_size_W) % window_size_W
			pad_b = (window_size_H - H % window_size_H) % window_size_H
			x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
			n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
			x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
			# attention
			b, c, h, w = x.shape
			qk = self.qk(x)
			qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
			q, k = qk[0], qk[1]
			attn_spa = (q @ k.transpose(-2, -1)) * self.scale
			attn_spa = attn_spa.softmax(dim=-1)
			attn_spa = self.attn_drop(attn_spa)
			if self.attn_pre:
				x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
				x_spa = attn_spa @ x
				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
				x_spa = self.v(x_spa)
			else:
				v = self.v(x)
				v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
				x_spa = attn_spa @ v
				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
			# unpadding
			x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
			if pad_r > 0 or pad_b > 0:
				x = x[:, :, :H, :W].contiguous()
		else:
			x = self.v(x)

		x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
		
		x = self.proj_drop(x)
		x = self.proj(x)
		
		x = (shortcut + self.drop_path(x)) if self.has_skip else x
		return x



class ETransBlock(nn.Module):
    def __init__(self, n_channels):
        super(ETransBlock, self).__init__()

        #self.patch_embed = OverlapPatchEmbed(in_c=64, embed_dim=48)
        self.occonv = OctaveConv(in_channels=n_channels, out_channels=n_channels, kernel_size=3, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                               groups=1, bias=False)
        self.exconv  = iRMB(dim_in=32, dim_out=32,dim_head=16)
        self.l_exconv =  ResT(inp_channels=32,out_channels=32, dim = 16, num_blocks = 4)


    def forward(self, x):
        #x = self.patch_embed(x)
        x_h, x_l = x if type(x) is tuple else (x, None)

        x_hh, x_ll = self.occonv((x_h, x_l))
        x_hh =  self.exconv(x_hh)
        x_ll =  self.l_exconv(x_ll)

        return x_hh, x_ll


class HFUR(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=2,
                 hr_in=False,
                 with_predeblur=False,
                 with_tsa=True):
        super(HFUR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa


        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extrat pyramid features
        self.feature_extraction = make_layer(
            ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(
            num_feat=num_feat, deformable_groups=deformable_groups)
        if self.with_tsa:
            self.fusion = TSAFusion(
                num_feat=num_feat,
                num_frame=num_frame,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

       
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.n_channels = num_feat
        self.n_pyramids = 1
        self.n_pyramid_cells = [3, 2, 1]
        #self.n_pyramid_channels = n_channels
        self.n_pyramid_channels = num_feat

        self.downscale_1 = nn.Sequential(
            PixelUnshuffle(downscale_factor=2),   
            nn.Conv2d(in_channels=self.n_channels * 2 * 2, out_channels=self.n_channels,
                      kernel_size=5, stride=1, padding=2, bias=False)
        )
        self.downscale_2 = nn.Sequential(
            PixelUnshuffle(downscale_factor=2),
            nn.Conv2d(in_channels=self.n_channels * 2 * 2, out_channels=self.n_channels,
                      kernel_size=5, stride=1, padding=2, bias=False)
        )

        self.conv_relu_X1_1 = ConvRelu(in_channels=self.n_channels, out_channels=self.n_channels, kernel=3, padding=1)

        # ×1
        self.dual_domain_blocks_x1 = self.make_layer(
            block=ImpFreqUpx1,
            num_of_layer=self.n_pyramids)
        #self.conv_relu_X1_2 = ConvRelu(in_channels=self.n_channels, out_channels=self.n_channels, kernel=3, padding=1)

        self.conv_relu_X2_1 = ConvRelu(in_channels=self.n_channels, out_channels=self.n_channels, kernel=3, padding=1)
        # ×2
        self.dual_domain_blocks_x2 = self.make_layer(
            block=ImpFreqUp,
            num_of_layer=self.n_pyramids)
        self.conv_relu_X2_2 = ConvRelu(in_channels=self.n_channels, out_channels=self.n_channels, kernel=3, padding=1)

        self.conv_relu_X4_1 = ConvRelu(in_channels=self.n_channels, out_channels=self.n_channels, kernel=3, padding=1)
        # ×4
        self.dual_domain_blocks_x4 = self.make_layer(
            block=ImpFreqUp,
            num_of_layer=self.n_pyramids)

        self.conv_relu_X4_2 = ConvRelu(in_channels=self.n_channels, out_channels=self.n_channels, kernel=3, padding=1)

        self.conv_relu_channel_merge_1 = ConvRelu(in_channels=self.n_channels * 2, out_channels=self.n_channels, kernel=3,
                                                  padding=1)
        self.conv_relu_channel_merge_2 = ConvRelu(in_channels=self.n_channels * 2, out_channels=self.n_channels, kernel=3,
                                                  padding=1)
     
        self.conv = OctaveConv2(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size=1, alpha_in=0, alpha_out=0.5, stride=1, padding=0, dilation=1,
                               groups=1, bias=False)
     
        self.conv3 = OctaveConv(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size=1, alpha_in=0.5, alpha_out=0, stride=1, padding=0, dilation=1,
                               groups=1, bias=False)

        self.conv2_1 = OctaveConv2(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size=1, alpha_in=0, alpha_out=0.5, stride=1, padding=0, dilation=1,
                               groups=1, bias=False)
    
        self.conv2_3 = OctaveConv(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size=1, alpha_in=0.5, alpha_out=0, stride=1, padding=0, dilation=1,
                               groups=1, bias=False)
    
        self.pixel_restoration2 = nn.Sequential(*[ETransBlock(n_channels=self.n_channels) for i in range(3)])

        self.pixel_restoration1 = nn.Sequential(*[ETransBlock(n_channels=self.n_channels) for i in range(3)])


    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(n_channels=self.n_channels, n_pyramid_cells=self.n_pyramid_cells,
                                n_pyramid_channels=self.n_pyramid_channels))
        return nn.Sequential(*layers)


    def forward(self, x):
        b, t, c, h, w = x.size()
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, (
                'The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, (
                'The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)

        t_x1 = self.fusion(aligned_feat)  # [4,64,96,96]
        t_x2 = self.downscale_1(t_x1)  # [4,64,48,48]
        t_x4 = self.downscale_2(t_x2)  # [4,64,24,24]
        
        t_x4 = self.conv_relu_X4_1(t_x4) # [4,64,24,24]
        t_x4 = self.dual_domain_blocks_x4(t_x4)  # [4,64,48,48]
        t_x4 = self.conv_relu_X4_2(t_x4)  # [4,64,48,48]

        # t_x4 = self.upscale_2(t_x4)
        t_x2 = torch.cat((t_x2, t_x4), 1)  # [4,128,32,32]   [1, 64, 48, 48]
        t_x2 = self.conv_relu_channel_merge_1(t_x2)  # [4,64,32,32]  [1, 64, 48, 48]

        t_x2 = self.conv_relu_X2_1(t_x2) # [1, 64, 48, 48]
        t_x2 = self.dual_domain_blocks_x2(t_x2)  # [1, 64, 96, 96]
        t_x2 = self.conv_relu_X2_2(t_x2)  # [1,64,96,96]
        
        a2,b2 = self.conv2_1(t_x2)
   
     
        a2,b2 = self.pixel_restoration2((a2,b2))
       
        t_x2 , b2 = self.conv2_3((a2,b2))

        # t_x2 = self.upscale_1(t_x2)  # [4,64,64,64]     
        t_x1 = torch.cat((t_x1, t_x2), 1)  # [4,128,64,64]  [1,128,96,96] 
        t_x1 = self.conv_relu_channel_merge_2(t_x1)  # [1,128,96,96]

        t_x1 = self.conv_relu_X1_1(t_x1)  # [1,64,96,96]
        t_x1 = self.dual_domain_blocks_x1(t_x1)  # [1,64,192,192]
        
        a1,b1 = self.conv(t_x1)
       
        a1,b1 = self.pixel_restoration1((a1,b1)) 
              
        t_x1 , b1 = self.conv3((a1,b1))

        out = self.lrelu(self.conv_hr(t_x1))
        
        out = self.conv_last(out)

        base = x_center
        out += base
        return out
