import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Function, Variable
import logging
import numpy as np
logging.basicConfig(format='%(message)s', level=logging.INFO)

from typing import Optional, Any, Tuple

BN_EPS=1e-4
BN_MOMENTUM=1e-3


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


def grad_reverse(x, lambd=1.0):
    return GradientReversalFunction.apply(x, lambd)



class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

# from clair3.task.main import GT21, GENOTYPE, VARIANT_LENGTH_1, VARIANT_LENGTH_2
params = dict(
            float_type=torch.float32,
            task_loss_weights=[
                1,                       # gt21
                1,                       # genotype
                1,                       # variant/indel length 0
                1,                       # variant/indel length 1
                1                        # l2 loss
            ],
            output_shape= 90,
            output_gt21_shape=21,
            output_genotype_shape=3,
            output_indel_length_shape_1=33,
            output_indel_length_shape_2=33,
            output_gt21_entropy_weights=[1] * 21,
            output_genotype_entropy_weights=[1] * 3,
            output_indel_length_entropy_weights_1=[1] * 33,
            output_indel_length_entropy_weights_2=[1] * 33,
            L3_dropout_rate=0,
            L4_num_units=256,
            L4_pileup_num_units=128,
            L4_dropout_rate=0,
            L5_1_num_units=128,
            L5_1_dropout_rate=0,
            L5_2_num_units=128,
            L5_2_dropout_rate=0,
            L5_3_num_units=128,
            L5_3_dropout_rate=0,
            L5_4_num_units=128,
            L5_4_dropout_rate=0,
            LSTM1_num_units=128,
            LSTM2_num_units=160,
            LSTM1_dropout_rate=0,
            LSTM2_dropout_rate=0.5,

            # parameters for transformer
            embed_dim = 768,

        )

import collections
from itertools import repeat
def _ntuple(n):
    """Copied from PyTorch since it's not importable as an internal function

    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/utils.py#L6
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)

class MaxPool2dStaticSamePadding(nn.Module):
    """
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

        # Setup internal representations
        kernel_size_ = _pair(self.kernel_size)
        dilation_ = _pair(1)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, x):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(x, self._reversed_padding_repeated_twice)
        x = self.pool(padded)
        return x


        
class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)

class PyramidPooling(nn.Module):
    def __init__(self, levels, mode="max"):
        """
        https://github.com/revidee/pytorch-pyramid-pooling/blob/master/pyramidpooling.py

        General Pyramid Pooling class which uses Spatial Pyramid Pooling by default and holds the static methods for both spatial and temporal pooling.
        :param levels defines the different divisions to be made in the width and (spatial) height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n], where  n: sum(filter_amount*level*level) for each level in levels (spatial) or
                                                                    n: sum(filter_amount*level) for each level in levels (temporal)
                                            which is the concentration of multi-level pooling
        """
        super(PyramidPooling, self).__init__()
        self.levels = levels
        self.mode = mode

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out

    @staticmethod
    def spatial_pyramid_pool(previous_conv, levels, mode):
        """
        Static Spatial Pyramid Pooling method, which divides the input Tensor vertically and horizontally
        (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width and height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        # print("building", previous_conv_size, levels) # 28 17
        for i in range(len(levels)):
            h_kernel = int(np.ceil(previous_conv_size[0] / levels[i]))
            w_kernel = int(np.ceil(previous_conv_size[1] / levels[i]))
            
            w_pad1 = int(np.floor((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(np.ceil((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            h_pad1 = int(np.floor((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            h_pad2 = int(np.ceil((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * levels[i] - previous_conv_size[1]) and \
                   h_pad1 + h_pad2 == (h_kernel * levels[i] - previous_conv_size[0])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                 mode='constant', value=0)

            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
                # pool = MaxPool2dStaticSamePadding((h_kernel, w_kernel), stride=(h_kernel, w_kernel)  )
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            # x = pool(previous_conv)

            if i == 0:
                spp = x.reshape(num_sample, -1)
            else:
                spp = torch.cat((spp, x.reshape(num_sample, -1)), 1)
            

        return spp

    @staticmethod
    def temporal_pyramid_pool(previous_conv, out_pool_size, mode):
        """
        Static Temporal Pyramid Pooling method, which divides the input Tensor horizontally (last dimensions)
        according to each level in the given levels and pools its value according to the given mode.
        In other words: It divides the Input Tensor in "level" horizontal stripes with width of roughly (previous_conv.size(3) / level)
        and the original height and pools the values inside this stripe
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)
            #
            h_kernel = previous_conv_size[0]
            w_kernel = int(np.ceil(previous_conv_size[1] / out_pool_size[i]))
            w_pad1 = int(np.floor((w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(np.ceil((w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * out_pool_size[i] - previous_conv_size[1])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                tpp = x.view(num_sample, -1)
            else:
                tpp = torch.cat((tpp, x.view(num_sample, -1)), 1)

        return tpp


class SpatialPyramidPooling(PyramidPooling):
    def __init__(self, levels, mode="max"):
        """
                Spatial Pyramid Pooling Module, which divides the input Tensor horizontally and horizontally
                (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
                Can be used as every other pytorch Module and has no learnable parameters since it's a static pooling.
                In other words: It divides the Input Tensor in level*level rectangles width of roughly (previous_conv.size(3) / level)
                and height of roughly (previous_conv.size(2) / level) and pools its value. (pads input to fit)
                :param levels defines the different divisions to be made in the width dimension
                :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
                :returns (forward) a tensor vector with shape [batch x 1 x n],
                                                    where n: sum(filter_amount*level*level) for each level in levels
                                                    which is the concentration of multi-level pooling
                """
        super(SpatialPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        """
                Calculates the output shape given a filter_amount: sum(filter_amount*level*level) for each level in levels
                Can be used to x.view(-1, spp.get_output_size(filter_amount)) for the fully-connected layers
                :param filters: the amount of filter of output fed into the spatial pyramid pooling
                :return: sum(filter_amount*level*level)
        """
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out



# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    if stride == 1:
        # return nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), 
        #              stride=stride, bias=True)
        return nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), 
                     stride=stride, padding="same", bias=True)
    else:
        # basicconv2d, stride is 2 and padding is same fix here
        return nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), 
                     stride=stride, padding='valid', bias=True)
        return Conv2dSame( in_channels, out_channels, 3 ,stride=stride, bias=True)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM, eps=BN_EPS)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM, eps=BN_EPS)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicConv2D, self).__init__()
        self.stride = stride
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM, eps=BN_EPS)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out
# ResNet
class Clair3(nn.Module):
    def __init__(self,  layers, platform, add_indel_length=False, predict=False, test_inception=False):
        super(Clair3, self).__init__()

        # Parameter setup here
        self.output_gt21_shape = params['output_gt21_shape']
        self.output_genotype_shape = params['output_genotype_shape']
        self.output_indel_length_shape_1 = params['output_indel_length_shape_1']
        self.output_indel_length_shape_2 = params['output_indel_length_shape_2']
        self.L3_dropout_rate = params['L3_dropout_rate']
        self.L4_num_units = params['L4_num_units']
        self.L4_dropout_rate = params['L4_dropout_rate']
        self.L5_1_num_units = params['L5_1_num_units']
        self.L5_1_dropout_rate = params['L5_1_dropout_rate']
        self.L5_2_num_units = params['L5_2_num_units']
        self.L5_2_dropout_rate = params['L5_2_dropout_rate']
        self.L5_3_num_units = params['L5_3_num_units']
        self.L5_3_dropout_rate = params['L5_3_dropout_rate']
        self.L5_4_num_units = params['L5_4_num_units']
        self.L5_4_dropout_rate = params['L5_4_dropout_rate']
        self.add_indel_length = add_indel_length
        self.predict = predict

        # model structure below
        self.in_channels = 64
        if platform == "ilmn":
            self.basic1 = BasicConv2D(7, 64, stride=2)
        else:
            self.basic1 = BasicConv2D(8, 64, stride=2)
        self.layer1 = self.make_layer(ResidualBlock, 64, 64, layers[0])
        self.basic2 = BasicConv2D(64, 128, stride=2)
        self.layer2 = self.make_layer(ResidualBlock,128, 128, layers[1])
        self.basic3 = BasicConv2D(128, 256, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 256, layers[2])
        # self.basic4 = BasicConv2D(256, 512, stride=2)
        # self.layer4 = self.make_layer(ResidualBlock, 512, 512, layers[2])
        # self.layer4 = self.make_layer(ResidualBlock, 512, layers[3], 2)
        # self.avgpool = SpatialPyramidPooling([3,2,1]) #nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveMaxPool2d((1, 2))

        # self.L3_dropout = nn.Dropout(self.L3_dropout_rate)
        
        # self.L4 = nn.Sequential( nn.Linear(self.avgpool.get_output_size(256), self.L4_num_units), nn.SELU() )
        feature_num = 2048 if test_inception else 512
        self.L4 = nn.Sequential( nn.Linear(feature_num, self.L4_num_units), nn.SELU() )
        # self.L4_dropout = nn.Dropout(self.L4_dropout_rate)

        # output 
        self.L5_1 = nn.Sequential( nn.Linear(feature_num, self.L4_num_units),  nn.LeakyReLU(inplace=True, negative_slope=0.1), nn.Linear(self.L4_num_units, self.L5_1_num_units),nn.LeakyReLU(inplace=True, negative_slope=0.1) )
        
        self.L5_1_dropout = nn.Dropout(self.L5_1_dropout_rate)
        self.L5_2 = nn.Sequential( nn.Linear(feature_num, self.L4_num_units),  nn.LeakyReLU(inplace=True, negative_slope=0.1), nn.Linear(self.L4_num_units, self.L5_2_num_units),nn.LeakyReLU(inplace=True, negative_slope=0.1) )
        self.L5_2_dropout = nn.Dropout(self.L5_2_dropout_rate)

        self.Y_gt21_logits = nn.Sequential( nn.Linear(self.L5_1_num_units, self.output_gt21_shape))
        self.Y_genotype_logits = nn.Sequential( nn.Linear(self.L5_2_num_units, self.output_genotype_shape))

        if self.add_indel_length:
            self.L5_3 = nn.Sequential(nn.Linear(feature_num, self.L4_num_units),  nn.LeakyReLU(inplace=True, negative_slope=0.1), nn.Linear(self.L4_num_units, self.L5_3_num_units),nn.LeakyReLU(inplace=True, negative_slope=0.1) )
            self.L5_3_dropout = nn.Dropout(self.L5_3_dropout_rate)
            self.L5_4 = nn.Sequential(nn.Linear(feature_num, self.L4_num_units),  nn.LeakyReLU(inplace=True, negative_slope=0.1), nn.Linear(self.L4_num_units, self.L5_4_num_units),nn.LeakyReLU(inplace=True, negative_slope=0.1))
            self.L5_4_dropout = nn.Dropout(self.L5_4_dropout_rate)
            self.Y_indel_length_logits_1 = nn.Sequential( nn.Linear(self.L5_3_num_units, self.output_indel_length_shape_1))
            self.Y_indel_length_logits_2 = nn.Sequential( nn.Linear(self.L5_4_num_units, self.output_indel_length_shape_2))

        if False:
            # initialize parameters
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.zero_()

    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM, eps=BN_EPS))
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # input normalization by 100, 
        # x = x / param.NORMALIZE_NUM # do this in datasequence loading
        out = self.basic1(x)
        out = self.layer1(out)
        out = self.basic2(out)
        out = self.layer2(out)
        out = self.basic3(out)
        out = self.layer3(out)
        # out = self.basic4(out)
        # out = self.layer4(out)
        out = self.avgpool(out)
        # out = self.L3_dropout(out)
        out = out.view(out.size(0), -1)
        # out = self.L4(out)
        # out = self.L4_dropout(out)

        l5_1_dropout = self.L5_1_dropout(self.L5_1(out))
        l5_2_dropout = self.L5_2_dropout(self.L5_2(out))
        y_gt21_logits = self.Y_gt21_logits(l5_1_dropout)
        y_genotype_logits = self.Y_genotype_logits(l5_2_dropout)

        if self.add_indel_length:
            l5_3_dropout = self.L5_3_dropout(self.L5_3(out))

            l5_4_dropout = self.L5_4_dropout(self.L5_4(out))

            y_indel_length_logits_1 = self.Y_indel_length_logits_1(l5_3_dropout)

            y_indel_length_logits_2 = self.Y_indel_length_logits_2(l5_4_dropout)
            if self.predict:

                return torch.cat([y_gt21_logits, y_genotype_logits, y_indel_length_logits_1, y_indel_length_logits_2], axis=1)

            return [y_gt21_logits, y_genotype_logits, y_indel_length_logits_1, y_indel_length_logits_2]
        if self.predict:

            return torch.cat([y_gt21_logits, y_genotype_logits],axis=1)

        return [y_gt21_logits, y_genotype_logits]


class Clair3_Feature(nn.Module):
    def __init__(self,  layers, platform):
        super(Clair3_Feature, self).__init__()

        # model structure below
        if platform == "ilmn":
            self.basic1 = BasicConv2D(7, 64, stride=2)
        else:
            self.basic1 = BasicConv2D(8, 64, stride=2)
        self.layer1 = self.make_layer(ResidualBlock, 64, 64, layers[0])
        self.basic2 = BasicConv2D(64, 128, stride=2)
        self.layer2 = self.make_layer(ResidualBlock,128, 128, layers[1])
        self.basic3 = BasicConv2D(128, 256, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 256, layers[2])

        # self.avgpool = SpatialPyramidPooling([3,2,1]) #nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveMaxPool2d((1, 2))
        if False:
            # initialize parameters
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.zero_()
    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM, eps=BN_EPS))
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.basic1(x)
        out = self.layer1(out)
        out = self.basic2(out)
        out = self.layer2(out)
        out = self.basic3(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


class Clair3_cls(nn.Module):
    def __init__(self,  add_indel_length=False, predict=False, test_inception=False):
        super(Clair3_cls, self).__init__()
        # Parameter setup here
        self.output_gt21_shape = params['output_gt21_shape']
        self.output_genotype_shape = params['output_genotype_shape']
        self.output_indel_length_shape_1 = params['output_indel_length_shape_1']
        self.output_indel_length_shape_2 = params['output_indel_length_shape_2']
        self.L3_dropout_rate = params['L3_dropout_rate']
        self.L4_num_units = params['L4_num_units']
        self.L4_dropout_rate = params['L4_dropout_rate']
        self.L5_1_num_units = params['L5_1_num_units']
        self.L5_1_dropout_rate = params['L5_1_dropout_rate']
        self.L5_2_num_units = params['L5_2_num_units']
        self.L5_2_dropout_rate = params['L5_2_dropout_rate']
        self.L5_3_num_units = params['L5_3_num_units']
        self.L5_3_dropout_rate = params['L5_3_dropout_rate']
        self.L5_4_num_units = params['L5_4_num_units']
        self.L5_4_dropout_rate = params['L5_4_dropout_rate']
        self.add_indel_length = add_indel_length
        self.predict = predict
        # self.L3_dropout = nn.Dropout(self.L3_dropout_rate)
        
        # self.L4 = nn.Sequential( nn.Linear(self.avgpool.get_output_size(256), self.L4_num_units),nn.LeakyReLU(inplace=True, negative_slope=0.1) )
        feature_num = 2048 if test_inception else 512
        self.L4 = nn.Sequential( nn.Linear(feature_num, self.L4_num_units),nn.LeakyReLU(inplace=True, negative_slope=0.1) )

        self.L4_dropout = nn.Dropout(self.L4_dropout_rate)

        # output 
        self.L5_1 = nn.Sequential( nn.Linear(feature_num, self.L4_num_units),  nn.LeakyReLU(inplace=True, negative_slope=0.1), nn.Linear(self.L4_num_units, self.L5_1_num_units),nn.LeakyReLU(inplace=True, negative_slope=0.1) )
        
        self.L5_1_dropout = nn.Dropout(self.L5_1_dropout_rate)
        self.L5_2 = nn.Sequential( nn.Linear(feature_num, self.L4_num_units),  nn.LeakyReLU(inplace=True, negative_slope=0.1), nn.Linear(self.L4_num_units, self.L5_2_num_units),nn.LeakyReLU(inplace=True, negative_slope=0.1) )
        self.L5_2_dropout = nn.Dropout(self.L5_2_dropout_rate)

        self.Y_gt21_logits = nn.Sequential( nn.Linear(self.L5_1_num_units, self.output_gt21_shape))
        self.Y_genotype_logits = nn.Sequential( nn.Linear(self.L5_2_num_units, self.output_genotype_shape))

        if self.add_indel_length:
            self.L5_3 = nn.Sequential(nn.Linear(feature_num, self.L4_num_units),  nn.LeakyReLU(inplace=True, negative_slope=0.1), nn.Linear(self.L4_num_units, self.L5_3_num_units),nn.LeakyReLU(inplace=True, negative_slope=0.1) )
            self.L5_3_dropout = nn.Dropout(self.L5_3_dropout_rate)
            self.L5_4 = nn.Sequential(nn.Linear(feature_num, self.L4_num_units),  nn.LeakyReLU(inplace=True, negative_slope=0.1), nn.Linear(self.L4_num_units, self.L5_4_num_units),nn.LeakyReLU(inplace=True, negative_slope=0.1))
            self.L5_4_dropout = nn.Dropout(self.L5_4_dropout_rate)
            self.Y_indel_length_logits_1 = nn.Sequential( nn.Linear(self.L5_3_num_units, self.output_indel_length_shape_1))
            self.Y_indel_length_logits_2 = nn.Sequential( nn.Linear(self.L5_4_num_units, self.output_indel_length_shape_2))

        if False:
            # initialize parameters
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.zero_()

    def forward(self, x, reverse=False,  eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        # out = self.L3_dropout(x)
        out = x.view(x.size(0), -1)
        # out = self.L4(out)
        # out = self.L4_dropout(out)

        l5_1_dropout = self.L5_1_dropout(self.L5_1(out))
        l5_2_dropout = self.L5_2_dropout(self.L5_2(out))
        y_gt21_logits = self.Y_gt21_logits(l5_1_dropout)
        y_genotype_logits = self.Y_genotype_logits(l5_2_dropout)

        if self.add_indel_length:
            l5_3_dropout = self.L5_3_dropout(self.L5_3(out))

            l5_4_dropout = self.L5_4_dropout(self.L5_4(out))

            y_indel_length_logits_1 = self.Y_indel_length_logits_1(l5_3_dropout)

            y_indel_length_logits_2 = self.Y_indel_length_logits_2(l5_4_dropout)
            if self.predict:

                return torch.cat([y_gt21_logits, y_genotype_logits, y_indel_length_logits_1, y_indel_length_logits_2], axis=1)

            return [y_gt21_logits, y_genotype_logits, y_indel_length_logits_1, y_indel_length_logits_2]
        if self.predict:

            return torch.cat([y_gt21_logits, y_genotype_logits],axis=1)

        return [y_gt21_logits, y_genotype_logits]


class Clair3_P(nn.Module):
    # Bi-lstm model for clair3 pileup input
    def __init__(self, add_indel_length=False, predict=False):
        super(Clair3_P, self).__init__()

        # output
        self.output_gt21_shape = params['output_gt21_shape']
        self.output_genotype_shape = params['output_genotype_shape']
        self.output_indel_length_shape_1 = params['output_indel_length_shape_1']
        self.output_indel_length_shape_2 = params['output_indel_length_shape_2']

        self.L3_dropout_rate = params['L3_dropout_rate']
        self.L4_num_units = params['L4_num_units']
        self.L4_pileup_num_units = params['L4_pileup_num_units']
        self.L4_dropout_rate = params['L4_dropout_rate']
        self.L5_1_num_units = params['L5_1_num_units']
        self.L5_1_dropout_rate = params['L5_1_dropout_rate']
        self.L5_2_num_units = params['L5_2_num_units']
        self.L5_2_dropout_rate = params['L5_2_dropout_rate']
        self.L5_3_num_units = params['L5_3_num_units']
        self.L5_3_dropout_rate = params['L5_3_dropout_rate']
        self.L5_4_num_units = params['L5_4_num_units']
        self.L5_4_dropout_rate = params['L5_4_dropout_rate']
        self.LSTM1_num_units = params['LSTM1_num_units'] # 128
        self.LSTM2_num_units = params['LSTM2_num_units'] # 160
        self.LSTM1_dropout_rate = params['LSTM1_dropout_rate']
        self.LSTM2_dropout_rate = params['LSTM2_dropout_rate']

        self.output_label_split = [
            self.output_gt21_shape,
            self.output_genotype_shape,
            self.output_indel_length_shape_1,
            self.output_indel_length_shape_2
        ]

        self.add_indel_length = add_indel_length
        self.predict = predict

        self.LSTM1 = nn.LSTM(
            input_size= 18,
            hidden_size=self.LSTM1_num_units,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )

        self.LSTM2 = nn.LSTM(
            input_size=self.LSTM1_num_units*2 ,
            hidden_size=self.LSTM2_num_units,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )

        self.L3_dropout = nn.Dropout(p=self.L3_dropout_rate)

        self.L3_dropout_flatten = nn.Flatten()

        self.L4 = nn.Sequential( nn.Linear(in_features=10560, out_features=self.L4_pileup_num_units) , nn.SELU() )

        self.L4_dropout = nn.Dropout(p=self.LSTM2_dropout_rate)

        self.L5_1 = nn.Sequential(  nn.Linear(in_features=self.L4_pileup_num_units, out_features=self.L5_1_num_units) , nn.SELU())

        self.L5_1_dropout = nn.Dropout(p=self.L5_1_dropout_rate)

        self.L5_2 = nn.Sequential( nn.Linear(in_features=self.L5_1_num_units, out_features=self.L5_2_num_units), nn.SELU())

        self.L5_2_dropout = nn.Dropout(p=self.L5_2_dropout_rate)

        
        self.Y_gt21_logits = nn.Sequential( nn.Linear(self.L5_1_num_units, self.output_gt21_shape), nn.Softmax(dim=1))
        self.Y_genotype_logits = nn.Sequential( nn.Linear(self.L5_2_num_units, self.output_genotype_shape), nn.Softmax(dim=1))

        if self.add_indel_length:
            self.L5_3 = nn.Sequential( nn.Linear(self.L4_pileup_num_units, self.L5_3_num_units), nn.SELU() )
            self.L5_3_dropout = nn.Dropout(self.L5_3_dropout_rate)
            self.L5_4 = nn.Sequential( nn.Linear(self.L4_pileup_num_units, self.L5_4_num_units), nn.SELU() )
            self.L5_4_dropout = nn.Dropout(self.L5_4_dropout_rate)
            self.Y_indel_length_logits_1 = nn.Sequential( nn.Linear(self.L5_3_num_units, self.output_indel_length_shape_1),nn.Softmax(dim=1) )
            self.Y_indel_length_logits_2 = nn.Sequential( nn.Linear(self.L5_4_num_units, self.output_indel_length_shape_2), nn.Softmax(dim=1) )
        if False:
            # initialize parameters
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.zero_()
    def forward(self, x,):
        x, (h,c) = self.LSTM1(x)  # (batch_size, inp_seq_len, d_model)
        # 33x256 :  33 x 128 x 2
        x, (h,c) = self.LSTM2(x)
        # 33 x 320
        x = self.L3_dropout(x)

        x = self.L3_dropout_flatten(x)
        x = self.L4(x)

        x = self.L4_dropout(x)

        l5_1_dropout = self.L5_1_dropout(self.L5_1(x))

        l5_2_dropout = self.L5_2_dropout(self.L5_2(x))

        y_gt21_logits = self.Y_gt21_logits(l5_1_dropout)

        y_genotype_logits = self.Y_genotype_logits(l5_2_dropout)

        if self.add_indel_length:
            l5_3_dropout = self.L5_3_dropout(self.L5_3(x))

            l5_4_dropout = self.L5_4_dropout(self.L5_4(x))

            y_indel_length_logits_1 = self.Y_indel_length_logits_1(l5_3_dropout)

            y_indel_length_logits_2 = self.Y_indel_length_logits_2(l5_4_dropout)

            if self.predict:
                return torch.cat([y_gt21_logits, y_genotype_logits, y_indel_length_logits_1, y_indel_length_logits_2], axis=1)

            return [y_gt21_logits, y_genotype_logits, y_indel_length_logits_1, y_indel_length_logits_2]

        if self.predict:
            return torch.cat([y_gt21_logits, y_genotype_logits],axis=1)

        return [y_gt21_logits, y_genotype_logits]
