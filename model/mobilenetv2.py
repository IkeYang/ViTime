import math
import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BatchNorm2d = nn.BatchNorm2d

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                #--------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                #--------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #-----------------------------------#
                #   利用1x1卷积进行通道数的调整
                #-----------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                #-----------------------------------#
                #   利用1x1卷积进行通道数的上升
                #-----------------------------------#
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #--------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                #--------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #-----------------------------------#
                #   利用1x1卷积进行通道数的下降
                #-----------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.,image_C=3,dropout=0.2,AdaFactor=1):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual

        input_channel = int(32*AdaFactor)
        print('AdaFactor',AdaFactor,'input_channel',input_channel)
        interverted_residual_setting = [
            # t, c, n, s
            [1, int(16*AdaFactor), 1, 1], # 256, 256, 32 -> 256, 256, 16
            [6, int(24*AdaFactor), 2, 2], # 256, 256, 16 -> 128, 128, 24   2
            [6, int(32*AdaFactor), 3, 2], # 128, 128, 24 -> 64, 64, 32     4
            [6, int(64*AdaFactor), 4, 2], # 64, 64, 32 -> 32, 32, 64       7
            [6, int(96*AdaFactor), 3, 1], # 32, 32, 64 -> 32, 32, 96
            [6, int(160*AdaFactor), 3, 2], # 32, 32, 96 -> 16, 16, 160     14
            [6, int(320*AdaFactor), 1, 1], # 16, 16, 160 -> 16, 16, 320
        ]

        self.in_channels=interverted_residual_setting[-1][1]
        self.low_level_channels=interverted_residual_setting[1][1]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)

        # 512, 512, 3 -> 256, 256, 32
        self.features = [conv_bn(image_C, input_channel, 2)]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        ##### dont work
        last_channel = 1280
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))



        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()






class InvertedResidual_2(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,ks):
        super(InvertedResidual_2, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                #--------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                #--------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #-----------------------------------#
                #   利用1x1卷积进行通道数的调整
                #-----------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                #-----------------------------------#
                #   利用1x1卷积进行通道数的上升
                #-----------------------------------#
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #--------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                #--------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, ks, stride, int((ks-1)/2), groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #-----------------------------------#
                #   利用1x1卷积进行通道数的下降
                #-----------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.,image_C=3,dropout=0.2,AdaFactor=1):
        super(MobileNetV2_2, self).__init__()
        block = InvertedResidual_2

        input_channel = int(32*AdaFactor)
        print('AdaFactor',AdaFactor,'input_channel',input_channel)
        interverted_residual_setting = [
            # t, c, n, s
            [1, int(16*AdaFactor), 1, 1, 3], # 256, 256, 32 -> 256, 256, 16
            [6, int(24*AdaFactor), 2, 2, 7], # 256, 256, 16 -> 128, 128, 24   2
            [6, int(32*AdaFactor), 3, 2, 11], # 128, 128, 24 -> 64, 64, 32     4
            [6, int(64*AdaFactor), 4, 2, 15], # 64, 64, 32 -> 32, 32, 64       7
            [6, int(96*AdaFactor), 3, 1, 15], # 32, 32, 64 -> 32, 32, 96
            [6, int(160*AdaFactor), 3, 2, 15], # 32, 32, 96 -> 16, 16, 160     14
            [6, int(320*AdaFactor), 1, 1, 15], # 16, 16, 160 -> 16, 16, 320
        ]

        self.in_channels=interverted_residual_setting[-1][1]
        self.low_level_channels=interverted_residual_setting[1][1]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)

        # 512, 512, 3 -> 256, 256, 32
        self.features = [conv_bn(image_C, input_channel, 2)]

        for t, c, n, s ,ks in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t,ks=ks))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t,ks=ks))
                input_channel = output_channel

        ##### dont work
        last_channel = 1280
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))



        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()















def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url,model_dir=model_dir)




def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(n_class=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar'), strict=False)
    return model



def mobilenetv2_2(pretrained=False, **kwargs):
    model = MobileNetV2_2(n_class=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar'), strict=False)
    return model










if __name__ == "__main__":
    model = mobilenetv2()
    for i, layer in enumerate(model.features):
        print(i, layer)
