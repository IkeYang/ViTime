import torch
import torch.nn as nn
import torch.nn.functional as F

from ViTime.model.mobilenetv2 import mobilenetv2



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer_ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_ChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel * reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * reduction, channel * reduction, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * reduction, channel, 1, bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y

class MobileNetV2(nn.Module):
    def __init__(self, AdaFactor=1, downsample_factor=8, pretrained=True, image_C=3, dropout=0.2):
        super(MobileNetV2, self).__init__()
        model = mobilenetv2(AdaFactor=AdaFactor, pretrained=pretrained, image_C=image_C, dropout=dropout)
        self.features = model.features[:-1]
        self.in_channels = model.in_channels
        self.low_level_channels = model.low_level_channels

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]
        if downsample_factor == 4:
            for i in range(self.down_idx[-3], self.down_idx[-2]):
                self.features[i].apply(self._nostride_dilate(dilate=2))
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(self._nostride_dilate(dilate=1))
        elif downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(self._nostride_dilate(dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(self._nostride_dilate(dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(self._nostride_dilate(dilate=2))
        elif downsample_factor == 32:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(self._nostride_dilate(dilate=1))
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(self._nostride_dilate(dilate=2))
        elif downsample_factor == 64:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(self._nostride_dilate(dilate=1))

    def _nostride_dilate(self, dilate):
        def apply_fn(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if m.stride == (2, 2):
                    m.stride = (1, 1)
                    if m.kernel_size == (3, 3):
                        m.dilation = (dilate // 2, dilate // 2)
                        m.padding = (dilate // 2, dilate // 2)
                else:
                    if m.kernel_size == (3, 3):
                        m.dilation = (dilate, dilate)
                        m.padding = (dilate, dilate)
        return apply_fn

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x

class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        global_feature = torch.mean(x, dim=(2, 3), keepdim=True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, size=x.shape[2:], mode='bilinear', align_corners=True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

class DeconvASPP(nn.Module):
    def __init__(self, dim_in, dim_out, bn_mom=0.1):
        super(DeconvASPP, self).__init__()

        self.branch1 = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, 2, stride=4, bias=True, dilation=1),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, 2, stride=4, padding=1, bias=True, dilation=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, 2, stride=4, padding=2, bias=True, dilation=3),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, 2, stride=4, padding=3, bias=True, dilation=4),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim_in, dim_out, 1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.conv_cat = nn.Sequential(
            SELayer_ChannelAttention(dim_out * 6, reduction=4),
        )

    def forward(self, x, size=None):
        if size is None:
            w, h = x.shape[-2] * 4, x.shape[-1] * 4
        else:
            w, h = size

        deconv1 = self.branch1(x)
        deconv2 = self.branch2(x)
        deconv3 = self.branch3(x)
        deconv4 = self.branch4(x)
        global_feature = self.branch5(x)
        global_feature = F.interpolate(global_feature, size=(w, h), mode='bilinear', align_corners=False)

        deconv1 = F.interpolate(deconv1, size=(w, h), mode='bilinear', align_corners=False)
        deconv2 = F.interpolate(deconv2, size=(w, h), mode='bilinear', align_corners=False)
        deconv3 = F.interpolate(deconv3, size=(w, h), mode='bilinear', align_corners=False)
        deconv4 = F.interpolate(deconv4, size=(w, h), mode='bilinear', align_corners=False)
        deconv5 = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=True)

        deconv_cat = torch.cat([deconv1, deconv2, deconv3, deconv4, global_feature, deconv5], dim=1)
        result = torch.sum(self.conv_cat(deconv_cat), dim=1, keepdim=True)

        return result

class RefiningModel(nn.Module):
    def __init__(self,  downsample_factor=16, dropout=0.1, args=None):
        super(RefiningModel, self).__init__()
        num_classes=1
        image_C=num_classes
        pretrained=False
        modelSize = args.modelSize if args else 1
        self.DO_ASPP = getattr(args, 'aspp', True)
        self.DO_LOWLEVEL = getattr(args, 'lowlevel', True)

        AdaFactor = args.modelSize / 20 if getattr(args, 'modelAda', False) else modelSize
        if AdaFactor <= (1 / 8):
            AdaFactor = 1 / 8
        dim_out = int(8 * 30 * AdaFactor)
        low_level_channels_out = int(8 * 30 * 1 / 5 * AdaFactor)

        self.backbone = MobileNetV2(AdaFactor=AdaFactor, downsample_factor=downsample_factor, pretrained=pretrained, image_C=image_C, dropout=dropout)
        in_channels = self.backbone.in_channels
        low_level_channels = self.backbone.low_level_channels


        self.aspp = ASPP(dim_in=in_channels, dim_out=dim_out, rate=16 // downsample_factor)
        self.upsampleASPP = DeconvASPP(dim_in=num_classes, dim_out=num_classes)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_channels_out, 1),
            nn.BatchNorm2d(low_level_channels_out),
            nn.ReLU(inplace=True)
        )
        if not self.DO_ASPP:
            dim_out = in_channels
        catin = low_level_channels_out + dim_out if self.DO_LOWLEVEL else dim_out
        self.cat_conv = nn.Sequential(
            nn.Conv2d(catin, dim_out, 3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls_conv = nn.Conv2d(dim_out, num_classes, 1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        low_level_features, x = self.backbone(x)
        if self.DO_ASPP:
            x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        if self.DO_LOWLEVEL:
            x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        else:
            x = self.cat_conv(x)
        x = self.cls_conv(x)
        x = self.upsampleASPP(x, size=(H, W))

        return x
