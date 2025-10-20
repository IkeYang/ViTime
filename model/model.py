# Author: Luoxiao Yang
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model.RefiningModel import RefiningModel
from model.ViTimeAutoencoder import ViTimeAutoencoder
import copy
from datafactory.loadData import Dataset_ViTime
import pickle
from scipy import interpolate
import numpy as np
from model.modelMAE import MaskedAutoencoderViT


class ChannelFusionBlock(nn.Module):
    """
    A three-layer convolutional block to fuse multiple input channels into one.
    Uses a bottleneck: expand -> depthwise conv -> compress.
    """
    def __init__(self, in_channels=2, out_channels=1, mid_channels=8):
        super().__init__()
        self.block = nn.Sequential(
            # 1. Expansion: 1x1 conv to expand channels to mid_channels
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # 2. Processing: 3x3 depthwise separable conv for spatial features
            # groups=mid_channels makes it depthwise
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # 3. Compression: 1x1 conv to reduce to out_channels
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=True)
        )

    def forward(self, x):
        return self.block(x)


class ViTime(nn.Module):
    """
    A combined model using Masked Autoencoder (MAE) and DeepLab for image processing.
    """

    def __init__(self, args=None):
        super().__init__()
        MAE_Modelsize = copy.deepcopy(args.modelSize)
        args.modelAda = False
        self.args = args
        self.model = ViTimeAutoencoder(args=args)
        args.modelSize = 40
        if hasattr(args, 'RefiningModel'):
            if not args.RefiningModel:
                self.RefiningModelState = False
                self.RefiningModel = nn.Identity()
            else:
                self.RefiningModelState = True
                self.RefiningModel = RefiningModel(
                    downsample_factor=args.downsample_factor,
                    dropout=args.dropout, args=args
                )
        else:
            self.RefiningModelState = True
            self.RefiningModel = RefiningModel(
                downsample_factor=args.downsample_factor,
                dropout=args.dropout, args=args
            )
        self.EMD = nn.Softmax(dim=-1)
        args.modelSize = MAE_Modelsize
        self.dataTool = Dataset_ViTime(args)
        self.device = args.device
        self.dataTool.inferenceType = 'General'
        self.dataTool.EnvelopeUpper = None
        self.dataTool.EnvelopeLower = None
        self.dataTool.KeyP = None
        print(self.RefiningModelState)
        if hasattr(args, 'MA_STD_KeyP') or hasattr(args, 'Env_double') or hasattr(args, 'MA_KeyP'):
            self.channel_combiner = ChannelFusionBlock(in_channels=3, out_channels=1, mid_channels=24)
        elif hasattr(args, 'MA_STD') or hasattr(args, 'MA') or hasattr(args, 'KeyP'):
            self.channel_combiner = ChannelFusionBlock(in_channels=2, out_channels=1, mid_channels=16)
        elif hasattr(args, 'Env_KeyP_MA'):
            self.channel_combiner = ChannelFusionBlock(in_channels=4, out_channels=1, mid_channels=32)
        else:
            self.channel_combiner = None

    def forward(self, x, temparture=1):
        """
        Forward pass of the combined model.

        Parameters:
        x (torch.Tensor): Input tensor.
        temparture (float): Temperature for softmax scaling.

        Returns:
        torch.Tensor: Output tensor.
        """
        bs, c, w, h = x.shape
        if c == 2 or c == 3:
            x = self.channel_combiner(x)
            c = 1  # update channel count
        x = x.view(bs * c, 1, w, h)
        mask = torch.ones_like(x[0, :])
        mask[:, :self.args.size[0], :] = 0
        mask = mask.to(x.device)

        x = self.model(x)

        if self.RefiningModelState:
            x = self.EMD(x / 10)
            x = 20 * x * mask + xO

            x = self.RefiningModel(x)
            x2 = self.EMD(x / 10)
            x = self.EMD(x / temparture)
            x = x.view(bs, c, w, h)
            return x
        else:
            x = self.EMD(x / temparture)
            x = x.view(bs, c, w, h)
            return x


class MAE(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, inchannel=1, T=24 * 4 * 5, out_channels=1, loss='MSE', args=None, DCNumber=3):
        super().__init__()  # xception mobilenet
        args.modelAda = True
        self.args = args
        self.model = MaskedAutoencoderViT(num_classes=1, image_C=1, dropout=args.dropout,
                                          args=args)

        self.EMD = nn.Softmax(dim=-1)
        self.Sigmoid = nn.Sigmoid()
        self.dataTool = Dataset_ViTime(args)
        self.device = args.device
        self.dataTool.inferenceType = 'General'
        self.dataTool.EnvelopeUpper = None
        self.dataTool.EnvelopeLower = None
        self.dataTool.KeyP = None
        if hasattr(args, 'MA_STD_KeyP') or hasattr(args, 'Env_double') or hasattr(args, 'MA_KeyP'):
            self.channel_combiner = ChannelFusionBlock(in_channels=3, out_channels=1, mid_channels=24)
        elif hasattr(args, 'MA_STD') or hasattr(args, 'MA'):
            self.channel_combiner = ChannelFusionBlock(in_channels=2, out_channels=1, mid_channels=16)
        elif hasattr(args, 'Env_KeyP_MA'):
            self.channel_combiner = ChannelFusionBlock(in_channels=4, out_channels=1, mid_channels=32)
        else:
            self.channel_combiner = None

    def forward(self, x, temparture=1):
        bs, c, w, h = x.shape

        if c >= 2 and self.args.features == 'S':
            x = self.channel_combiner(x)
            c = 1  # update channel count
        x = x.view(bs * c, 1, w, h)
        x_copy = copy.deepcopy(x)
        x = self.model(x)
        if hasattr(self.args, 'lossType'):
            if self.args.lossType == 'MSE':
                x = self.Sigmoid(x)
            else:
                x = self.EMD(x / temparture)
        else:
            x = self.EMD(x / temparture)
        x = x.view(bs, c, w, h)
        return x

    def forward2(self, x, temparture=1):
        """
        Forward pass of the combined model.

        Parameters:
        x (torch.Tensor): Input tensor.
        temparture (float): Temperature for softmax scaling.

        Returns:
        torch.Tensor: Output tensor.
        """
        bs, c, w, h = x.shape
        x = x.view(bs * c, 1, w, h)
        mask = torch.ones_like(x[0, :])
        mask[:, :self.args.size[0], :] = 0
        mask = mask.to(x.device)

        xO = copy.deepcopy(x)
        x = self.model(x)

        x = self.EMD(x / 10)

        x = 20 * x * mask + xO

        x = self.RefiningModel(x)
        return x

    def inference(self, data_x, mu=None, std=None, returnFullSequence=False, inferenceType='General'):
        self.inferenceType = inferenceType
        self.dataTool.mu = mu
        self.dataTool.std = std
        if len(data_x.shape) == 1:
            data_x = data_x.reshape(1, -1, 1)
        elif len(data_x.shape) == 2:
            T, C = data_x.shape
            data_x = data_x.reshape(1, T, C)

        x, d, mu, std = self.dataTool.dataTransformationBatch(data_x)
        xInput = x.to(self.device)

        xpred = self.forward(xInput).detach().cpu().numpy()

        ypredMax = self.dataTool.Pixel2data(xpred, method='max')
        ypredExp = self.dataTool.Pixel2data(xpred, method='expection')

        yp = (ypredExp[:, self.args.size[0]:self.args.size[0] + self.args.size[2], :] * std + mu)
        ypFull = ypredExp * std + mu
        if self.args.upscal:
            yp = yp[:, 1::2, :]
            ypFull = ypFull[:, 1::2, :]
        if returnFullSequence:
            return yp, ypFull
        else:
            return yp

    def inferenceDecomp(self, data_x, mu=None, std=None):
        self.dataTool.mu = mu
        self.dataTool.std = std
        self.dataTool.inferenceType = 'Decomp'

        if len(data_x.shape) == 1:
            data_x = data_x.reshape(1, -1, 1)
        elif len(data_x.shape) == 2:
            T, C = data_x.shape
            data_x = data_x.reshape(1, T, C)

        x, d, mu, std = self.dataTool.dataTransformationBatch(data_x)
        xInput = x.to(self.device)

        x = self.forward(xInput).detach().cpu().numpy()

        ypredExp = self.dataTool.Pixel2data(x, method='expection')

        yp = (ypredExp * std + mu)

        return yp

    def inferenceKeyP(self, data_x, KeyP, returnFullSequence=False):

        self.dataTool.mu = None
        self.dataTool.std = None
        self.dataTool.KeyP = KeyP
        if KeyP is None:
            self.dataTool.inferenceType = 'General'
        else:
            self.dataTool.inferenceType = 'KeyP'

        if len(data_x.shape) == 1:
            data_x = data_x.reshape(1, -1, 1)
        elif len(data_x.shape) == 2:
            T, C = data_x.shape
            data_x = data_x.reshape(1, T, C)

        x, d, mu, std = self.dataTool.dataTransformationBatch(data_x)
        xInput = x.to(self.device)

        xpred = self.forward(xInput).detach().cpu().numpy()

        ypredMax = self.dataTool.Pixel2data(xpred, method='max')
        ypredExp = self.dataTool.Pixel2data(xpred, method='expection')

        yp = (ypredMax[:, self.args.size[0]:self.args.size[0] + self.args.size[2], :] * std + mu)
        ypFull = ypredExp * std + mu
        if self.args.upscal:
            yp = yp[:, 1::2, :]
            ypFull = ypFull[:, 1::2, :]
        if returnFullSequence:
            return yp, ypFull
        else:
            return yp

    def inferenceEnvelope(self, data_x, mu=None, std=None, returnFullSequence=False, EnvelopeUpper=None,
                          EnvelopeLower=None):
        from anomaly import filter_outliers_3sigma_interpolate_continuous
        EnvelopeUpper = filter_outliers_3sigma_interpolate_continuous(EnvelopeUpper)
        EnvelopeLower = filter_outliers_3sigma_interpolate_continuous(EnvelopeLower)
        self.dataTool.inferenceType = 'Envelope'
        self.dataTool.mu = mu
        self.dataTool.std = std
        self.dataTool.EnvelopeUpper = EnvelopeUpper
        self.dataTool.EnvelopeLower = EnvelopeLower
        if len(data_x.shape) == 1:
            data_x = data_x.reshape(1, -1, 1)
        elif len(data_x.shape) == 2:
            T, C = data_x.shape
            data_x = data_x.reshape(1, T, C)

        x, d, mu, std = self.dataTool.dataTransformationBatch(data_x)
        xInput = x.to(self.device)

        xpred = self.forward(xInput).detach().cpu().numpy()

        ypredMax = self.dataTool.Pixel2data(xpred, method='max') * std + mu
        ypredExp = self.dataTool.Pixel2data(xpred, method='expection') * std + mu
        if self.args.upscal:
            ypredMax = ypredMax[:, 1::2, :]
            ypredExp = ypredExp[:, 1::2, :]
        # envelope pred
        yp = (ypredExp[:, self.args.size[0]:self.args.size[0] + self.args.size[2], :])
        ypFull = ypredExp

        if returnFullSequence:
            return yp, ypFull
        else:
            return yp

def modelDict():
    mdict = {
        'MAE': MAE,
    }
    return mdict
