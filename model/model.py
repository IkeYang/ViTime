# Author: Luoxiao Yang
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model.RefiningModel import RefiningModel
from model.ViTimeAutoencoder import ViTimeAutoencoder
import copy
from datafactory.loadData import Dataset_ViTime


class ViTime(nn.Module):
    """
    A combined model using Masked Autoencoder (MAE) and DeepLab for image processing.
    """

    def __init__(self, args=None):
        super().__init__()
        MAE_Modelsize = copy.deepcopy(args.modelSize)
        args.modelAda = True
        self.args = args
        self.model = ViTimeAutoencoder(args=args
        )
        args.modelSize = 40
        self.RefiningModel = RefiningModel(
  
            downsample_factor=args.downsample_factor,
            dropout=args.dropout, args=args
        )
        self.EMD = nn.Softmax(dim=-1)
        args.modelSize = MAE_Modelsize
        self.dataTool=Dataset_ViTime(args)
        self.device=args.device

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
        x = x.view(bs * c, 1, w, h)
        mask = torch.ones_like(x[0, :])
        mask[:, :self.args.size[0], :] = 0
        mask = mask.to(x.device)

        xO = copy.deepcopy(x)
        x = self.model(x)
        x = self.EMD(x / 10)
        x = 20 * x * mask + xO
        x = self.RefiningModel(x)
        x = self.EMD(x / temparture)
        x = x.view(bs, c, w, h)
        return x


    def inference(self, data_x):
        if len(data_x.shape)==1:
            data_x=data_x.reshape(1,-1,1)
        elif len(data_x.shape) == 2:
            T,C=data_x.shape
            data_x = data_x.reshape(1, T, C)

        x,d,mu,std=self.dataTool.dataTransformationBatch(data_x)

        xInput = x.to(self.device)
        x = self.forward(xInput).detach().cpu().numpy()

        # ypredMax = self.dataTool.Pixel2data(x, method='max')
        ypredExp = self.dataTool.Pixel2data(x, method='expection')

        yp = (ypredExp[:, self.args.size[0]:self.args.size[0] + self.args.size[2], :] * std + mu)
        if self.args.upscal:
            yp = yp[:, 1::2, :]

        return yp
