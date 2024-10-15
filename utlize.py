#Author:ike yang
from typing import List
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import os
import torch
import torch.nn as nn
import pynvml
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def paddingData(data,args):
    value = 0
    # value = 0
    bs=data.shape[0]
    dataLen=data.shape[1]
    newdata=np.ones([bs,args.size[0]+args.size[2]])*value
    newdata[:,:dataLen]=data
    return newdata
def data2Pixel(args, dataXIn,declineFactor=1):
    '''

    :param dataX: tin,pX dataY: whole,pY
    :return: imgX, (w,h,px)*C imgY, w,h,pY
    '''

    dataX = np.copy(dataXIn)
    dataY = paddingData(dataXIn,args=args)
    maxScal=args.maxScal*declineFactor
    dataX[dataX > maxScal] = maxScal
    dataX[dataX < -maxScal] = -maxScal

    dataY[dataY > maxScal] = maxScal
    dataY[dataY < -maxScal] = -maxScal
    px = dataX.shape[0]
    py = dataY.shape[0]
    TY = dataY.shape[1]
    TX = dataX.shape[1]



    imgY0 = np.zeros([py, TY, args.h])

    maxstd = maxScal
    resolution = maxstd * 2 / (args.h - 1)
    indY = np.floor((dataY + maxstd) / resolution).astype('int16')

    aY = imgY0
    aY =aY.reshape(-1, args.h)
    aY[np.arange(TY * py), indY.astype('int16').flatten()] = 1
    imgY0= aY.reshape(py, TY, args.h)


    imgX0=np.copy(imgY0)

    imgX0[:,TX:,:]=0

    return imgX0.reshape(1,py, TY, args.h)

def pixel2Data(self, imgX0, method='expection',declineFactor=1):
    if len(imgX0.shape) == 3:
        imgX0 = imgX0.unsqueeze(0)
    # bs,c,w,h imgX0
    bs, ch, w, h = imgX0.shape
    # res=np.zeros([bs,w,ch])
    try:
        imgX0 = imgX0.cpu().detach().numpy()
    except:
        pass
    if method == 'max':
        indx = np.argmax(imgX0, axis=-1)
    elif method == 'expection':
        imgX0 = imgX0 / np.sum(imgX0, axis=-1, keepdims=True)
        indNumber = np.arange(0, h)
        imgX0 *= indNumber
        indx = np.sum(imgX0, axis=-1)
    maxstd = self.maxScal*declineFactor
    resolution = maxstd * 2 / (self.h - 1)
    res = np.transpose(indx, (0, 2, 1)) * resolution - maxstd


    return res
def GPUMemoryCheck(devicenumber=0):
     
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(devicenumber) # 0表示显卡标号
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print('total',meminfo.total/1024**2,'Used',meminfo.used/1024**2,'GPU-Util',utilization.gpu) #总的显存大小
  
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print(
            "---  new folder...  ---")
        print(
            "---  OK  ---")
    else:
        print(
            "---  There is this folder!  ---")
class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

class EMD():
    def __init__(self,MSE=False,onlyOut=False,arg=None):
        self.MSE=MSE
        self.onlyOut=onlyOut
        self.size=arg.size

        if MSE:
            self.mseloss=nn.MSELoss()
        
    def __call__(self,pred,d,yt):
        #d bs ,c, t,h 
        if self.MSE:
            if self.onlyOut:
                return torch.mean(pred[:,:,self.size[0]:,:]*d[:,:,self.size[0]:,:])*100+self.mseloss(pred[:,:,self.size[0]:,:],yt[:,:,self.size[0]:,:])*100
            else:
                return torch.mean(pred*d)*100+self.mseloss(pred,yt)*100
        else:
            if self.onlyOut:
                return torch.mean(pred[:,self.size[0]-self.size[1]:,:]*d[:,self.size[0]-self.size[1]:,:])
            else:
                return torch.mean(pred*d)*10


class EMD_Cycle():
    def __init__(self,args=None):

        self.args = args
        self.size = args.size
        self.focus=args.focus
        self.cycleStep=int((args.size[-1]+self.size[1])/args.cycleTime)
        self.normL=np.linspace(6,0.5,args.cycleTime)
    def __call__(self, pred, d, cycleNumber):
        # d bs ,c, t,h
        # norm=self.normL[cycleNumber-1]
        norm=1
        # if self.args.lossType=='EMD':
        # ind=torch.where(d==0)
        # size=d.shape[0]*d.shape[-1]
        # loss2=pred[ind]*torch.log(2*pred[ind])-(pred[ind]+1)*torch.log(pred[ind]+1)/size
        baseloss=torch.mean(pred[:,:, self.size[0] - self.size[1]:, :] * d[:,:, self.size[0] - self.size[1]:, :]**norm)
        # baseloss=0

        addloss=torch.mean(pred[:,:,self.size[0] - self.size[1]:self.size[0] - self.size[1]+self.cycleStep*cycleNumber, :] * d[:,:, self.size[0] - self.size[1]:self.size[0] - self.size[1]+self.cycleStep*cycleNumber, :]**norm)
        # elif self.args.lossType=='MSE':
        #     baseloss=torch.mean(pred[:,:, self.size[0] - self.size[1]:, :] * d[:,:, self.size[0] - self.size[1]:, :]**norm)
        #     # baseloss=0
        #
        #     addloss = torch.mean(
        #         pred[:, :, self.size[0] - self.size[1]:self.size[0] - self.size[1] + self.cycleStep * cycleNumber,
        #         :] * d[:, :, self.size[0] - self.size[1]:self.size[0] - self.size[1] + self.cycleStep * cycleNumber,
        #              :] ** norm)


        return baseloss+self.focus*addloss

class EMD_Cycle_JSD():
    def __init__(self,args=None):

        self.args = args
        self.size = args.size
        self.focus=args.focus
        self.cycleStep=int((args.size[-1]+self.size[1])/args.cycleTime)
        self.normL=np.linspace(6,0.5,args.cycleTime)
    def __call__(self, pred, d, cycleNumber):
        # d bs ,c, t,h
        # norm=self.normL[cycleNumber-1]
        norm=1
        # if self.args.lossType=='EMD':
        ind=torch.where(d==0)
        size=d.shape[0]*d.shape[-1]
        loss2=torch.mean(pred[ind]*torch.log(2*pred[ind]+1e-7)-(pred[ind]+1)*torch.log(pred[ind]+1))+0.693

        baseloss=torch.mean(pred[:,:, self.size[0] - self.size[1]:, :] * d[:,:, self.size[0] - self.size[1]:, :]**norm)
        # baseloss=0

        addloss=torch.mean(pred[:,:,self.size[0] - self.size[1]:self.size[0] - self.size[1]+self.cycleStep*cycleNumber, :] * d[:,:, self.size[0] - self.size[1]:self.size[0] - self.size[1]+self.cycleStep*cycleNumber, :]**norm)
        # elif self.args.lossType=='MSE':
        #     baseloss=torch.mean(pred[:,:, self.size[0] - self.size[1]:, :] * d[:,:, self.size[0] - self.size[1]:, :]**norm)
        #     # baseloss=0
        #
        #     addloss = torch.mean(
        #         pred[:, :, self.size[0] - self.size[1]:self.size[0] - self.size[1] + self.cycleStep * cycleNumber,
        #         :] * d[:, :, self.size[0] - self.size[1]:self.size[0] - self.size[1] + self.cycleStep * cycleNumber,
        #              :] ** norm)


        return baseloss+loss2/5


class EMD_Cycle_JSD_Extand():
    def __init__(self, args=None):
        self.args = args
        self.size = args.size
        self.focus = args.focus
        self.cycleStep = int((args.size[-1] + self.size[1]) / args.cycleTime)
        self.normL = np.linspace(6, 0.5, args.cycleTime)
        self.ExtandLength=int(args.h/100)
    def __call__(self, pred, d, cycleNumber):
        # d bs ,c, t,h
        # norm=self.normL[cycleNumber-1]
        norm = 1
        # if self.args.lossType=='EMD':
        ind = torch.where(d == 0)
        size = d.shape[0] * d.shape[-1]

        loss2 = torch.mean(
            pred[ind] * torch.log(2 * pred[ind] + 1e-7) - (pred[ind] + 1) * torch.log(pred[ind] + 1)) + 0.693
        for dExtend in range(self.ExtandLength):
            ind3=ind[3]+dExtend+1
            ind3[ind3 > (self.args.h - 1)] = self.args.h - 1
            loss2 += torch.mean(
                pred[ind3] * torch.log(2 * pred[ind3] + 1e-7) - (pred[ind3] + 1) * torch.log(pred[ind3] + 1)) + 0.693

            ind3 = ind[3] - dExtend - 1
            ind3[ind3 <0] = 0
            loss2 += torch.mean(
                pred[ind3] * torch.log(2 * pred[ind3] + 1e-7) - (pred[ind3] + 1) * torch.log(pred[ind3] + 1)) + 0.693


        baseloss = torch.mean(
            pred[:, :, self.size[0] - self.size[1]:, :] * d[:, :, self.size[0] - self.size[1]:, :] ** norm)
        # baseloss=0

        addloss = torch.mean(
            pred[:, :, self.size[0] - self.size[1]:self.size[0] - self.size[1] + self.cycleStep * cycleNumber, :] * d[:,
                                                                                                                    :,
                                                                                                                    self.size[
                                                                                                                        0] -
                                                                                                                    self.size[
                                                                                                                        1]:
                                                                                                                    self.size[
                                                                                                                        0] -
                                                                                                                    self.size[
                                                                                                                        1] + self.cycleStep * cycleNumber,
                                                                                                                    :] ** norm)
        # elif self.args.lossType=='MSE':
        #     baseloss=torch.mean(pred[:,:, self.size[0] - self.size[1]:, :] * d[:,:, self.size[0] - self.size[1]:, :]**norm)
        #     # baseloss=0
        #
        #     addloss = torch.mean(
        #         pred[:, :, self.size[0] - self.size[1]:self.size[0] - self.size[1] + self.cycleStep * cycleNumber,
        #         :] * d[:, :, self.size[0] - self.size[1]:self.size[0] - self.size[1] + self.cycleStep * cycleNumber,
        #              :] ** norm)

        return baseloss + loss2 / 5/self.ExtandLength

class EMD_Cycle_onlyout():
    def __init__(self,args=None):

        self.args = args
        self.size = args.size
        self.focus=args.focus
        self.cycleStep=int((args.size[-1]+self.size[1])/args.cycleTime)
        self.normL=np.linspace(6,0.5,args.cycleTime)
    def __call__(self, pred, d, cycleNumber):
        # d bs ,c, t,h
        # norm=self.normL[cycleNumber-1]
        norm=1
        # if self.args.lossType=='EMD':
        # baseloss=torch.mean(pred[:,:, self.size[0] - self.size[1]:, :] * d[:,:, self.size[0] - self.size[1]:, :]**norm)
        # # baseloss=0
        #
        # addloss=torch.mean(pred[:,:,self.size[0] - self.size[1]:self.size[0] - self.size[1]+self.cycleStep*cycleNumber, :] * d[:,:, self.size[0] - self.size[1]:self.size[0] - self.size[1]+self.cycleStep*cycleNumber, :]**norm)
        # # elif self.args.lossType=='MSE':
        # #     baseloss=torch.mean(pred[:,:, self.size[0] - self.size[1]:, :] * d[:,:, self.size[0] - self.size[1]:, :]**norm)
        # #     # baseloss=0
        # #
        # #     addloss = torch.mean(
        # #         pred[:, :, self.size[0] - self.size[1]:self.size[0] - self.size[1] + self.cycleStep * cycleNumber,
        # #         :] * d[:, :, self.size[0] - self.size[1]:self.size[0] - self.size[1] + self.cycleStep * cycleNumber,
        # #              :] ** norm)


        return torch.mean(pred * d[:,:, self.size[0] - self.size[1]:, :]**norm)

# class EMD():
#     def __init__(self,arg=None):
#
#         self.size = arg.size
#
#
#     def __call__(self, pred, d, yt):
#
#         return torch.mean(pred[:, self.size[0] - self.size[1]:, :] * d[:, self.size[0] - self.size[1]:, :])

class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
import torch
import torch.nn as nn
import torch.nn.functional as F
