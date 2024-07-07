import os
import numpy as np
import pandas as pd
import torch
import sys
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from functools import partial
import warnings
import cv2
import pickle
from PyEMD import EMD
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt






# Custom dataset for VR
class Dataset_ViTime(Dataset):
    def __init__(self, args):
        self.args = args
        size = args.size
        self.flag = args.flag
        self.h = args.h
        self.maxScal = args.maxScal
        self.target = args.target
        if self.args.upscal:
            self.seq_len = int(size[0] / 2)
            self.label_len = int(size[1] / 2)
            self.pred_len = int(size[2] / 2)
        else:
            self.seq_len = int(size[0] )
            self.label_len = int(size[1] )
            self.pred_len = int(size[2] )
        self.Norm = args.dNorm
        self.__prepareD__()

    def __prepareD__(self):
        self.taskType = 'regression'
        self.D = np.zeros([self.h, self.h])
        for i in range(self.h):
            self.D[i, :i] = np.arange(1, i + 1)[::-1]
            self.D[i, i:] = np.arange(0, self.h - i)
        self.D = self.D ** self.Norm


    def data2Pixel(self, dataXIn, dataYIN):
        if dataYIN is None:
            dataX = np.clip(dataXIn.T, -self.maxScal, self.maxScal)
            px,TX = dataX.shape
            imgX0 = np.zeros([px, TX, self.h])
            resolution = self.maxScal * 2 / (self.h - 1)
            indX = np.floor((dataX + self.maxScal) / resolution).astype('int16')
            aX = imgX0.reshape(-1, self.h)
            aX[np.arange(TX * px), indX.astype('int16').flatten()] = 1
            imgX0 = aX.reshape(px, TX, self.h)
            d = self.D[list(indX), :]
            return imgX0, d
        else:

            dataX = np.clip(dataXIn.T, -self.maxScal, self.maxScal)
            dataY = np.clip(dataYIN.T, -self.maxScal, self.maxScal)
            px, py = dataX.shape[0], dataY.shape[0]
            TY, TX = dataY.shape[1], dataX.shape[1]


            imgY0 = np.zeros([py, TY, self.h])
            resolution = self.maxScal * 2 / (self.h - 1)
            indY = np.floor((dataY + self.maxScal) / resolution).astype('int16')
            aY = imgY0.reshape(-1, self.h)
            aY[np.arange(TY * py), indY.astype('int16').flatten()] = 1
            imgY0 = aY.reshape(py, TY, self.h)
            d = self.D[list(indY), :]
            imgX0 = np.copy(imgY0)
            imgX0[:, TX:, :] = 0
            return imgX0, imgY0, d

    def Pixel2data(self, imgX0, method='max'):
        if len(imgX0.shape) == 3:
            imgX0 = imgX0.unsqueeze(0)
        bs, ch, w, h = imgX0.shape
        imgX0 = imgX0.cpu().detach().numpy() if torch.is_tensor(imgX0) else imgX0

        if method == 'max':
            indx = np.argmax(imgX0, axis=-1)
        elif method == 'expection':
            imgX0 = imgX0 / np.sum(imgX0, axis=-1, keepdims=True)
            indNumber = np.arange(0, h)
            imgX0 *= indNumber
            indx = np.sum(imgX0, axis=-1)

        resolution = self.maxScal * 2 / (self.h - 1)
        res = np.transpose(indx, (0, 2, 1)) * resolution - self.maxScal

        return res

    def linear_interpolation(self, arr):
        n, c = arr.shape
        new_arr = np.zeros((2 * n, c))
        new_arr[0::2] = arr
        new_arr[1:-1:2] = (arr[:-1] + arr[1:]) / 2
        new_arr = np.concatenate((new_arr[0:1, :], new_arr[0:-1, :]))
        return new_arr

    def interpolate_sequence(self, sequence, target_length):
        T, c = sequence.shape
        interpolated_sequence = np.zeros((target_length, c))
        for i in range(c):
            f = interp1d(np.arange(T), sequence[:, i], kind='linear')
            interpolated_sequence[:, i] = f(np.linspace(0, T - 1, target_length))
        return interpolated_sequence

    def reverse_interpolate_sequence(self, sequence, original_length):
        bs, T, C = sequence.shape
        source_length = T
        x = np.linspace(0, source_length - 1, T)
        x_new = np.linspace(0, source_length - 1, original_length)
        reverse_interpolated_sequence = np.zeros((bs, original_length, C))

        for i in range(C):
            f = interp1d(x, sequence[:, :, i], kind='linear', axis=1)
            reverse_interpolated_sequence[:, :, i] = f(x_new)
        return reverse_interpolated_sequence


    def dataTransformation(self,dataX):
        '''

        :param data: T,C
        :return:
        '''


        T, c = dataX.shape

        realInputLength=T

        seq_xO= np.copy(dataX)



        std = (np.std(seq_xO, axis=0).reshape(1, -1) + 1e-7)
        seq = (seq_xO ** self.args.muNorm) * np.sign(seq_xO)
        mu0 = np.mean(seq, axis=0) + 1e-7
        mu = np.sqrt(np.abs(mu0)) * np.sign(mu0).reshape(1, -1)
        seq_x = (seq_xO - mu) / std

        if realInputLength < self.seq_len:
            seq0 = np.ones([self.seq_len - seq_x.shape[0], seq_x.shape[1]]) * mu
            seq1 = np.ones([self.pred_len, seq_x.shape[1]]) * mu
            seq_x = np.concatenate((seq0, seq_x,seq1), axis=0)
        else:
            seq1 = np.ones([self.pred_len, seq_x.shape[1]]) * mu
            seq_x = np.concatenate((seq_x, seq1), axis=0)
        if self.args.upscal:
            seq_x = self.linear_interpolation(seq_x)


        x, d = self.data2Pixel(seq_x, None)

        if self.args.ks[0] != 1 or self.args.ks[1] != 1:
            kernel_size = (self.args.ks[0], self.args.ks[1])
            sigmaX = 0
            for i in range(x.shape[0]):
                x[i] = cv2.GaussianBlur(x[i], kernel_size, sigmaX) * kernel_size[0]
        if realInputLength < self.seq_len and self.args.upscal:
            x[:, :self.args.size[0] -realInputLength * 2, :] = 0
        elif realInputLength < self.seq_len and not self.args.upscal:
            x[:, :self.args.size[0] - realInputLength , :] = 0
        x[:, self.args.size[0]:, :] = 0


        return torch.from_numpy(x).float(), torch.from_numpy(d).float(),mu,std

    def dataTransformationBatch(self, dataX):
        bs,T,C=dataX.shape
        for i in range(bs):
            if i==0:
                x,d,mu,std=self.dataTransformation(dataX[i,:])
                x=x.unsqueeze(0)
                d=d.unsqueeze(0)
                mu = np.expand_dims(mu, axis=0)
                std = np.expand_dims(std, axis=0)

            else:
                x0, d0, mu0, std0 =self.dataTransformation(dataX[i,:])
                x=torch.cat([x,x0.unsqueeze(0)],dim=0)
                d=torch.cat([d,d0.unsqueeze(0)],dim=0)
                mu=np.concatenate([mu,np.expand_dims(mu0, axis=0)],axis=0)
                std=np.concatenate([std,np.expand_dims(std0, axis=0)],axis=0)
        return x,d,mu,std


    def __getitem__(self, index):
        if self.flag == 'train' and hasattr(self.args, 'dataPercent'):
            index = self.wholeDataLenOrgional - index - 2




        s_begin, s_end = index, index + int(getattr(self.args, 'realInputLength', self.seq_len))
        r_begin, r_end = s_end - self.label_len, s_end - self.label_len + self.pred_len

        seq_xO, seq_yO = np.copy(self.data_x[s_begin:s_end]), np.copy(self.data_y[s_begin:r_end])

        seq_yO_save = np.copy(seq_yO)


        std = (np.std(seq_xO, axis=0).reshape(1, -1) + 1e-7)
        seq = (seq_xO ** self.args.muNorm) * np.sign(seq_xO)
        mu0 = np.mean(seq, axis=0) + 1e-7
        mu = np.sqrt(np.abs(mu0)) * np.sign(mu0).reshape(1, -1)
        seq_x, seq_y = (seq_xO - mu) / std, (seq_yO - mu) / std

        if  self.args.realInputLength < self.seq_len:
            seq_yO = np.copy(seq_yO_save)
            seq0 = np.ones([self.seq_len - seq_x.shape[0], seq_x.shape[1]]) * mu
            seq_x, seq_y = np.concatenate((seq0, seq_x), axis=0), np.concatenate((seq0, seq_y), axis=0)




        seq_x, seq_y = self.linear_interpolation(seq_x), self.linear_interpolation(seq_y)

        if self.flag == 'train':
            seq_y += np.random.rand(1, seq_y.shape[1]) - 0.5 + np.random.randn(seq_y.shape[0], seq_y.shape[1]) * 0.05 * 6


        x, y, d = self.data2Pixel(seq_x, seq_y)

        if self.args.ks[0] != 1 or self.args.ks[1] != 1:
            kernel_size = (self.args.ks[0], self.args.ks[1])
            sigmaX = 0
            for i in range(x.shape[0]):
                x[i] = cv2.GaussianBlur(x[i], kernel_size, sigmaX) * kernel_size[0]

        if self.args.realInputLength < self.seq_len:
            x[:,:self.args.size[0] - self.args.realInputLength * 2, :] = 0
            y[:,:self.args.size[0] - self.args.realInputLength * 2 , :] = 0

        return self.format_output(x, y, d, seq_xO, seq_yO, mu, std)

    def resize_sequences(self, seq_x, seq_yO, seq_y):
        seq_x = interp1d(np.arange(seq_x.size), seq_x.flatten(), kind='linear')(
            np.linspace(0, seq_x.size - 1, self.args.size[0])).reshape(-1, 1)
        seq_yO = interp1d(np.arange(seq_yO.size), seq_yO.flatten(), kind='linear')(
            np.linspace(0, seq_yO.size - 1, self.args.size[0] + self.args.size[2])).reshape(-1, 1)
        seq_y = interp1d(np.arange(seq_y.size), seq_y.flatten(), kind='linear')(
            np.linspace(0, seq_y.size - 1, self.args.size[0] + self.args.size[2])).reshape(-1, 1)
        return seq_x, seq_yO, seq_y

    def format_output(self, x, y, d, seq_xO, seq_yO, mu, std):
        if 'train' not in self.flag:

            return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(d).float(), \
                   torch.from_numpy(seq_xO).float(), torch.from_numpy(seq_yO).float(), torch.from_numpy(
                mu).float(), torch.from_numpy(std).float()
        else:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(d).float()

    def __len__(self):
        if self.flag == 'train' and hasattr(self.args, 'dataPercent'):
            self.wholeDataLenOrgional = len(self.data_x) - self.seq_len - self.pred_len + 1
            self.UsedDataLenOrgional = int((len(self.data_x) - self.seq_len - self.pred_len + 1) * self.args.dataPercent)
            return self.UsedDataLenOrgional
        return len(self.data_x) - self.seq_len - self.pred_len + 1





# Custom dataset for VR with additional functionality
class Dataset_Custom(Dataset_ViTime):
    def __init__(self, args):
        self.args = args

        size = args.size
        self.flag = args.flag
        self.h = args.h
        self.data_path = args.data_path
        self.features = args.features
        self.maxScal = args.maxScal
        self.target = args.target
        self.scale = True

        self.freq = 'h'

        self.seq_len = int(size[0] / 2)
        self.label_len = int(size[1] / 2)
        self.pred_len = int(size[2] / 2)
        self.Norm = args.dNorm

        assert self.flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[self.flag]




        self.__read_data__()
        self.__prepareD__()


        self.data_x = interX(self.data_x, args.RescaleFactors)
        self.data_y = interX(self.data_y, args.RescaleFactors)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.args.root_path, self.data_path))
        self.scalerStand = StandardScaler()
        self.scaler = StandardScaler()

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            self.featureNumber = len(cols_data)
        elif self.features == 'S':
            cols_data = df_raw.columns[1:]
            self.target = cols_data[int(self.target)] if self.target != 'OT' else self.target
            df_data = df_raw[[self.target]]
            self.featureNumber = 1

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            self.scalerStand.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

def interX(x, scal):
    n, c = x.shape
    new_length = int(scal * n)
    new_indices = np.linspace(start=0, stop=n - 1, num=new_length)
    res = np.zeros([new_length, c])
    for i in range(c):
        res[:, i] = np.interp(new_indices, np.arange(n), x[:, i])
    return res

# Custom dataset for ETTminVR
class Dataset_ETTminVR(Dataset_ViTime):
    def __init__(self, args):
        self.args = args
        size = args.size
        self.flag = args.flag
        self.h = args.h
        self.data_path = args.data_path

        self.maxScal = args.maxScal
        self.target = args.target
        self.scale = True

        self.freq = 't'
        self.seq_len = int(size[0] / 2)
        self.label_len = int(size[1] / 2)
        self.pred_len = int(size[2] / 2)
        self.Norm = args.dNorm

        assert self.flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[self.flag]


        self.__read_data__()
        self.__prepareD__()

        self.data_x = interX(self.data_x, args.RescaleFactors)
        self.data_y = interX(self.data_y, args.RescaleFactors)
    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.args.root_path, self.data_path))
        self.scalerStand = StandardScaler()
        self.scaler = StandardScaler()

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            self.featureNumber = len(cols_data)
        elif self.features == 'S':
            cols_data = df_raw.columns[1:]
            self.target = cols_data[int(self.target)] if self.target != 'OT' else self.target
            df_data = df_raw[[self.target]]
            self.featureNumber = 1

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            self.scalerStand.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

# Custom dataset for ETThourVR
class Dataset_ETThourVR(Dataset_ViTime):
    def __init__(self, args):
        self.args = args
        size = args.size
        self.flag = args.flag
        self.h = args.h
        self.data_path = args.data_path

        self.maxScal = args.maxScal
        self.target = args.target
        self.scale = True

        self.freq = 'h'
        self.seq_len = int(size[0] / 2)
        self.label_len = int(size[1] / 2)
        self.pred_len = int(size[2] / 2)
        self.Norm = args.dNorm
        assert self.flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[self.flag]
        self.__read_data__()
        self.__prepareD__()

        self.data_x = interX(self.data_x, args.RescaleFactors)
        self.data_y = interX(self.data_y, args.RescaleFactors)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.args.root_path, self.data_path))
        self.scalerStand = StandardScaler()
        self.scaler = StandardScaler()

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            self.featureNumber = len(cols_data)
        elif self.features == 'S':
            cols_data = df_raw.columns[1:]
            self.target = cols_data[int(self.target)] if self.target != 'OT' else self.target
            df_data = df_raw[[self.target]]
            self.featureNumber = 1

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            self.scalerStand.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]











# Custom dataset for RealTS
class RealTS(Dataset_ViTime):
    def __init__(self, args):
        self.args = args
        self.load_distribution_data()
        self.funcL = self.get_func_list(args)
        size = args.size
        self.flag = args.flag
        self.h = args.h
        self.maxScal = args.maxScal
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.__prepareD__()


    def load_distribution_data(self):
        with open('IFFTB_Distribution1', 'rb') as f:
            self.mean_magnitudes1, self.std_magnitudes1, self.mean_phases1, self.std_phases1 = pickle.load(f)
        with open('IFFTB_Distribution2', 'rb') as f:
            self.mean_magnitudes2, self.std_magnitudes2, self.mean_phases2, self.std_phases2 = pickle.load(f)

    def get_func_list(self, args):
        return [
            RWB,
            seasonal_periodicity,
            PWB,
            TWDB,
            LGB,
            partial(IFFTB, self.mean_magnitudes1, self.std_magnitudes1, self.mean_phases1, self.std_phases1, 'Type1', args),
            partial(IFFTB, self.mean_magnitudes2, self.std_magnitudes2, self.mean_phases2, self.std_phases2, 'Type2', args),
        ]

    def __getitem__(self, index):
        indL = list(np.arange(len(self.funcL)))
        pcyclic_pattern = 0.3
        pM = (1 - pcyclic_pattern * 2) / 5
        p = [pM, pM, pM, pM, pM, pcyclic_pattern, pcyclic_pattern]

        func = self.funcL[np.random.choice(indL, p=p)]
        seq_x = func(self.seq_len + self.pred_len).reshape(-1, 1)

        if np.random.rand() < 0.5:
            seq_x = seq_x[::-1, :]

        std = (np.std(seq_x, axis=0).reshape(1, -1) + np.random.rand() * 1e-7)
        seq = (seq_x ** self.args.muNorm) * np.sign(seq_x)
        mu0 = np.mean(seq, axis=0) + 1e-7
        mu = np.sqrt(np.abs(mu0)) * np.sign(mu0).reshape(1, -1)

        if getattr(self.args, 'MoreRandom', False):
            if np.random.rand() < 0.2:
                scalSTD = np.random.rand() * 3 + 0.5
                std *= scalSTD
                if np.random.rand() < 0.2:
                    mu = mu + std
                elif np.random.rand() < 0.2:
                    mu = mu - std
            else:
                if np.random.rand() < 0.1:
                    mu = mu + std
                elif np.random.rand() < 0.1:
                    mu = mu - std

        seq_x = (seq_x - mu) / std
        if np.random.rand() < 0.5:
            seq_x = -seq_x


        x, y, d = self.data2Pixel(seq_x[:self.seq_len], seq_x)

        if self.args.ks[0] != 1 or self.args.ks[1] != 1:
            kernel_size = (self.args.ks[0], self.args.ks[1])
            sigmaX = 0
            x[0] = cv2.GaussianBlur(x[0], kernel_size, sigmaX) * kernel_size[0]

        return self.format_output(x, y, d, seq_x)

    def format_output(self, x, y, d, seq_x):
        if self.flag != 'train':
            return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(d).float(), \
                   torch.from_numpy(seq_x.copy()).float()
        else:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(d).float()

    def __len__(self):
        return 20000
