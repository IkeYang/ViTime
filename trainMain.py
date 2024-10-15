import os
import pickle
import sys
import torch
import time
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch.optim as optim
from loadData import Dataset_synthesis_Cycle_FFT
from model.model import modelDict
from loadData import Dataset_CustomVR,Dataset_ETTminVR,Dataset_ETThourVR,Dataset_synthesis_Cycle_FFT
from test import test, testGPT, testCycle, testCycle2
import argparse
from utlize import mkdir, EMD, GPUMemoryCheck, EMD_Cycle, EMD_Cycle_onlyout, EMD_Cycle_JSD, EMD_Cycle_JSD_Extand
import torch.nn as nn
import datetime
import os



modelSizeDict = {
    'exchange_rate.csv': 4,
    'ETTm2.csv': 4,
    'ETTm1.csv': 4,
    'ETTh1.csv': 4,
    'ETTh2.csv': 4,
    'electricity.csv': 30,
    'traffic.csv': 75,
    'weather.csv': 8,
    'national_illness.csv': 4,
}





def trainCycle3(h, lookBackWindow, T, bs, data_path, MS, modelName, epochs, Norm_Insequence=True, modelAda=False,
                model_path=r'E:\YANG Luoxiao\Model\VR\Synthsis'
                , dropout=0.1, TAP=3, TA=0.9, features='S', opt='Adam', num_workers=0, lr=2e-4, out_channels=8,
                root_path=r'E:\YANG Luoxiao\Code\DLinear\data',
                dNorm=1, weight_decay=0, modelSize=4, name='VR', downsample_factor=16, modelExtend=False,
                oneChannel=False, deviceNum=0, cycleTime=5,
                changeEpoch=5, ks=31, focus=2, memoryTestMode=False, FFT=False, temparture=1, aspp=True, lowlevel=True,
                loadHistory=False,scal=1,deepcycleTime=1,randomCut=False,
                historyName=False, lossType='EMD', patch_size=(4, 32), gpu_ids=None, curve=False, MoreRandom=False, muNorm=1,expandData=1,setSynDataType='all'
                ):
    from torch.optim import lr_scheduler
    print(datetime.datetime.now())
    print('PID: ', os.getpid(), '***********')
    size = [lookBackWindow, 0, T]
    if gpu_ids is None:
        torch.cuda.set_device(deviceNum)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5,6'  ##设置主卡 为卡1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='VI Forecasting')

    parser.add_argument('--data_path', type=str, required=False, default=data_path, help='data_path')
    parser.add_argument('--root_path', type=str, required=False, default=root_path, help='root_path')
    parser.add_argument('--h', type=int, required=False, default=h, help='h')
    parser.add_argument('--size', required=False, default=size, help='size')
    parser.add_argument('--flag', required=False, default='train', help='flag')
    parser.add_argument('--features', required=False, default=features, help='features')
    parser.add_argument('--target', required=False, default='OT', help='target')
    parser.add_argument('--loss', required=False, default=lossType, help='loss')
    parser.add_argument('--maxScal', required=False, default=MS, help='maxScal')
    parser.add_argument('--weight_decay', required=False, default=weight_decay, help='weight_decay')
    parser.add_argument('--modelName', required=False, default=modelName, help='modelName')
    parser.add_argument('--name', required=False, default=name, help='name')
    parser.add_argument('--dNorm', required=False, default=dNorm, help='dNorm')
    parser.add_argument('--bs', required=False, default=bs, help='bs')
    parser.add_argument('--maxTest', required=False, default=500, help='maxTest')
    parser.add_argument('--num_workers', required=False, default=num_workers, help='num_workers')
    parser.add_argument('--reducedChannelNumber', required=False, default=None, help='reducedChannelNumber')
    parser.add_argument('--TA', required=False, default=TA, help='TA')
    parser.add_argument('--TAP', required=False, default=TAP, help='TAP')
    parser.add_argument('--dropout', required=False, default=dropout, help='dropout')
    parser.add_argument('--Norm_Insequence', required=False, default=Norm_Insequence, help='Norm_Insequence')
    parser.add_argument('--modelSize', required=False, default=modelSize, help='modelSize')
    parser.add_argument('--modelAda', required=False, default=modelAda, help='modelAda')
    parser.add_argument('--downsample_factor', required=False, default=downsample_factor, help='downsample_factor')
    parser.add_argument('--modelExtend', required=False, default=modelExtend, help='modelExtend')
    parser.add_argument('--oneChannel', required=False, default=oneChannel, help='oneChannel')
    parser.add_argument('--out_channels', required=False, default=out_channels, help='out_channels')
    parser.add_argument('--model_path', required=False, default=model_path, help='model_path')
    parser.add_argument('--embed_dim', required=False, default=modelSize, help='embed_dim')
    parser.add_argument('--cycleTime', required=False, default=cycleTime, help='cycleTime')
    parser.add_argument('--deepcycleTime', required=False, default=deepcycleTime, help='deepcycleTime')
    parser.add_argument('--changeEpoch', required=False, default=changeEpoch, help='changeEpoch')
    parser.add_argument('--focus', required=False, default=focus, help='focus')
    parser.add_argument('--aspp', required=False, default=aspp, help='aspp')
    parser.add_argument('--lowlevel', required=False, default=lowlevel, help='lowlevel')
    parser.add_argument('--patch_size', required=False, default=patch_size, help='patch_size')
    parser.add_argument('--ks', required=False, default=ks, help='ks')
    parser.add_argument('--earlystop', required=False, default=50, help='ks')
    parser.add_argument('--curve', required=False, default=curve, help='ks')
    parser.add_argument('--lossType', required=False, default=lossType, help='ks')
    parser.add_argument('--MoreRandom', required=False, default=MoreRandom, help='MoreRandom')
    parser.add_argument('--muNorm', required=False, default=muNorm, help='muNorm')
    parser.add_argument('--expandData', required=False, default=expandData, help='expandData')
    parser.add_argument('--setSynDataType', required=False, default=setSynDataType, help='setSynDataType')
    parser.add_argument('--scal', required=False, default=scal, help='scal')
    parser.add_argument('--randomCut', required=False, default=randomCut, help='randomCut')
    args = parser.parse_args()

    if modelName == 'SwinUnet':
        from transformer_config_my import get_defaultconfig
        config = get_defaultconfig()
        config.DATA.IMG_SIZE = (size[0] + size[-1], h)
        config.DATA.BATCH_SIZE = bs
        config.MODEL.SWIN.EMBED_DIM = int(96 * modelSize)
        args.config = config

    # if loadHistory:
    #     try:
    #         with open('argsSetting/%s' % (historyName), 'rb') as f:
    #             argsHis = pickle.load(f)
    #     except:
    #         try:
    #             with open('argsSetting/%s' % (historyName[3:]), 'rb') as f:
    #                 argsHis = pickle.load(f)
    #         except:
    #             with open('argsSetting/%s' % (historyName[3:30]), 'rb') as f:
    #                 argsHis = pickle.load(f)
    #     if  args.modelName ==argsHis.modelName:
    #         args.name = argsHis.name
    #
    #     # args.name ='Cycle3'+lossType+modelName+'2023-06-08-17-51-01'
    #     # with open('argsSetting/%s'%(args.name),'wb') as f:
    #     #     pickle.dump(args,f)
    #
    # else:

    with open('argsSetting/%s' % (name), 'wb') as f:
        pickle.dump(args, f)

    print(args)
    if FFT:
        trainDatas = Dataset_synthesis_FFT(args)
    else:
        # trainDatas = Dataset_synthesis_Cycle(args)
        trainDatas = Dataset_synthesis_Cycle_FFT(args)

    def cycleForword(model, x, epoch, xO,mask):

        # cycleNumber = int(epoch / changeEpoch) + 1
        # 
        # if cycleNumber > cycleTime:
        #     cycleNumber = np.random.randint(1, cycleTime + 1)
        # if memoryTestMode:
        #     cycleNumber = cycleTime

        cycleNumber = np.random.randint(1, cycleTime + 1)
        with torch.no_grad():
            for i in range(cycleNumber - 1):
                x = model(x, temparture=temparture)
                x = x * mask + xO
        x = model(x, temparture=temparture)
        return x, cycleNumber

    args.reducedChannelNumber = 1
    batch_size = bs
    lr = lr
    modeldict = modelDict()
    model = modeldict[modelName](inchannel=1, T=trainDatas.seq_len + trainDatas.pred_len, args=args, DCNumber=None,
                                 out_channels=out_channels, loss=args.loss)

    if gpu_ids is not None:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    else:
        model.to(device)
    if loadHistory:
        checkpoint = torch.load('%s/%s_VR3_data_syn_size_336.pth' % (model_path, historyName), map_location=device)
        oldargs=checkpoint['args']
        # checkpoint = torch.load('%s/%s_VR3_data_syn_size_336.pth' % (model_path, historyName))
        try:
            model.load_state_dict(checkpoint['model'])  # load trained model parameters
        except:

            remove_prefix = 'module.'
            checkpoint['model'] = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in
                                   checkpoint['model'].items()}
            try:
                model.load_state_dict(checkpoint['model'])  # load trained model parameters
            except:
                pass

        model.to(device)
    if setSynDataType=='cycleOnly':
        dataSet2='electricity.csv'

    elif setSynDataType=='all':
        dataSet2 = 'ETTh1.csv'
    if opt == 'Adam':
        if 'Full' in modelName:
            model_params = [
                {"params": model.model.parameters(), "lr": lr / 10},
                {"params": model.modelDeeplab.parameters(), "lr": lr}
            ]

            # 创建优化器
            optimizer = optim.Adam(model_params, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(list(model.parameters()), lr=lr, weight_decay=weight_decay)
    if opt == 'SGD':
        optimizer = optim.SGD(list(model.parameters()), lr=lr, weight_decay=weight_decay)
    dataloaderT = torch.utils.data.DataLoader(trainDatas, batch_size=batch_size, pin_memory=False,
                                              shuffle=True, num_workers=int(num_workers))
    schedulaer = lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    onlyOut = True
    bestLoss = 100000
    lossF = EMD_Cycle(args=args)
    if lossType == 'CE':
        loss2 = CELoss()
    elif lossType == 'JSD':
        lossF = EMD_Cycle_JSD(args=args)
    for epoch in range(epochs):
   
        args.flag = 'train'  ####cai fa xian a !!!!!! shaole pingyi fangsuo deng zeng qiang
        model.train()
        # if epoch<10:
        #     cycleTime1=1
        # else:
        #     cycleTime1+=int(epoch/10)
        # if cycleTime1>cycleTime:
        cycleTime1 = cycleTime
        if temparture > 1:
            temparture *= 0.99
        for i, (x, y, d, id) in enumerate(dataloaderT):
            # break

            if x.shape[0] == 1:
                continue
            x = x.to(device)  # bs,c,w,h
            xO = copy.deepcopy(x)
            y = y.to(device)
            d = d.to(device)
            mask = torch.ones_like(xO)
            mask[:, :, :args.size[0], :] = 0
            mask = mask.to(device)
            if oneChannel:
                id[:] = 0
            else:
                id[:int(batch_size / 4)] = 0
            yobs = y.detach().cpu().numpy()
            optimizer.zero_grad()

            ypred, cycleNumber = cycleForword(model, x, epoch, xO,mask)
            discount = 50 ** (2 - args.dNorm)
            if lossType == 'MSE':
                loss = torch.mean((ypred[:,:,size[0]:,:] - y[:,:,size[0]:,:]) ** 2) * 200
            else:
                loss = lossF(ypred, d, cycleNumber) * discount
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)

            optimizer.step()

            if (i) % int(len(dataloaderT) / 4) == 0:
                print('[%d/%d][%d/%d]\tLoss: %.8f\t  '
                      % (epoch, epochs, i, len(dataloaderT), loss,))

        schedulaer.step()
        args.flag = 'test'
        print('Testing......')
        args.target = 'OT'
        args.data_path = data_path
        lossNow1 = testCycle2(args, model=model, epoch=epoch)[-2]

        args.data_path = dataSet2
        lossNow2 = testCycle2(args, model=model, epoch=epoch)[-2]

        if args.setSynDataType == 'all':
            args.target = 'OT'
            args.data_path = 'electricity.csv'
        else:
            args.target = '0'
            args.data_path = data_path
        lossNow3 = testCycle2(args, model=model, epoch=epoch)[-2]



        if args.setSynDataType == 'all':
            args.target = '0'
            args.data_path = data_path
            lossNow4 = testCycle2(args, model=model, epoch=epoch)[-2]
        if hasattr(args, 'setSynDataType'):
            if args.setSynDataType=='cycleOnly':
                lossNow = lossNow1 / 0.187 + lossNow2 / 0.422 + lossNow3/0.8
            else:
                lossNow = lossNow1 / 0.187 + lossNow2 / 0.087 + lossNow3/0.422+ lossNow4/0.8
        else:
            lossNow = lossNow1 / 0.187 + lossNow2 / 0.087 + lossNow3

        # lossNow = lossNow3
        if bestLoss > lossNow:
            if args.setSynDataType == 'all':
                print('BESTLOSS', lossNow, data_path + 'OT', lossNow1, dataSet2, lossNow2, 'electricity' + 'OT', lossNow3, data_path + '0', lossNow4)
            else:
                print('BESTLOSS', lossNow, data_path+'OT', lossNow1, dataSet2, lossNow2, data_path+'0', lossNow3)
            bestLoss = lossNow
            state = {'model': model.state_dict(), 'epoch': epoch, 'args': args}
            torch.save(state, '%s/%s_VR3_data_syn_size_336.pth' % (model_path, args.name))
        state = {'model': model.state_dict(), 'epoch': epoch, 'args': args}
        torch.save(state, '%s/New%s_VR3_data_syn_size_336.pth' % (model_path, args.name))



def fine_Tunetrain(h, lookBackWindow, T, bs, data_path, MS, modelName, epochs, Norm_Insequence=True, modelAda=False,
                model_path=r'E:\YANG Luoxiao\Model\VR\Synthsis',dataPercent=1
                , dropout=0.1, TAP=3, TA=0.9, features='S', opt='Adam', num_workers=0, lr=2e-4, out_channels=8,
                root_path=r'E:\YANG Luoxiao\Code\DLinear\data',
                dNorm=1, weight_decay=0, modelSize=4, name='VR', downsample_factor=16, modelExtend=False,
                oneChannel=False, deviceNum=0, cycleTime=5,
                changeEpoch=5, ks=31, focus=2, memoryTestMode=False, FFT=False, temparture=1, aspp=True, lowlevel=True,
                loadHistory=False,scal=1,deepcycleTime=1,
                historyName=False, lossType='EMD', patch_size=(4, 32), gpu_ids=None, curve=False, MoreRandom=False, muNorm=1,expandData=1,setSynDataType='all'
                ):
    from torch.optim import lr_scheduler
    print(datetime.datetime.now())
    print('PID: ', os.getpid(), '***********')
    size = [lookBackWindow, 0, T]
    if gpu_ids is None:
        torch.cuda.set_device(deviceNum)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5,6'  ##设置主卡 为卡1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='VI Forecasting')

    parser.add_argument('--data_path', type=str, required=False, default=data_path, help='data_path')
    parser.add_argument('--root_path', type=str, required=False, default=root_path, help='root_path')
    parser.add_argument('--h', type=int, required=False, default=h, help='h')
    parser.add_argument('--size', required=False, default=size, help='size')
    parser.add_argument('--flag', required=False, default='train', help='flag')
    parser.add_argument('--features', required=False, default=features, help='features')
    parser.add_argument('--target', required=False, default='OT', help='target')
    parser.add_argument('--loss', required=False, default=lossType, help='loss')
    parser.add_argument('--maxScal', required=False, default=MS, help='maxScal')
    parser.add_argument('--weight_decay', required=False, default=weight_decay, help='weight_decay')
    parser.add_argument('--modelName', required=False, default=modelName, help='modelName')
    parser.add_argument('--name', required=False, default=name, help='name')
    parser.add_argument('--dNorm', required=False, default=dNorm, help='dNorm')
    parser.add_argument('--bs', required=False, default=bs, help='bs')
    parser.add_argument('--maxTest', required=False, default=500, help='maxTest')
    parser.add_argument('--num_workers', required=False, default=num_workers, help='num_workers')
    parser.add_argument('--reducedChannelNumber', required=False, default=None, help='reducedChannelNumber')
    parser.add_argument('--TA', required=False, default=TA, help='TA')
    parser.add_argument('--TAP', required=False, default=TAP, help='TAP')
    parser.add_argument('--dropout', required=False, default=dropout, help='dropout')
    parser.add_argument('--Norm_Insequence', required=False, default=Norm_Insequence, help='Norm_Insequence')
    parser.add_argument('--modelSize', required=False, default=modelSize, help='modelSize')
    parser.add_argument('--modelAda', required=False, default=modelAda, help='modelAda')
    parser.add_argument('--downsample_factor', required=False, default=downsample_factor, help='downsample_factor')
    parser.add_argument('--modelExtend', required=False, default=modelExtend, help='modelExtend')
    parser.add_argument('--oneChannel', required=False, default=oneChannel, help='oneChannel')
    parser.add_argument('--out_channels', required=False, default=out_channels, help='out_channels')
    parser.add_argument('--model_path', required=False, default=model_path, help='model_path')
    parser.add_argument('--embed_dim', required=False, default=modelSize, help='embed_dim')
    parser.add_argument('--cycleTime', required=False, default=cycleTime, help='cycleTime')
    parser.add_argument('--deepcycleTime', required=False, default=deepcycleTime, help='deepcycleTime')
    parser.add_argument('--changeEpoch', required=False, default=changeEpoch, help='changeEpoch')
    parser.add_argument('--focus', required=False, default=focus, help='focus')
    parser.add_argument('--aspp', required=False, default=aspp, help='aspp')
    parser.add_argument('--lowlevel', required=False, default=lowlevel, help='lowlevel')
    parser.add_argument('--patch_size', required=False, default=patch_size, help='patch_size')
    parser.add_argument('--ks', required=False, default=ks, help='ks')
    parser.add_argument('--earlystop', required=False, default=50, help='ks')
    parser.add_argument('--curve', required=False, default=curve, help='ks')
    parser.add_argument('--lossType', required=False, default=lossType, help='ks')
    parser.add_argument('--MoreRandom', required=False, default=MoreRandom, help='MoreRandom')
    parser.add_argument('--muNorm', required=False, default=muNorm, help='muNorm')
    parser.add_argument('--expandData', required=False, default=expandData, help='expandData')
    parser.add_argument('--setSynDataType', required=False, default=setSynDataType, help='setSynDataType')
    parser.add_argument('--scal', required=False, default=scal, help='scal')
    parser.add_argument('--dataPercent', required=False, default=dataPercent, help='dataPercent')
    args = parser.parse_args()

   

    bsOr=copy.deepcopy(bs)

    with open('argsSetting/%s' % (name), 'wb') as f:
        pickle.dump(args, f)

    print(args)
    if 'ETTh' in args.data_path:
        trainDatas = Dataset_ETThourVR(args)

    elif 'ETTm' in args.data_path:
        trainDatas = Dataset_ETTminVR(args)

    else:
        trainDatas = Dataset_CustomVR(args)

    def cycleForword(model, x, epoch, xO,mask):

        # cycleNumber = int(epoch / changeEpoch) + 1
        # 
        # if cycleNumber > cycleTime:
        #     cycleNumber = np.random.randint(1, cycleTime + 1)
        # if memoryTestMode:
        #     cycleNumber = cycleTime

        cycleNumber = np.random.randint(1, cycleTime + 1)
        with torch.no_grad():
            for i in range(cycleNumber - 1):
                x = model(x, temparture=temparture)
                x = x * mask + xO
        x = model(x, temparture=temparture)
        return x, cycleNumber

    args.reducedChannelNumber = 1
    batch_size = bs
    lr = lr
    modeldict = modelDict()
    model = modeldict[modelName](inchannel=1, T=trainDatas.seq_len + trainDatas.pred_len, args=args, DCNumber=None,
                                 out_channels=out_channels, loss=args.loss)

    if gpu_ids is not None:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    else:
        model.to(device)
    if loadHistory:
        checkpoint = torch.load('%s/%s_VR3_data_syn_size_336.pth' % (model_path, historyName), map_location=device)
        oldargs=checkpoint['args']
        # checkpoint = torch.load('%s/%s_VR3_data_syn_size_336.pth' % (model_path, historyName))
        try:
            model.load_state_dict(checkpoint['model'])  # load trained model parameters
        except:

            remove_prefix = 'module.'
            checkpoint['model'] = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in
                                   checkpoint['model'].items()}
            try:
                model.load_state_dict(checkpoint['model'])  # load trained model parameters
            except:
                pass

        model.to(device)
    if setSynDataType=='cycleOnly':
        dataSet2='electricity.csv'

    elif setSynDataType=='all':
        dataSet2 = 'ETTh1.csv'
    if opt == 'Adam':
        if 'Full' in modelName:
            model_params = [
                {"params": model.model.parameters(), "lr": lr / 10},
                {"params": model.modelDeeplab.parameters(), "lr": lr}
            ]

            # 创建优化器
            optimizer = optim.Adam(model_params, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(list(model.parameters()), lr=lr, weight_decay=weight_decay)
    if opt == 'SGD':
        optimizer = optim.SGD(list(model.parameters()), lr=lr, weight_decay=weight_decay)
    dataloaderT = torch.utils.data.DataLoader(trainDatas, batch_size=batch_size, pin_memory=False,
                                              shuffle=True, num_workers=int(num_workers))
    schedulaer = lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    onlyOut = True
    bestLoss = 100000
    lossF = EMD_Cycle(args=args)
    if lossType == 'CE':
        loss2 = CELoss()
    elif lossType == 'JSD':
        lossF = EMD_Cycle_JSD(args=args)
    for epoch in range(epochs):
   
        args.flag = 'train'  ####cai fa xian a !!!!!! shaole pingyi fangsuo deng zeng qiang
        model.train()
        # if epoch<10:
        #     cycleTime1=1
        # else:
        #     cycleTime1+=int(epoch/10)
        # if cycleTime1>cycleTime:
        cycleTime1 = cycleTime
        if temparture > 1:
            temparture *= 0.99
        for i, (x, y, d, id) in enumerate(dataloaderT):
            # break

            if x.shape[0] == 1:
                continue
            bs, c, w, h = x.shape

            x = x.to(device).view(bs * c, 1, w, h)  # bs,c,w,h
            y = y.to(device).view(bs * c, 1, w, h)
            d = d.to(device).view(bs * c, 1, w, h)
            

            if  c!=1:
                indices = torch.randperm(bs * c)[:bs]
                x = x[indices]
                y = y[indices]
                d = d[indices]
                c=1
            
            
            xO = copy.deepcopy(x)
            mask = torch.ones_like(xO)
            mask[:, :, :args.size[0], :] = 0
            mask = mask.to(device)
            if oneChannel:
                id[:] = 0
            else:
                id[:int(batch_size / 4)] = 0
            yobs = y.detach().cpu().numpy()
            optimizer.zero_grad()

            ypred, cycleNumber = cycleForword(model, x, epoch, xO,mask)
            discount = 50 ** (2 - args.dNorm)
            if lossType == 'MSE':
                loss = torch.mean((ypred[:,:,size[0]:,:] - y[:,:,size[0]:,:]) ** 2) * 200
            else:
                loss = lossF(ypred, d, cycleNumber) * discount
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)

            optimizer.step()

            if (i) % int(len(dataloaderT) / 4) == 0:
                print('[%d/%d][%d/%d]\tLoss: %.8f\t  '
                      % (epoch, epochs, i, len(dataloaderT), loss,))

        schedulaer.step()
        args.flag = 'test'
        print('Testing......')
        args.target = 'OT'
        args.bs=2*bsOr
        lossNow = testCycle2(args, model=model, epoch=epoch)[-2]


        # lossNow = lossNow3
        if bestLoss > lossNow:
           
            print('BESTLOSS', lossNow)
            bestLoss = lossNow
            state = {'model': model.state_dict(), 'epoch': epoch, 'args': args}
            torch.save(state, '%s/%s_VR3_data_syn_size_fineTune_%s.pth' % (model_path, args.name,args.data_path))
        state = {'model': model.state_dict(), 'epoch': epoch, 'args': args}
        torch.save(state, '%s/New%s_VR3_data_syn_size_fineTune_%s.pth' % (model_path, args.name,args.data_path))


