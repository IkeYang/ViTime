
from trainMain import trainCycle3
from scipy.stats import norm
from scipy import integrate
from scipy.optimize import root_scalar
import numpy as np

import time
tsp = time.time()
print(tsp)

tsp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(tsp))


lookBackWindow=1792-720
lookBackWindow=512*2#1072
T=720*2
bs= 8  #full 5
data_path='traffic.csv'
h=128

MS =3.5

modelName='ViTime'

root_path=r'/home/ike/YLX/GVR'
model_path=r'model'


#model_path = r'E:\YANG Luoxiao\Model\VR\Synthsis'
#root_path=r'E:\YANG Luoxiao\Code\DLinear\data'
epochs=2000
dNorm=1
memoryTestMode=True
temparture=1

cycleTime=1
deepcycleTime=1
loadHistory=False
trainCycle3(h,lookBackWindow,T,bs,data_path,MS,modelName,epochs,patch_size=(4, 32),ks=(31,31),root_path=root_path,model_path=model_path,memoryTestMode=memoryTestMode,deviceNum=0,randomCut=True,
      dNorm=dNorm,temparture=temparture,modelSize=0.75,lossType='JSD',name='Cycle'+modelName+str(tsp),downsample_factor=8,num_workers=8,curve=False,setSynDataType='all',scal=2,deepcycleTime=deepcycleTime,
            modelExtend=True,cycleTime=cycleTime,loadHistory=loadHistory,gpu_ids=None,MoreRandom=True,muNorm=2)










