import matplotlib.pyplot as plt
from model.model import ViTime
import numpy as np
import torch

deviceNum = 0
torch.cuda.set_device(deviceNum)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(r'ViTime_V2.pth', map_location=device)
args = checkpoint['args']
args.device = device
args.flag = 'test'

# Set upscaling parameters
args.upscal = True  # True: max input length = 512, max prediction length = 720
                    # False: max input length = 1024, max prediction length = 1440
model = ViTime(args=args)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Example data
xData=np.sin(np.arange(512)/10)+np.sin(np.arange(512)/5+50)+np.cos(np.arange(512)+50)
args.realInputLength = len(xData)
yp = model.inference(xData)

# Plot results
plt.plot(np.concatenate([xData,yp.flatten()],axis=0),label='Prediction')
plt.plot(xData,label='Input Sequence')
plt.legend()
plt.show()
