import argparse
import matplotlib.pyplot as plt
from model.model import ViTime
import numpy as np
import torch
from scipy import interpolate
def interpolate_to_512(original_sequence):
    n = len(original_sequence)
    x_original = np.linspace(0, 1, n)
    x_interpolated = np.linspace(0, 1, 512)
    f = interpolate.interp1d(x_original, original_sequence)
    interpolated_sequence = f(x_interpolated)
    return interpolated_sequence
def inverse_interpolate(processed_sequence, original_length):
    processed_length = len(processed_sequence)
    z = int(original_length * 720 / 512)
    x_processed = np.linspace(0, 1, processed_length)
    x_inverse = np.linspace(0, 1, z)
    f_inverse = interpolate.interp1d(x_processed, processed_sequence)
    inverse_interpolated_sequence = f_inverse(x_inverse)
    return inverse_interpolated_sequence
def main(modelpath, savepath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(modelpath, map_location=device)
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
    interpolated_sequence = interpolate_to_512(xData)
    args.realInputLength = len(interpolated_sequence)
    yp = model.inference(interpolated_sequence).flatten()
    yp=inverse_interpolate(yp, len(xData)).flatten()

    # Plot results
    plt.plot(np.concatenate([xData, yp.flatten()], axis=0), label='Prediction')
    plt.plot(xData, label='Input Sequence')
    plt.legend()
    plt.savefig(savepath)  # 保存图形到指定路径
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViTime model inference')
    parser.add_argument('--modelpath', type=str, required=True, help='Path to the model checkpoint file')
    parser.add_argument('--savepath', type=str, default='plot.png', help='Path to save the plot image (default: plot.png)')
    args = parser.parse_args()
    main(args.modelpath, args.savepath)
