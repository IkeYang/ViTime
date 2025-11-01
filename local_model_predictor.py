# local_model_predictor.py
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
import cv2
from collections import OrderedDict
from model.model import modelDict


class InferenceInterface:
    def __init__(self, model_path, model_name='MAE', device='cuda:0'):
        """
        Initializes the inference interface for time series prediction.

        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model checkpoints and their corresponding arguments
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.args = checkpoint['args']
       
        self.args.RefiningModel = False
        self.args.device = device
        self.args.flag = 'test'
        ##### args.upscal=True   max input length =512    max prediction length =720
        ##### args.upscal=False  max input length =512*2  max prediction length =720*2
        self.args.upscal = False
        self.args.RefiningModel = False
 

        # Set necessary parameters from the loaded arguments
        self.h = self.args.h  # Height of the pixel representation
        self.maxScal = self.args.maxScal  # Maximum scale value for normalization
        self.seq_len = self.args.size[0]  # Length of the input sequence
        self.label_len = self.args.size[1] if self.args.size[1] > 0 else int(self.seq_len * 0.5)
        self.pred_len = self.args.size[2]  # Length of the prediction sequence

        # Initialize the D matrix for distance calculation in pixel space.
        # D[i, j] stores the distance between pixel row i and pixel row j.
        # This is pre-calculated to speed up the _data2pixel conversion.
        self.D = np.zeros([self.h, self.h])
        for i in range(self.h):
            self.D[i, :i] = np.arange(1, i + 1)[::-1]
            self.D[i, i:] = np.arange(0, self.h - i)
        self.D = self.D ** self.args.dNorm # Apply a norm to the distances

        # Initialize the models using a model factory
        modeldict = modelDict()  # Assumes modelDict is a factory function for creating models.
        
        # Envelope-constrained model
        self.model = modeldict[model_name](
            inchannel=1,
            T=self.seq_len + self.pred_len,
            args=self.args,
            DCNumber=None,
            out_channels=self.args.out_channels,
            loss=self.args.loss
        )
      

        # Load trained weights into the models
        self._load_model_weights(self.model, checkpoint['model'])
       
        
        # Move models to the specified device and set to evaluation mode
        self.model.to(self.device)

        self.model.eval()


        # Set default inference parameters
        self.temparture = 1 # Temperature for softmax, not currently used in the provided forward pass
        self.cycleTime = self.args.cycleTime if hasattr(self.args, 'cycleTime') else 1

    def _load_model_weights(self, model, state_dict):
        """
        Loads model weights, handling cases where the model was saved with nn.DataParallel.
        """
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # This handles cases where the state_dict keys have a 'module.' prefix
            # (from nn.DataParallel) but the model doesn't.
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:]  # remove 'module.' prefix
                else:
                    name = k # if keys already match
                new_state_dict[name] = v
            
            try:
                model.load_state_dict(new_state_dict)
            except Exception as e:
                # As a fallback, try adding the prefix if the first attempt failed
                print(f"Standard and prefix-removed loading failed. Trying to add prefix. Error: {e}")
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = 'module.' + k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict, strict=False)

    
    def _interpolate_sequence(self, sequence, target_length):
        """
        Interpolates a time series sequence to a target length using linear interpolation.

        This function can handle 2D (T, C) and 3D (T, C, S) arrays, where T is the time
        axis, C is the channels axis, and S is the samples axis.

        Args:
            sequence (np.ndarray): The input sequence.
                                Shape (T, C) for standard time series, or
                                Shape (T, C, S) for sampled time series.
            target_length (int): The desired length of the output sequence (new T).

        Returns:
            np.ndarray: The interpolated sequence with shape (target_length, C) or
                        (target_length, C, S).
        """
        original_length = sequence.shape[0]

        # Return the original sequence if no interpolation is needed.
        if original_length == target_length:
            return sequence

        # Store original shape details and reshape for interpolation if necessary.
        # The interpolation function works efficiently on 2D arrays of shape (T, Features).
        original_ndim = sequence.ndim
        sequence_to_interp = sequence
        if original_ndim == 3:
            # Reshape (T, C, S) -> (T, C * S) to treat channels and samples as features.
            t, c, s = sequence.shape
            sequence_to_interp = sequence.reshape(t, c * s)

        # Create original and new time axes for interpolation.
        x_original = np.arange(original_length)
        x_new = np.linspace(0, original_length - 1, target_length)

        # Create the interpolation function.
        # axis=0 ensures that we interpolate along the time dimension.
        # This single call replaces the previous for-loop.
        f = interp1d(x_original, sequence_to_interp, kind='linear', axis=0, fill_value='extrapolate')
        interpolated_sequence = f(x_new)

        # Reshape the result back to its original dimensionality.
        if original_ndim == 3:
            # Reshape (target_length, C * S) -> (target_length, C, S)
            interpolated_sequence = interpolated_sequence.reshape(target_length, c, s)

        return interpolated_sequence

    def _data2pixel(self, dataX, dataY, curve=False):
        """
        Converts numerical time series data into a pixel-based image representation.
        Each time step becomes a column in the image, and the value is represented
        by a one-hot encoded pixel in that column.

        Args:
            dataX (np.ndarray): The known part of the sequence (history). Shape (T_hist, C).
            dataY (np.ndarray): The full sequence (history + future). Shape (T_total, C).
            curve (bool): If True, use a different (currently disabled) conversion method.

        Returns:
            tuple: A tuple containing:
                - imgX0 (np.ndarray): Pixel representation of the history.
                - imgY0 (np.ndarray): Pixel representation of the full sequence.
                - d (np.ndarray): Pre-calculated distance matrix for the sequence values.
        """
        dataX = np.copy(dataX.T)
        dataY = np.copy(dataY.T)

        # Clip data to the defined range [-maxScal, maxScal]
        dataX = np.clip(dataX, -self.maxScal, self.maxScal)
        dataY = np.clip(dataY, -self.maxScal, self.maxScal)

        px, TX = dataX.shape # C, T_hist
        py, TY = dataY.shape # C, T_total

        maxstd = self.maxScal
        # Calculate the value represented by each pixel row
        resolution = maxstd * 2 / (self.h - 1)

        if curve:
            # Curve mode is currently disabled in the main logic
            raise NotImplementedError("Curve mode is not implemented.")
        else:
            # Point mode: Create a one-hot encoded image representation
            imgY0 = np.zeros([py, TY, self.h])
            # Calculate the pixel row index for each data point
            indY = np.floor((dataY + maxstd) / resolution).astype('int16')
            indY = np.clip(indY, 0, self.h - 1)

            # Efficiently set the one-hot values
            aY = imgY0.reshape(-1, self.h)
            aY[np.arange(TY * py), indY.flatten()] = 1
            imgY0 = aY.reshape(py, TY, self.h)
            
            # Create the history image by copying the full image and zeroing out the future part
            imgX0 = np.copy(imgY0)
            imgX0[:, TX:, :] = 0

        # Look up the distance vectors for each point in the sequence from the pre-calculated D matrix.
        # This is not directly returned but used for some loss calculations during training.
        d = self.D[list(indY), :]

        return imgX0, imgY0, d

    def _pixel2data(self, imgX0, method='max', sampleNumber=None):
        """
        将基于像素的图像表示转换回数值数据。
        如果提供了 sampleNumber，则从 H 维度的分布中采样。

        Args:
            imgX0 (torch.Tensor or np.ndarray): 像素数据，形状为 (B, C, W, H)。
                                                H 维度是一个经过 softmax 归一化的分布。
            method (str): 当 sampleNumber 为 None 时的转换方法。
                        'max' 取具有最高概率的像素索引。
                        'expectation' 计算加权平均值（期望值）。
            sampleNumber (int, optional): 如果为整数，则代表在 H 维度上按照其概率
                                        采样 sampleNumber 个索引。
                                        如果为 None，则遵循 'method' 参数的逻辑。
                                        默认为 None。

        Returns:
            np.ndarray: 转换后的数值数据。
                        如果 sampleNumber 为 None，形状为 (B, W, C)。
                        如果 sampleNumber 是整数，形状为 (B, W, C, sampleNumber)。
        """
        # 确保输入是4D张量
        if len(imgX0.shape) == 3:
            if isinstance(imgX0, torch.Tensor):
                imgX0 = imgX0.unsqueeze(0)
            else:
                imgX0 = np.expand_dims(imgX0, 0)

        bs, ch, w, h = imgX0.shape

        # --- 采样或确定性转换逻辑 ---
        if sampleNumber is not None:
            # --- 新增：按概率采样 ---
            # 确保数据是 torch.Tensor 以使用 torch.multinomial
            if isinstance(imgX0, np.ndarray):
                imgX0_torch = torch.from_numpy(imgX0).to(self.device) # 假定 self.device 存在
            else:
                imgX0_torch = imgX0

            # 重塑张量以便进行批处理采样: (B, C, W, H) -> (B*C*W, H)
            # .contiguous() 确保张量在内存中是连续的
            probs_flat = imgX0_torch.permute(0, 1, 2, 3).contiguous().view(-1, h)

            # 从每个分布中采样 sampleNumber 个索引
            # replacement=True 意味着可以重复采样同一个索引
            # 结果形状为 (B*C*W, sampleNumber)
            sampled_indices_flat = torch.multinomial(probs_flat, sampleNumber, replacement=True)

            # 将采样结果重塑回原始维度: (B, C, W, sampleNumber)
            indx = sampled_indices_flat.view(bs, ch, w, sampleNumber)
            
            # 将结果转换为 numpy 数组以进行后续计算
            indx = indx.cpu().detach().numpy()

        else:
            # --- 原有逻辑：当 sampleNumber 为 None 时 ---
            if isinstance(imgX0, torch.Tensor):
                imgX0 = imgX0.cpu().detach().numpy()

            if method == 'max':
                # 在每个列中找到具有最大值的像素的索引
                indx = np.argmax(imgX0, axis=-1)
            elif method == 'expectation':
                # 计算像素索引的期望值
                # 假设 imgX0 已经是归一化的，但为稳健性起见，保留归一化步骤
                imgX0_norm = imgX0 / (np.sum(imgX0, axis=-1, keepdims=True) + 1e-8)
                indNumber = np.arange(h) # 代表每个像素位置的索引值
                # 利用广播机制计算期望值
                indx = np.sum(imgX0_norm * indNumber, axis=-1)
            else:
                raise ValueError("方法必须是 'max' 或 'expectation'")

        # --- 将像素索引转换回数值 ---
        maxstd = self.maxScal
        # 注意：这里使用 self.h 是为了与原函数的逻辑保持一致
        resolution = maxstd * 2 / (self.h - 1)

        # 根据是否采样来调整转置操作
        if sampleNumber is not None:
            # indx 形状: (B, C, W, sampleNumber) -> 转置后形状: (B, W, C, sampleNumber)
            transposed_indx = np.transpose(indx, (0, 2, 1, 3))
        else:
            # indx 形状: (B, C, W) -> 转置后形状: (B, W, C)
            transposed_indx = np.transpose(indx, (0, 2, 1))

        res = transposed_indx * resolution - maxstd

        return res

    def _cycleForward(self, model, x):
        """
        Performs a forward pass through the model.
        The name suggests a cyclical or iterative process, but here it's a single pass.

        Args:
            model (nn.Module): The model to use for the forward pass.
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The mask tensor (currently unused in this function).
            xO (torch.Tensor): Original input tensor (currently unused in this function).

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): The model's output tensor.
                - cycleNumber (int): A random number of cycles (for potential future use).
        """
        # The cycleNumber is generated but not used in the current implementation.
        cycleNumber = np.random.randint(1, self.cycleTime + 1)

        with torch.no_grad():
            # The mask and original input xO are not passed to the model here,
            # but the MAE architecture internally handles masking.
            x = model(x, temparture=self.temparture)

        return x, cycleNumber

    def inference(self, x, prediction_length=None,sampleNumber=None,tempature=1):
        self.tempature=tempature
        with torch.no_grad():
          
            # vitime pred
            # --- 1. Data Preparation and Interpolation ---
            x = np.array(x)
            

            if x.ndim == 1: x = x.reshape(-1, 1)
            

            
            # Store original time series lengths
            t_his_original = x.shape[0]
            t_total_original = prediction_length+t_his_original

            # Calculate target lengths for interpolation to match model's fixed input size
            target_total_length = self.seq_len + self.pred_len # e.g., 512*2 + 720*2
            t_his_ratio = t_his_original / t_total_original
            target_his_length = int(t_his_ratio * target_total_length)

            # Interpolate all time series to the target length
            x_interp = self._interpolate_sequence(x, target_his_length)
      
            
            # --- 2. Normalization ---
            # Construct a full sequence for robust normalization
            seq_y = np.zeros((target_total_length, x_interp.shape[1]))
            seq_y[:target_his_length] = x_interp
            seq_y[target_his_length:] = np.mean(x_interp) # Fill future with mean for now

            
            scale = 1
            std = (np.std(x_interp, axis=0).reshape(1, -1) + 1e-7) / scale
            swift = 0
            if hasattr(self.args, 'muNorm'):
                seq = (x_interp ** self.args.muNorm) * np.sign(x_interp)
                mu0 = np.mean(seq, axis=0) + 1e-7
                mu = np.sqrt(np.abs(mu0)) * np.sign(mu0).reshape(1, -1) - swift
            else:
                mu = np.mean(x_interp, axis=0).reshape(1, -1) - swift
  

            # Normalize the data
            seq_x_norm = (x_interp - mu) / std
            seq_y_norm = (seq_y - mu) / std
        

            # --- 3. Convert to Pixel Representation ---
            x_img, y_img, d = self._data2pixel(seq_x_norm, seq_y_norm)
            

            # Apply Gaussian blur to create a soft distribution instead of a single point
            kernel_size = (31, 31)
            sigmaX = 0
            for i in range(x_img.shape[0]):
                x_img[i] = cv2.GaussianBlur(x_img[i], kernel_size, sigmaX) * kernel_size[0]

            # Process and combine envelope images
        

            # --- 4. Model Inference ---
            # Concatenate all processed images into a multi-channel input
            x_combined = x_img
            # print(f"Combined input shape for envelope model: {x_combined.shape}")

            
            # Convert to a PyTorch tensor
            x_tensor = torch.from_numpy(x_combined).float().unsqueeze(0).to(self.device)

            # Create mask (not explicitly used in _cycleForward but required by some model architectures)
            mask = torch.ones_like(x_tensor)
            mask[:, :, :self.seq_len, :] = 0 # 0 indicates known (history), 1 indicates unknown (future)

            # Perform model inference
            xO = x_tensor.clone()
            y_pred, _ = self._cycleForward(self.model, x_tensor)

    
            
            y_pred_np = self._pixel2data(y_pred[:, 0:1, :, :], sampleNumber=sampleNumber) # Use only the first channel for output
            
            # 2. De-normalize the data.
            #    - We take the first item from the batch, shape: (W, C_slice, S), e.g., (64, 1, 5)
            #    - The de-normalization is applied element-wise.
            y_pred_denorm = y_pred_np[0] * std + mu

            # 3. Interpolate the prediction back to the original length.
            #    - The new _interpolate_sequence is designed to handle this 3D input.
            #    - Input `y_pred_denorm` shape: (W, C_slice, S), e.g., (64, 1, 5)
            #    - The function will interpolate the first dimension from W -> t_total_original.
            #    - Output `y_pred_original` shape: (t_total_original, C_slice, S), e.g., (100, 1, 5)
            y_pred_original = self._interpolate_sequence(y_pred_denorm, t_total_original)
            
            # print(y_pred_original.shape,sampleNumber)
            return y_pred_original
