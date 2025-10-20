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

        Args:
            sequence (np.ndarray): The input sequence, shape (T, C) or (T,).
            target_length (int): The desired length of the output sequence.

        Returns:
            np.ndarray: The interpolated sequence, shape (target_length, C).
        """
        if len(sequence.shape) == 1:
            sequence = sequence.reshape(-1, 1)

        T, C = sequence.shape
        if T == target_length:
            return sequence

        interpolated_sequence = np.zeros((target_length, C))
        x_original = np.arange(T)
        x_new = np.linspace(0, T - 1, target_length)

        for i in range(C):
            f = interp1d(x_original, sequence[:, i], kind='linear', fill_value='extrapolate')
            interpolated_sequence[:, i] = f(x_new)
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

    def _pixel2data(self, imgX0, method='max'):
        """
        Converts the pixel-based image representation back to numerical data.

        Args:
            imgX0 (torch.Tensor or np.ndarray): The pixel data, shape (B, C, W, H).
            method (str): Method for conversion. 'max' takes the pixel with the highest
                          intensity. 'expectation' calculates a weighted average.

        Returns:
            np.ndarray: The converted numerical data, shape (B, C, W).
        """
        if len(imgX0.shape) == 3:
            imgX0 = imgX0.unsqueeze(0)

        bs, ch, w, h = imgX0.shape

        if isinstance(imgX0, torch.Tensor):
            imgX0 = imgX0.cpu().detach().numpy()

        if method == 'max':
            # Find the index of the pixel with the maximum value in each column
            indx = np.argmax(imgX0, axis=-1)
        elif method == 'expectation':
            # Calculate the expected value of the pixel index
            # Normalize probabilities along the height dimension
            imgX0 = imgX0 / (np.sum(imgX0, axis=-1, keepdims=True) + 1e-8)
            indNumber = np.arange(0, h)
            imgX0 *= indNumber # Weight each probability by its index
            indx = np.sum(imgX0, axis=-1) # Sum to get the expectation

        # Convert pixel index back to numerical value
        maxstd = self.maxScal
        resolution = maxstd * 2 / (self.h - 1)
        res = np.transpose(indx, (0, 2, 1)) * resolution - maxstd

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

    def inference(self, x, prediction_length=None):
        
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

    
            
            # Extract the prediction and convert it back to numerical data
            y_pred_np = self._pixel2data(y_pred[:, 0:1, :, :]) # Use only the first channel for output

            # De-normalize the data to its original scale
            y_pred_denorm = y_pred_np[0] * std + mu

            # Interpolate the prediction back to the original length
            y_pred_original = self._interpolate_sequence(y_pred_denorm, t_total_original)
            
            
            return y_pred_original
