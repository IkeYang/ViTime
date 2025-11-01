import numpy as np
import matplotlib.pyplot as plt


# ============ Import tools ============
from tools import (
    ViTimePredictor,
)

import numpy as np
import pandas as pd
import pandas as pd
import numpy as np

class ViTimePrediction():
    def __init__(self, device='cuda:0', model_name='MAE',lookbackRatio=1,tempature=1):
        """
        Initialize the ViTime predictor.

        Args:
            device (str): Compute device (e.g., 'cuda:0' or 'cpu').
            model_name (str): Model name to select backbone/weights.
            lookbackRatio (float): Fixed lookback ratio when not adaptive.
           
        """
        
        self.lookbackRatio = lookbackRatio
        self.predictor = ViTimePredictor(device=device, model_name=model_name,tempature=tempature)
        
         
    def prediction(self, historical_data, future_length,sampleNumber=None):
        '''
        historical_data: n-dimensional numpy array (T[, C]).
        Returns an array of length `future_length`.
        if sampleNumber is None, Output a deterministic prediction; if not, switch to the probabilistic prediction mode and indicate the number of samples.
        '''

            
        historical_length_orig = historical_data.shape[0]
        
        # Apply lookbackRatio to crop history
        if self.lookbackRatio is not None:
            lookback_len = int(future_length * self.lookbackRatio)
        else:
            lookback_len=historical_length_orig
        # Ensure we do not exceed original history length
        lookback_len = min(lookback_len, historical_length_orig)
        
        if lookback_len > 0:
            historical_data = historical_data[-lookback_len:]
        
       

        predictor = self.predictor
        full_prediction = predictor(historical_data, future_length,sampleNumber=sampleNumber)[:,0]
        
        prediction = full_prediction[len(historical_data):len(historical_data)+future_length]
        
        # Step 3: check and linearly impute NaNs in output
        if np.isnan(np.sum(prediction)):
            s = pd.Series(prediction)
            s.interpolate(method='linear', limit_direction='both', inplace=True)
            prediction = s.to_numpy()


        return prediction



if __name__ == '__main__':
    xData=np.sin(np.arange(512)/10)+np.sin(np.arange(512)/5+50)+np.cos(np.arange(512)+50)
    prediction_length=720
    vitime = ViTimePrediction(device='cuda:0',model_name='MAE',lookbackRatio=None,tempature=1)
    prediction=vitime.prediction(xData,prediction_length)
    plt.plot(np.concatenate([xData,prediction.flatten()],axis=0),label='Prediction')
    plt.plot(xData,label='Input Sequence')
    plt.legend()
    plt.savefig('test.jpg')



    ### Probability prediction shows that under such conditions, we usually set the temperature to 8 and set the sampleNumber to the number of samples we want to take.
    vitime = ViTimePrediction(device='cuda:0',model_name='MAE',lookbackRatio=None,tempature=8)
    prediction_samples = vitime.prediction(xData, prediction_length, sampleNumber=100)

    # --- 2. Calculate statistics from samples (No change in this logic) ---
    median_prediction = np.median(prediction_samples, axis=1)
    p95 = np.percentile(prediction_samples, 95, axis=1)
    p5 = np.percentile(prediction_samples, 5, axis=1)
    p75 = np.percentile(prediction_samples, 75, axis=1)
    p25 = np.percentile(prediction_samples, 25, axis=1)

    # --- 3. Create time axes for plotting (No change in this logic) ---
    len_input = len(xData)
    x_axis_input = np.arange(len_input)
    x_axis_pred = np.arange(len_input, len_input + prediction_length)

    # --- 4. Create the refined visualization ---

    # Apply a modern plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define a professional color palette
    INPUT_COLOR = 'black'
    MEDIAN_COLOR = '#E63946'  # A muted, strong red
    CI_50_COLOR = '#457B9D'   # A deep, flat blue
    CI_90_COLOR = '#A8DADC'   # A light, complementary cyan

    # Plot the 90% prediction interval (lightest background)
    ax.fill_between(x_axis_pred, p5, p95, color=CI_90_COLOR, alpha=0.9, label='90% Prediction Interval')
    # Plot the 50% prediction interval (darker foreground)
    ax.fill_between(x_axis_pred, p25, p75, color=CI_50_COLOR, alpha=0.8, label='50% Prediction Interval')
    # Plot the historical data
    ax.plot(x_axis_input, xData, color=INPUT_COLOR, linewidth=1.5, label='Input Sequence')
    # Plot the median forecast line on top of the intervals
    ax.plot(x_axis_pred, median_prediction, color=MEDIAN_COLOR, linestyle='--', linewidth=2.0, label='Median Prediction')

    # Customize titles, labels, and legend for a clean look
    ax.set_title('Probabilistic Time Series Forecast', fontsize=16)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5) # Make grid lines subtle

    # Ensure the plot layout is tight
    plt.tight_layout()

    # Save the figure
    save_path = 'test_Prob.jpg'
    plt.savefig(save_path, dpi=300) # dpi=300 for higher resolution






