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
    def __init__(self, device='cuda:0', model_name='MAE',lookbackRatio=1):
        """
        Initialize the ViTime predictor.

        Args:
            device (str): Compute device (e.g., 'cuda:0' or 'cpu').
            model_name (str): Model name to select backbone/weights.
            lookbackRatio (float): Fixed lookback ratio when not adaptive.
           
        """
        
        self.lookbackRatio = lookbackRatio
        self.predictor = ViTimePredictor(device=device, model_name=model_name)
         
    def prediction(self, historical_data, future_length):
        '''
        historical_data: n-dimensional numpy array (T[, C]).
        Returns an array of length `future_length`.
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
        
       

    
        full_prediction = self.predictor(historical_data, future_length)
        
        prediction = np.asarray(full_prediction).flatten()[len(historical_data):len(historical_data)+future_length]
        

        return prediction



if __name__ == '__main__':
    xData=np.sin(np.arange(512)/10)+np.sin(np.arange(512)/5+50)+np.cos(np.arange(512)+50)
    prediction_length=720
    vitime = ViTimePrediction(device='cuda:0',model_name='MAE',lookbackRatio=None)
    prediction=vitime.prediction(xData,prediction_length)
    plt.plot(np.concatenate([xData,prediction.flatten()],axis=0),label='Prediction')
    plt.plot(xData,label='Input Sequence')
    plt.legend()
    plt.savefig('test.jpg')







