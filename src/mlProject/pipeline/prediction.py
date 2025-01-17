import joblib 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path


class linearRegression(nn.Module): # all the dependencies from torch will be given to this class [parent class] # nn.Module contains all the building block of neural networks:
  def __init__(self,input_dim):
    super(linearRegression,self).__init__()  # building connection with parent and child classes
    self.fc1=nn.Linear(input_dim,64)          # hidden layer 1
    self.fc2=nn.Linear(64,32)                  # hidden layer 2
    self.fc3=nn.Linear(32,16)                   # hidden layer 3
    self.fc4=nn.Linear(16,8)                   #hidden layer 4
    self.fc5=nn.Linear(8,1)                   # last layer

  def forward(self,d):
    out=torch.relu(self.fc1(d))              # input * weights + bias for layer 1
    out=torch.relu(self.fc2(out))            # input * weights + bias for layer 2
    out=torch.relu(self.fc3(out))            # input * weights + bias for layer 3
    out=torch.relu(self.fc4(out))            # input * weights + bias for layer 4
    out=self.fc5(out)                        # input * weights + bias for last layer
    return out                               # final outcome

class PredictionPipeline:
    def __init__(self):
        pass

    
    def predict(self, data):
        data_tensor = torch.tensor(data.values, dtype=torch.float32)
        input_dim = data_tensor.shape[1]
        model = linearRegression(input_dim)
        saved_data = joblib.load(Path('artifacts/model_trainer/model.pkl'))
        model.load_state_dict(saved_data['model_state_dict'])
        with torch.no_grad():
                model.eval()   # make model in evaluation stage
                predicted_tensor = model(data_tensor)
        prediction = predicted_tensor.numpy()
        return prediction