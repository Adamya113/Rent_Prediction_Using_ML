import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import dagshub
from src.mlProject.entity.config_entity import ModelEvaluationConfig
from src.mlProject.utils.common import save_json
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

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        test_x_tensor = torch.tensor(test_x.values, dtype=torch.float32)
        input_dim = test_x_tensor.shape[1]
        model = linearRegression(input_dim)
        saved_data = joblib.load(self.config.model_path)
        model.load_state_dict(saved_data['model_state_dict'])

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        dagshub.init(repo_owner='Adamya113', repo_name='Rent_Prediction_Using_ML', mlflow=True)
        with mlflow.start_run():
   
            with torch.no_grad():
                model.eval()   # make model in evaluation stage
                predicted_tensor = model(test_x_tensor)
            predicted_y =  predicted_tensor.numpy()

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_y)
            
            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(model, "model", registered_model_name="LinearRegression")
            else:
                mlflow.pytorch.log_model(model, "model")

    
