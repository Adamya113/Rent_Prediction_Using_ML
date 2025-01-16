import pandas as pd
import os
from src.mlProject import logger
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from src.mlProject.entity.config_entity import ModelTrainerConfig

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

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        # Convert to PyTorch tensors
        X_train = torch.tensor(train_x.values, dtype=torch.float32)
        X_test = torch.tensor(test_x.values, dtype=torch.float32)
        y_train = torch.tensor(train_y.values, dtype=torch.float32)
        y_test = torch.tensor(test_y.values, dtype=torch.float32)

        input_dim=X_train.shape[1]
        torch.manual_seed(42)  # to make initialized weights stable:
        model = linearRegression(input_dim)
        loss=nn.MSELoss() # loss function
        optimizers=optim.Adam(params=model.parameters(),lr=self.config.lr)

        # training the model:
        for i in range(self.config.num_of_epochs):
            # give the input data to the architecure
            y_train_prediction=model(X_train)  # model initializing
            loss_value=loss(y_train_prediction.squeeze(),y_train)   # find the loss function:
            optimizers.zero_grad() # make gradients zero for every iteration so next iteration it will be clear
            loss_value.backward()  # back propagation
            optimizers.step()  # update weights in NN

            # print the loss in training part:
            if i % 10 == 0:
                logger.info(f"[epoch:{i}]: The training loss={loss_value}")

        joblib.dump({'model_state_dict': model.state_dict()}, os.path.join(self.config.root_dir, self.config.model_name))
