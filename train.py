# manipulating data
import numpy as np

# 
from sklearn.utils.class_weight import compute_class_weight

# Neural Networks
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim

# Custom Metrics and Evaluation
from metrics import bin_class_metrics, multiclass_metrics, evaluate_model_metrics, print_metrics, PlotLoss
from ModelDefinitions import RNNClassifier, LSTMClassifier, GRUClassifier
from TrainFunctions import create_file, load_data, getDataloaders, train_model

# handling time data
import time # for counting time for something to run

# plotting
import matplotlib.pyplot as plt 

# handling files
import os

# select GPU / CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Device:', device)


# train RNN, LSTM, and GRU
ModelNames = ["gru", "lstm", "vanilla_rnn"]
ModelInstantiators = [GRUClassifier, LSTMClassifier, RNNClassifier]

# choose which problem to solve
output_size = 3

# get datasets
train_dataset, test_dataset = load_data(output_size)

# parameters
input_size = len(train_dataset[0][0][0])
n_layers = 1
bias = False
learning_rate = 0.001
num_epochs = 35

# hyperparameters to tune
hidden_size = [64, 128]
dropout = [0, 0.2, 0.3, 0.4]

for ind, Classifier in enumerate(ModelInstantiators):
    print("------------------------------------------------------------")
    print(ModelNames[ind].upper())
    print("------------------------------------------------------------")

    model_name = ModelNames[ind]

    # get class weights for imbalanced datasets
    y = train_dataset[:][1]
    class_weights=compute_class_weight('balanced',classes = np.unique(y), y = y.numpy().reshape(-1))
    class_weights=torch.tensor(class_weights,dtype=torch.float)

    # store all losses, models for evaluation
    model_val_loss_tot = []
    model_train_loss_tot = []
    model_tot = []
    hyperparams_tot = []

    # train model on hyperparameters
    for h in hidden_size:
            for d in dropout:
                b = 64
                
                hyperparams = {"hidden_size": h, "batch_size": b, "dropout": d}
                print(hyperparams)
                hyperparams_tot.append(hyperparams)

                # create path to store checkpoints
                path = create_file(model_name, output_size, h, b, d)

                # get dataloaders
                train_dataloader, val_dataloader, test_dataloader = getDataloaders(train_dataset, test_dataset, b)

                # instantiate model
                model = Classifier(input_size, h, output_size, device, n_layers, d, bias).to(device)

                loss_fn = torch.nn.CrossEntropyLoss() # weight = class_weights
                torch.set_default_tensor_type(torch.FloatTensor)

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                start_time = time.time()
                # train
                model, model_train_loss, model_val_loss = train_model(model, loss_fn, optimizer, num_epochs, train_dataloader, val_dataloader, 2, path, device, False)
                model_tot.append(model)
                model_val_loss_tot.append(model_val_loss)
                model_train_loss_tot.append(model_train_loss)

                print("--- %s seconds ---" % (time.time() - start_time))
