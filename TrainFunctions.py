# Neural Networks
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split

# custom import for early stopping
from earlystopping import EarlyStopping

import os

import numpy as np

def train_model(model, loss_fn, optimizer, num_epochs, train_dataloader, val_dataloader, patience, path, device, load_best = False):

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path = path)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = data[0].float().to(device)
            labels = data[1].float().to(device)

            #labels = labels.long() # convert to expected target datatype (Long which is equivalent to int here)
#             labels = labels.type(torch.LongTensor)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, h = model(inputs)

            loss = loss_fn(outputs,labels.view(-1).long()) # do i need to fix what's in here (even necessary to have it)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())

#             # print statistics
#             running_loss += loss.item()
#             if i % 2000 == 1999:    # print every 2000 mini-batches
#                 print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#                 running_loss = 0.0
              
            
        # Validation Loss
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(val_dataloader, 0):
                inputs, labels = data
                inputs = data[0].float().to(device)
                labels = data[1].float().to(device)
            
            outputs, h = model(inputs)
            
            loss = loss_fn(outputs,labels.view(-1).long())
            
            valid_losses.append(loss.item())
            
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)   
        
        epoch_len = len(str(num_epochs))
        print_update = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        
        print(print_update)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        early_stopping(valid_loss, model)
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(path))
    
    
    print('Finished Training')

    return  model, avg_train_losses, avg_valid_losses


def getDataloaders(train_dataset, test_dataset, batch_size):
    tr_sz_ratio = 0.8
    tr_sz = int(len(train_dataset) * tr_sz_ratio)
    train_subset, val_subset = random_split(
        train_dataset, [tr_sz, len(train_dataset) - tr_sz])

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, val_dataloader, test_dataloader


def create_file(model_name, output_size, hidden_size, batch_size, dropout = 0, file_name = "/checkpoint.pt"):
    # create directory and file if does not exist
    dir_path = "./model_checkpoints/{}".format(model_name)    
    isExist = os.path.exists(dir_path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(dir_path)
    f = open(dir_path + "/output{}&hidden={}&batch={}&dropout={}-checkpoint.pt".format(output_size, hidden_size, batch_size, dropout) , 'w')
    f.close()
    path = dir_path + "/output{}&hidden={}&batch={}&dropout={}-checkpoint.pt".format(output_size, hidden_size, batch_size, dropout)
    
    return path

def load_data(output_size):
    # datasets and dataloaders
    data_path = "/home/awozny/click_data"
    trainset_path = "/train_dataset_{}.pt".format(output_size)
    testset_path = "/test_dataset_{}.pt".format(output_size)
    train_dataset = torch.load(data_path + trainset_path)
    test_dataset = torch.load(data_path + testset_path)
    
    return train_dataset, test_dataset
