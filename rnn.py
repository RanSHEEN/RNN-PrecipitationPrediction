# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:20:43 2024

@author: Ran XIN
"""

# to download the laser time series
import pyreadr

import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setting the default font size globally
plt.rcParams.update({'font.size': 30})

class RNN(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_dim=32, n_layers=3, seq_length=10, dropout_rate=0.25):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # Définit une couche RNN en utilisant la classe nn.RNN de PyTorch
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, nonlinearity='relu')
        # self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        # Une couche de dropout pour réduire le surajustement
        self.dropout = nn.Dropout(dropout_rate)
        # Définit une couche linéaire (ou fully-connected) 
        # Cette couche est utilisée pour transformer la sortie du RNN en une sortie de la taille désirée.
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # Sortie avant f(Vht + by) et ht+1
        r_out, hidden = self.rnn(x, hidden)
        r_out = self.dropout(r_out)
        # sortie prédit
        r_out = self.fc(r_out)
        return r_out, hidden
    
    
    
class Dataloader:
    def __init__(self, data, test_size_ratio = 0.2):
        self.data = data
        self.separate_train_test(test_size_ratio)
    
    def __len__(self):
        return self.data.shape
        
    def separate_train_test(self, test_size_ratio):
        len_train = int(self.data.shape[0] * (1-test_size_ratio))
        self.train = self.data[:len_train]
        self.test = self.data[len_train:]
        
        # printing out train and test sets
        print('train datas: ')
        print(self.train.head())
        print(self.train.shape)
        print('')
        print('test datas: ')
        print(self.test.head())
        print(self.test.shape)
        print('')
        
        
        
class Dataset: # generer la sequence
    def __init__(self, data, seq_length = 20, step = 1):
        # Vérifier si 'data' est un tableau numpy
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy table.")
        # Cette ligne de code redimensionne le tableau data pour qu'il ait une taille de (seq_length + 1, 1), 
        self.seq_length = seq_length
        tmp = data[step*seq_length:min((step+1)*seq_length+1, len(data))]
        # x n'affiche pas le dèrnier résultat, y n'affiche pas le premier résultat
        self.x, self.y = tmp[:-1], tmp[1:]
        
        
        
###############################################################################
###############################################################################
# def evaluate_errors(true_values, predictions):
#     from sklearn.metrics import mean_absolute_error, mean_squared_error
#     mae = mean_absolute_error(true_values, predictions)
#     mse = mean_squared_error(true_values, predictions)
#     rmse = np.sqrt(mse)
#     return mae, mse, rmse


def train(rnn, train_data, test_data, print_every=20, num_epochs=50):
    # initialize the hidden state
    hidden = None
    #hidden_init = True
    train_losses_epoch = []
    val_losses_epoch = []  # Liste pour stocker les pertes de validation
    train_prediction = []
    
    # L1loss and Adam optimizer with a learning rate of 0.01
    # nn.L1Loss is generally used in regression problems where the goal is to predict a continuous output. 
    criterion = nn.MSELoss() # nn.L1Loss penalizes the errors linearly. 
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        train_losses = []
        validation_losses = []
        
        # Train
        rnn.train()
        for batch_i, dataset in enumerate(train_data):
            # Convertir ensuite les données numpy en tenseurs PyTorch.
            x_tensor = torch.Tensor(dataset.x).unsqueeze(0) # unsqueeze gives a 1, batch_size dimension 
            # unsqueeze(0) est pour creer le premier 1 dans le batch [1,20,1]
            y_tensor = torch.Tensor(dataset.y).unsqueeze(0)
            # Check if the sequence length is greater than 0
            if x_tensor.shape[1] == 0: continue
            # outputs from the rnn
            prediction, hidden = rnn(x_tensor, hidden)
            
            ## Representing Memory ##
            # make a new variable for hidden and detach the hidden state from its history
            # this way, we don't backpropagate through the entire history
            hidden = hidden.data
            # hidden = (hidden[0].detach(), hidden[1].detach())
            
            # calculate the loss 
            loss = criterion(prediction, y_tensor)
            train_losses.append(loss.item())
            # zero gradients(réinitialise les gradients) 
            optimizer.zero_grad()
            # perform backprop and update weights
            loss.backward(retain_graph=True)
            optimizer.step()
        train_losses_epoch.append(np.mean(train_losses))
        
        # Validation
        rnn.eval()  # Mettre le modèle en mode évaluation
        with torch.no_grad():
            for dataset in test_data:
                x_val_tensor = torch.Tensor(dataset.x).unsqueeze(0)
                y_val_tensor = torch.Tensor(dataset.y).unsqueeze(0)
                # Check if the sequence length is greater than 0
                if x_val_tensor.shape[1] == 0: continue
                # Prediction
                val_prediction, _ = rnn(x_val_tensor, hidden)
                # Calculate the loss
                val_loss = criterion(val_prediction, y_val_tensor)
                validation_losses.append(val_loss.item())  # Ajouter la perte de validation à la liste
        val_losses_epoch.append(np.mean(validation_losses))
        
    # Prediction
    train_prediction = []
    test_prediction = []
    len_train = 0
    len_test = 0
    # hidden = None
    for batch_i, dataset in enumerate(train_data):
        x_pred_tensor = torch.Tensor(dataset.x).unsqueeze(0)
        if x_pred_tensor.shape[1] == 0: continue
        pred, hidden = rnn(x_pred_tensor, hidden)
        p = pred.detach().numpy().flatten()
        train_prediction.append(p)
        len_train += x_pred_tensor.shape[1]
    for batch_i, dataset in enumerate(test_data):
        x_pred_tensor = torch.Tensor(dataset.x).unsqueeze(0)
        if x_pred_tensor.shape[1] == 0: continue
        pred, hidden = rnn(x_pred_tensor, hidden)
        p = pred.detach().numpy().flatten()
        test_prediction.append(p)
        len_test += x_pred_tensor.shape[1]
    len_total = len_train+len_test+1 # Calculer la longueur des ensembles de donnees
    
    # Tracer des prédictions
    plt.figure(figsize=(16, 4))
    plt.plot(np.concatenate([data.x.flatten() for data in train_data]), 'b', label='Train data', markersize = 3)
    plt.plot(np.concatenate(train_prediction), 'r', label='Predictions of train data', markersize = 3)
    plt.plot(range(len_train+1,len_total), np.concatenate([data.x.flatten() for data in test_data]), 'g', label='Validation data')
    plt.plot(range(len_train+1,len_total), np.concatenate(test_prediction), color = 'orange', label='Predictions of validation data')
    plt.xlabel('Timesteps')
    plt.ylabel('Values')
    plt.title('Data and Predictions with nn.MSELoss()')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure()
    plt.plot(train_losses_epoch, 'b', label='Train Loss')
    plt.plot(val_losses_epoch, 'r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Time with nn.MSELoss()')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return rnn


def count_parameters(rnn):
    return sum(p.numel() for p in rnn.parameters() if p.requires_grad)

###############################################################################
###############################################################################
if __name__ == "__main__":
    # close all the previous figures
    plt.close('all')
    
    # Lire les fichiers
    filename = "SantaFe.A.cont.rda"
    filename = "SantaFe.A.rda"
    result = pyreadr.read_r(filename) # also works for Rds
    # result is a dictionary where keys are the name of objects 
    # and the values python objects
    print(result.keys()) # let's check what objects we got
    santaFe_a = result['SantaFe.A'] # extract the pandas data frame for object df1
    santaFe_a.T
    # Afficher le fichier
    plt.figure(figsize=(16,4))
    plt.plot(santaFe_a['V1'])
    plt.grid(True);
    
    # Définir les hyperparamètres
    input_size=1
    output_size=1
    hidden_dim=32
    n_layers=3
    n_steps = 200 # nombre total d'itérations
    seq_length = 20
    print_every = 50 # frequence d'imprimer
    test_size_ratio = 0.2
    n_steps_sin = 100
    print_every_sin = 20

    # Data
    data_1 = Dataloader(santaFe_a)
    #time_steps_1 = list(range(seq_length+1))
    #data_1.train.sort_index(inplace=True)
    #data_1.test.sort_index(inplace=True)
    train_array = data_1.train.to_numpy(dtype=np.float32)
    data_train_1 = [Dataset(train_array, step = n) for n in range(n_steps)]
    test_array = data_1.test.to_numpy(dtype=np.float32)
    data_test_1 = [Dataset(test_array, step = n) for n in range(n_steps)]
    plt.figure(figsize=(16,4))
    plt.plot(data_1.train, 'b', label='Train data')
    plt.plot(data_1.test, 'g', label='Test data')
    # instantiate an RNN
    rnn = RNN(input_size, output_size, hidden_dim, n_layers, seq_length)
    print("Model structure: ", rnn)
    print("")
    # train the rnn and monitor results
    trained_rnn = train(rnn, data_train_1, data_test_1, print_every)
    print("Number of parameters: ", count_parameters(rnn))
        
    # Vérifier avec data sinusoidal
    d_sin = {'sin':np.sin(np.linspace(0, (n_steps_sin+1)*np.pi, (seq_length+1)*n_steps_sin))}
    df_sin = pd.DataFrame(data=d_sin)
    df_sin.T
    #time_steps_sin = np.linspace(0, np.pi, seq_length+1)
    data_sin = Dataloader(df_sin)
    data_sin.train.sort_index(inplace=True)
    data_sin.test.sort_index(inplace=True)
    train_array_sin = data_sin.train.to_numpy(dtype=np.float32)
    data_train_sin = [Dataset(train_array_sin, step = n) for n in range(n_steps_sin)]
    test_array_sin = data_sin.test.to_numpy(dtype=np.float32)
    data_test_sin = [Dataset(test_array_sin, step = n) for n in range(n_steps_sin)]
    plt.figure(figsize=(16,4))
    plt.plot(data_sin.train, 'b', label='Train data')
    plt.plot(data_sin.test, 'g', label='Test data')
    # train the rnn and monitor results
    trained_rnn_sin = train(rnn, data_train_sin, data_test_sin, print_every_sin)
    print("Number of parameters: ", count_parameters(rnn))
        
# Les problemes de predictions des donnes forte: prob d'initialition