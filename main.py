import numpy as np
import random
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from Models import VAE

writer = SummaryWriter('runs/copy_task_lvf_experiment_1')

""" info: encode characters array to one vector array that represent input sequence.
    input: [int] sequence , int dict_size , int seq_len, int batch_size
    output: [dict_size*1 vector]"""


# ---- hyper parameters ---- #
lr = 0.001
beta = 1

# ---- collect and prepare data ---- #


# ---- create model and optimizer ---- #
ourModel = VAE()
ourOptimizer = torch.optim.Adam(ourModel.parameters(), lr=lr)

def trainModel(inputs: Tensor, epochs, optimizer, model: VAE):
    for epoch in range(0, epochs+1):
        # switch to train mode
        model.train(True)
        # train of each training sample
        trainingLoss = 0
        for x in inputs:
            # calculated the reconstructed x
            recX, mu, logVar = model.forward(x, True)

            # loss calculation
            loss = model.loss(x, recX, mu, logVar)
            trainingLoss += loss.item()

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epochs % 10 == 0:
            print(f'===> Epoch: {epoch}/{epochs} Average loss: {trainingLoss / len(inputs):.4f}')
























