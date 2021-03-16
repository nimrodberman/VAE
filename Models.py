import numpy as np
import torch.nn as nn
import torch


class VAE(nn.Module):
    def __init__(self, sampleSize: int, latentSpaceSize: int, beta: int):
        super(VAE, self).__init__()
        # the encoder component constructed from 2 hidden layer with LeakyReLU as activation function
        self.encoder = nn.Sequential(
            nn.Linear(sampleSize, latentSpaceSize ** 2, True),
            nn.LeakyReLU(),
            nn.Linear(latentSpaceSize ** 2, latentSpaceSize * 2, True),
            nn.LeakyReLU(),
        )

        # the decoder component constructed from 2 hidden layer with LeakyReLU as activation function
        # and Sigmoid as output function
        self.decoder = nn.Sequential(
            nn.Linear(latentSpaceSize, latentSpaceSize ** 2, True),
            nn.LeakyReLU(),
            nn.Linear(latentSpaceSize ** 2, sampleSize, True),
            nn.Sigmoid()
        )
        # the mean and variance connection layers
        self.mu_layer = nn.Linear(latentSpaceSize * 2, latentSpaceSize)
        self.var_layer = nn.Linear(latentSpaceSize * 2, latentSpaceSize)
        self.beta = beta

    def reparameterize(self, mu, logVar, training):
        if training:
            epsilon = torch.rand_like(logVar)
            std = torch.exp(0.5 * logVar)
            return mu + epsilon * std
        else:
            return mu

    def forward(self, x, training):
        muAndVar = self.encoder(x)
        mu = self.mu_layer(muAndVar)
        logVar = self.var_layer(muAndVar)
        z = self.reparameterize(mu, logVar, training)
        x_recon = self.decode(z)
        return x_recon, mu, logVar

    def loss(self, x, y, mu, logVar):
        reconstructionError = torch.nn.BCELoss(x, y)
        klError = (self.beta * 0.5) * torch.sum(torch.exp(logVar) - logVar - 1 + (mu ** 2))
        return reconstructionError + klError
