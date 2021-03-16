import numpy as np
import random
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CelebA


class CelebsFacesTask:

    def __init__(self, trainingSize, testingSize):
        self.trainingSize = trainingSize
        self.testingSize = testingSize

        self.trainData = self.getTestData()
        self.testData = self.testData()

    def getTrainData(self):

        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.ToTensor()])

        dataset = CelebA(root="\celebA",
                         split="train",
                         transform=transform,
                         download=False)

        return dataset

    def getTestData(self):
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.ToTensor()])

        dataset = CelebA(root="\celebA",
                         split="test",
                         transform=transform,
                         download=False)

        return dataset

