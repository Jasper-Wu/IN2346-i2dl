"""SegmentationNN"""
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
import torchvision.models as models

# https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        model_layers = list(models.resnet50(pretrained=True).children)[:-2]
        self.features = nn.Sequential(*model_layers)
        self.classifiers = nn.Sequential(nn.Dropout2d(),
                                  nn.Conv2d(2048, 2048, kernel_size=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(2048, num_classes, kernel_size=1))

        # model.fc = nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1)


        # self.features   = models.alexnet(pretrained=True).features
        # self.classifier = nn.Sequential(nn.Dropout(),
        #                           nn.Conv2d(256, 4096, kernel_size=1),
        #                           nn.ReLU(inplace=True),
        #                           nn.Dropout(),
        #                           nn.Conv2d(4096, 4096, kernel_size=1),
        #                           nn.ReLU(inplace=True),
        #                           nn.Conv2d(4096, num_classes, kernel_size=1))

        # self.conv = nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        input_shape = x.shape[-2:]

        x = self.features(x)
        x = self.classifiers(x)

        x = nn.functional.interpolate(input=x, size=input_shape, mode='bilinear', align_corners=False)

        # x = self.features(x)
        # x = self.classifier(x)
        
        # x = nn.functional.interpolate(input=x, scale_factor=40)
        # x = self.conv(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
