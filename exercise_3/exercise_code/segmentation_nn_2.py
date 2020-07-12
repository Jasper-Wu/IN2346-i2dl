"""SegmentationNN"""
import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.models as models

# https://github.com/meetshah1995/pytorch-semseg
# https://github.com/ferrophile/in2346

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        self.num_classes = num_classes

        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True))
        # pay much attention to this padding size !!!

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.num_classes, 1))

        self.pool_4 = nn.Conv2d(512, self.num_classes, 1)
        self.pool_3 = nn.Conv2d(256, self.num_classes, 1)
        # aligned to classification task (softmax)

        # use pre-trained vgg16 model parameters from PyTorch
        
        vgg16 = models.vgg16(pretrained=True)
        blocks = [self.conv_1,
                  self.conv_2,
                  self.conv_3,
                  self.conv_4,
                  self.conv_5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, block in enumerate(blocks):
            # for layer in each blocks, assign values **in pairs**
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

        for i1, i2 in zip([0, 3], [0, 3]):
            # operation to each convolutional layer
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
            # in vgg16, the layers in classifier are fully connected layers
            # numbers of parameters: 4096*7*7*512
            # here is fully convolutional layers, so needs to be reformed

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

        conv1 = self.conv_1(x)
        conv2 = self.conv_2(conv1)
        conv3 = self.conv_3(conv2)
        conv4 = self.conv_4(conv3)
        conv5 = self.conv_5(conv4)

        score = self.classifier(conv5)
        score_pool_4 = self.pool_4(conv4)
        score_pool_3 = self.pool_3(conv3)

        # interpolate / upsample_bilinear
        score = F.interpolate(score, score_pool_4.size()[2:])
        score += score_pool_4
        score = F.interpolate(score, score_pool_3.size()[2:])
        score += score_pool_3
        x = F.interpolate(score, x.size()[2:])
        # i think this procedure may be different from what i have read ?
        # shouldn't 32x upsampling to become the origin size and add the 16x
        # upsampling from 4th pooling layer ?

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
