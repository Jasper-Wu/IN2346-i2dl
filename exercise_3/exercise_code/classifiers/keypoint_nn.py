import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################

        # input 1 * 96 * 96
        self.conv_1 = nn.Conv2d(1, 32, 4) # 32 * 93 * 93
        init.xavier_normal_(self.conv_1.weight.data)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(2, 2) # 32 * 46 * 46
        self.dropout_1 = nn.Dropout2d(p=0.1)

        self.conv_2 = nn.Conv2d(32, 64, 3)  # 64 * 44 * 44
        init.xavier_normal_(self.conv_2.weight.data)
        # self.elu_2 = nn.ELU()
        # self.pool_2 = nn.MaxPool2d(2, 2)
        self.dropout_2 = nn.Dropout2d(p=0.2)

        self.conv_3 = nn.Conv2d(64, 128, 2)
        init.xavier_normal_(self.conv_3.weight.data)
        # self.elu_3 = nn.ELU()
        # self.pool_3 = nn.MaxPool2d(2, 2) # 128 * 10 * 10
        self.dropout_3 = nn.Dropout2d(p=0.3)

        self.conv_4 = nn.Conv2d(128, 256, 1) # 256 * 10 * 10
        init.xavier_normal_(self.conv_4.weight.data)
        # self.elu_4 = nn.ELU()
        # self.pool_4 = nn.MaxPool2d(2, 2) # 256 * 5 * 5
        self.dropout_4 = nn.Dropout2d(p=0.4)

        self.fc_1 = nn.Linear(256 * 5 * 5, 1000)
        init.xavier_normal_(self.fc_1.weight.data)
        self.dropout_5 = nn.Dropout2d(p=0.5)

        self.fc_2 = nn.Linear(1000, 1000)
        init.xavier_normal_(self.fc_2.weight.data)
        self.dropout_6 = nn.Dropout2d(p=0.6)

        self.fc_3 = nn.Linear(1000, 30)
        init.xavier_normal_(self.fc_3.weight.data)

        # in the paper, conv weights are initialized with random numbers
        # drawn from uniform distribution
        # fc layers weights are initialized using Glorot uniform initialization

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################

        # if (x.shape == torch.Size([1, 96, 96])):
        #     x = x.unsqueeze(0)

        x = self.dropout_1(self.pool(self.elu(self.conv_1(x))))
        x = self.dropout_2(self.pool(self.elu(self.conv_2(x))))
        x = self.dropout_3(self.pool(self.elu(self.conv_3(x))))
        x = self.dropout_4(self.pool(self.elu(self.conv_4(x))))

        x = x.view(x.size()[0], -1)

        x = self.dropout_5(self.elu(self.fc_1(x)))
        x = self.dropout_6(self.fc_2(x))
        x = self.fc_3(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
