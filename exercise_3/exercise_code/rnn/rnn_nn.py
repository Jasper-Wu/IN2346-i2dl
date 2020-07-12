import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        super().__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        #######################################################################
        # TODO: Build a simple one layer RNN with an activation with the      #
        # attributes defined above and a forward function below. Use the      #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h as 0 if these values are not given.                     #
        #######################################################################

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)
        if activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################

        T = x.size()[0]
        if h == None:
            h = torch.zeros((1, x.size()[1], self.hidden_size))
        h = h[-1, :, :]
        for t in range(T):
            x_t = x[t, :, :]
            h = self.act(self.i2h(x_t) + self.h2h(h))
            h_seq.append(h)
        h_seq = torch.stack(h_seq)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, h


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=20):
        super().__init__()
        #######################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes #
        # defined above and a forward function below. Use the                 #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h and c as 0 if these values are not given.               #
        #######################################################################

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_forget = nn.Linear(input_size, hidden_size)
        self.U_forget = nn.Linear(hidden_size, hidden_size)
        self.W_input = nn.Linear(input_size, hidden_size)
        self.U_input = nn.Linear(hidden_size, hidden_size)
        self.W_output = nn.Linear(input_size, hidden_size)
        self.U_output = nn.Linear(hidden_size, hidden_size)
        self.W_cell = nn.Linear(input_size, hidden_size)
        self.U_cell = nn.Linear(hidden_size, hidden_size)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None, c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = None
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################

        seq_len, batch_size, input_size = x.size()
        if h is None:
            h = torch.zeros((1, batch_size, self.hidden_size))
        if c is None:
            c = torch.zeros((1, batch_size, self.hidden_size))
        h = h[-1, :, :]
        c = c[-1, :, :]
        h_seq = []
        
        for t in range(seq_len):
            x_t = x[t, :, :]
            forget_t = torch.sigmoid(self.W_forget(x_t) + self.U_forget(h))
            input_t = torch.sigmoid(self.W_input(x_t) + self.U_input(h))
            output_t = torch.sigmoid(self.W_output(x_t) + self.U_output(h))
            c = torch.mul(forget_t, c)
            c += torch.mul(input_t, torch.tanh(self.W_cell(x_t) + self.U_cell(h)))
            h = torch.mul(output_t, torch.tanh(c))
            h_seq.append(h)
        h_seq = torch.stack(h_seq)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, (h, c)


class RNN_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128,
                 activation="relu"):
        super(RNN_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a RNN classifier                                       #
        #######################################################################

        self.hidden_size = hidden_size
        self.RNN = nn.RNN(input_size, hidden_size)
        self.fc_1 = nn.Linear(hidden_size, 64)
        self.fc_2 = nn.Linear(64, classes)

    def forward(self, x):
        batch_size = x.size()[1]
        h_seq, h = self.RNN(x)
        h = F.dropout(F.relu(self.fc_1(h.reshape(batch_size, self.hidden_size))))
        h = F.relu(self.fc_2(h))
        return x

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################

        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size, hidden_size)
        self.fc_1 = nn.Linear(hidden_size, 64)
        self.fc_2 = nn.Linear(64, classes)

    def forward(self, x):
        batch_size = x.size()[1]
        h_seq, (h, _) = self.LSTM(x)
        h = F.dropout(F.relu(self.fc_1(h.reshape(batch_size, self.hidden_size))))
        h = F.relu(self.fc_2(h))
        return h

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
