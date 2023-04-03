import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvRNN(nn.Module):
    def __init__(self, input_dim, timesteps, output_dim, kernel_size1=9, kernel_size2=7, kernel_size3=5, n_channels1=32, n_channels2=32, n_channels3=32, n_units1=32, n_units2=32, n_units3=32):
        super().__init__()

        # Initialize pooling layers
        self.avg_pool1 = nn.AvgPool1d(4, 4)
        self.avg_pool2 = nn.AvgPool1d(8, 8)

        # Initialize convolution layers
        self.conv1 = nn.Conv1d(input_dim, n_channels1, kernel_size=kernel_size1)
        self.conv2 = nn.Conv1d(input_dim, n_channels2, kernel_size=kernel_size2)
        self.conv3 = nn.Conv1d(input_dim, n_channels3, kernel_size=kernel_size3)

        # Initialize GRU layers
        self.gru1 = nn.GRU(n_channels1, n_units1, batch_first=True)
        self.gru2 = nn.GRU(n_channels2, n_units2, batch_first=True)
        self.gru3 = nn.GRU(n_channels3, n_units3, batch_first=True)

        # Initialize linear layers
        self.linear1 = nn.Linear(n_units1 + n_units2 + n_units3, output_dim)
        self.linear2 = nn.Linear(input_dim * timesteps, output_dim)

        # Initialize padding layers
        self.pad1 = nn.ConstantPad1d(((kernel_size1 - 1), 0), 0)
        self.pad2 = nn.ConstantPad1d(((kernel_size2 - 1), 0), 0)
        self.pad3 = nn.ConstantPad1d(((kernel_size3 - 1), 0), 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        # Path 1: Padding -> Conv1 -> ReLU -> GRU1
        path1 = self.pad1(x)
        path1 = torch.relu(self.conv1(path1))
        path1 = path1.permute(0, 2, 1)
        _, hidden1 = self.gru1(path1)

        # Path 2: AvgPool1 -> Padding -> Conv2 -> ReLU -> GRU2
        path2 = self.avg_pool1(x)
        path2 = self.pad2(path2)
        path2 = torch.relu(self.conv2(path2))
        path2 = path2.permute(0, 2, 1)
        _, hidden2 = self.gru2(path2)

        # Path 3: AvgPool2 -> Padding -> Conv3 -> ReLU -> GRU3
        path3 = self.avg_pool2(x)
        path3 = self.pad3(path3)
        path3 = torch.relu(self.conv3(path3))
        path3 = path3.permute(0, 2, 1)
        _, hidden3 = self.gru3(path3)

        # Concatenate hidden states and pass through linear layer
        hidden_concat = torch.cat([hidden1[-1], hidden2[-1], hidden3[-1]], dim=1)
        output = self.linear1(hidden_concat)

        return output



