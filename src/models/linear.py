# import torch
import torch.nn as nn

class FCModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, select, name=None):
        super(FCModel, self).__init__()
        self.model_name = name if name else self.__class__.__name__
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
        )
        if select == "both":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x
    
