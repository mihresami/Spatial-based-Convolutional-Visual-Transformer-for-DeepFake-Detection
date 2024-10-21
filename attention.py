
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        # self.q = nn.ModuleList([nn.Linear(layer,layer) for _, layer in enumerate(self.layers)])
        # self.k = nn.ModuleList([nn.Linear(layer,layer) for _, layer in enumerate(layers)])
        # self.v = nn.ModuleList([nn.Linear(layer,layer) for _, layer in enumerate(layers)])

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class SelfAttention2(nn.Module):
    def __init__(self, input_dim,layers:dict):
        super(SelfAttention, self).__init__()
        self.layers = layers
        self.input_dim = input_dim
        self.softmax = nn.Softmax(dim=-1)
        self.q = nn.ModuleList([nn.Linear(layer,layer) for _, layer in enumerate(self.layers)])
        self.k = nn.ModuleList([nn.Linear(layer,layer) for _, layer in enumerate(layers)])
        self.v = nn.ModuleList([nn.Linear(layer,layer) for _, layer in enumerate(layers)])

    def forward(self, x):
        queries = self.q(x)
        keys = self.k(x)
        values = self.v(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted
     

# x = torch.randn(2,4,20)
# layers = [10,20]
# model = SelfAttention(layers)
# out = model(x)