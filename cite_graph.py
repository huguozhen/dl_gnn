import time
from dgl.data import citation_graph as citegrh
import numpy as np
import networkx as nx
import dgl
import dgl.function as fn

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GATConv


class CoNet(th.nn.Module):
    def __init__(self, in_channels, out_channels, f_drop=0):
        super(CoNet, self).__init__()

        self.layer1 = SAGEConv(
            in_channels, out_channels, 'mean', feat_drop=f_drop)
        self.layer2 = GraphConv(in_channels, out_channels)
        self.layer3 = GATConv(in_channels, out_channels, 1, feat_drop=f_drop)

        self.w = nn.Parameter(th.tensor([1, 1, 1], dtype=th.float))

    def reset_parameters(self):

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()

        init.uniform_(self.w)

    def forward(self, g, x):

        x1 = self.layer1(g, x)
        x2 = self.layer2(g, x)
        x3 = self.layer3(g, x)
        x3 = x3.squeeze(1)

        # weights = self.w / th.sum(self.w, 0)
        weights = F.softmax(F.leaky_relu(self.w), 0)

        return weights[0] * x1 + weights[1] * x2 + weights[2] * x3


class Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Net, self).__init__()

        self.layer1 = CoNet(in_channels, hidden_channels)
        self.layer2 = th.nn.BatchNorm1d(hidden_channels)
        self.layer3 = CoNet(hidden_channels, out_channels, f_drop=0)
        # self.layer1 = GATConv(in_channels, hidden_channels, 1)
        # self.layer2 = th.nn.BatchNorm1d(hidden_channels)
        # self.layer3 = GATConv(hidden_channels, out_channels, 1)

    def reset_parameters(self):

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()

    def forward(self, g, x):

        x = self.layer1(g, x)
        # x = x.squeeze(1)
        # x = x.view(x.size(0), 1, -1).squeeze(1)
        x = self.layer2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layer3(g, x)
        # x = x.squeeze(1)

        return x

###############################################################################
# We load the cora dataset using DGL's built-in data module.


def load_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, train_mask, test_mask

###############################################################################
# When a model is trained, we can use the following method to evaluate
# the performance of the model on the test dataset:


def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

###############################################################################
# We then train the network as follows:


g, features, labels, train_mask, test_mask = load_data()
net = Net(features.size(1), 128, 7)
print(net)
optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
dur = []
for epoch in range(200):

    net.train()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = evaluate(net, g, features, labels, test_mask)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f}".format(
        epoch, loss.item(), acc))
