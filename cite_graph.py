import numpy as np
from dgl.data import citation_graph as citegrh
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

        weights = self.w / th.sum(self.w, 0)

        return weights[0] * x1 + weights[1] * x2 + weights[2] * x3


class Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Net, self).__init__()

        self.layer1 = CoNet(in_channels, hidden_channels)
        self.layer2 = th.nn.BatchNorm1d(hidden_channels)
        self.layer3 = CoNet(hidden_channels, out_channels, f_drop=0)

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
        x = F.log_softmax(x, 1)

        return x


def load_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, train_mask, test_mask


def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        res = model(g, features)
        res = res[mask]
        labels = labels[mask]
        _, indices = th.max(res, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


g, features, labels, train_mask, test_mask = load_data()

device = f'cuda:{0}' if th.cuda.is_available() else 'cpu'
device = th.device(device)

features = features.to(device)
labels = labels.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)

net = Net(features.size(1), 128, 7).to(device)
print(net)
optimizer = th.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
for epoch in range(500):

    net.train()
    res = net(g, features)
    loss = F.nll_loss(res[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = evaluate(net, g, features, labels, test_mask)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f}".format(
        epoch, loss.item(), acc))
