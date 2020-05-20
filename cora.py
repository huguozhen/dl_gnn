import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GATConv

###############################################################################
# The forward function is essentially the same as any other commonly seen NNs
# model in PyTorch.  We can initialize GCN like any ``nn.Module``. For example,
# let's define a simple neural network consisting of two GCN layers. Suppose we
# are training the classifier for the cora dataset (the input feature size is
# 1433 and the number of classes is 7). The last GCN layer computes node embeddings,
# so the last layer in general does not apply activation.

# multihead attention 的融合, 输入 k*m, 输出 1*m
class MHI(nn.Module):
    def __init__(self, k, m):
        super(MHI, self).__init__()

        self.layer = nn.Linear(m, m)

        self.k = k
        self.m = m

        self.a = nn.Parameter(th.Tensor(2*m, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()
        nn.init.xavier_normal_(self.a)

    # 输入为 k*m
    def forward(self, x):
        feat = x
        x = self.layer(x)
        e = th.Tensor(self.k, 1)

        x_mean = th.mean(x, 0)
        for i in range(self.k):
          e[i] = self.a.t() @ th.cat((x[i,:], x_mean), 0)

        e = F.relu(e)
        alpha = F.softmax(e, 0)

        return th.sum(feat*alpha,0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GATConv(1433, 8, 8, feat_drop=0.6)
        self.layer2 = MHI(8, 8)
        self.layer3 = GATConv(8, 7, 1, feat_drop=0.6)

    def forward(self, g, features):
        x = self.layer1(g, features)
        # x = self.layer2(x)
        x = F.relu(x)
        
        h = th.Tensor(x.size(0), x.size(2))
        for i in range(x.size(0)):
          h[i] = self.layer2(x[i])
        # print(h.size())

        h = self.layer3(g, h)
        h = h.squeeze(1)
        return h
net = Net()
print(net)

###############################################################################
# We load the cora dataset using DGL's built-in data module.

from dgl.data import citation_graph as citegrh
import networkx as nx
def load_cora_data():
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

import time
import numpy as np
g, features, labels, train_mask, test_mask = load_cora_data()
optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
dur = []
for epoch in range(100):

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