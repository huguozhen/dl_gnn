import argparse

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
    def __init__(self, in_channels, out_channels, model):
        super(CoNet, self).__init__()

        if model == 'AFFN':
            self.layer1 = SAGEConv(
                in_channels, out_channels, 'mean')
            self.layer2 = GraphConv(in_channels, out_channels)
            self.layer3 = GATConv(in_channels, out_channels, 1)
        elif model == 'GCN':
            self.layer1 = GraphConv(in_channels, out_channels)
            self.layer2 = GraphConv(in_channels, out_channels)
            self.layer3 = GraphConv(in_channels, out_channels)
        elif model == 'SAGE':
            self.layer1 = SAGEConv(in_channels, out_channels, 'mean')
            self.layer2 = SAGEConv(in_channels, out_channels, 'mean')
            self.layer3 = SAGEConv(in_channels, out_channels, 'mean')
        else:
            self.layer1 = GATConv(in_channels, out_channels, 1)
            self.layer2 = GATConv(in_channels, out_channels, 1)
            self.layer3 = GATConv(in_channels, out_channels, 1)

        self.w = nn.Parameter(th.tensor([1, 1, 1], dtype=th.float))
        self.model = model

    def reset_parameters(self):

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()

        init.uniform_(self.w)

    def forward(self, g, x):

        x1 = self.layer1(g, x)
        x2 = self.layer2(g, x)
        x3 = self.layer3(g, x)
        if self.model == 'AFFN':
            x3 = x3.squeeze(1)
        elif self.model == 'GAT':
            x1 = x1.squeeze(1)
            x2 = x2.squeeze(1)
            x3 = x3.squeeze(1)

        weights = self.w / th.sum(self.w, 0)

        return weights[0] * x1 + weights[1] * x2 + weights[2] * x3


class Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model, dropout):
        super(Net, self).__init__()

        self.layer1 = CoNet(in_channels, hidden_channels, model)
        self.layer2 = th.nn.BatchNorm1d(hidden_channels)
        self.layer3 = CoNet(hidden_channels, out_channels, model)
        self.dropout = dropout

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
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer3(g, x)
        # x = x.squeeze(1)
        x = F.log_softmax(x, 1)

        return x


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def load_data(dataset):
    if dataset == 'cora':
        data = citegrh.load_cora()
    elif dataset == 'pubmed':
        data = citegrh.load_pubmed()
    else:
        data = citegrh.load_citeseer()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    num_labels = data.num_labels
    split1 = int(0.7*len(labels))
    split2 = int(0.9*len(labels))
    train_mask = th.BoolTensor(_sample_mask(range(split1), labels.shape[0]))
    val_mask = th.BoolTensor(_sample_mask(
        range(split1, split2), labels.shape[0]))
    test_mask = th.BoolTensor(_sample_mask(
        range(split2, labels.shape[0]-1), labels.shape[0]))
    g = DGLGraph(data.graph)
    print("Total size: {:}| Feature dims: {:}| Train size: {:}| Val size: {:}| Test size: {:}| Num of labels: {:}".format(
        features.size(0), features.size(1), len(labels[train_mask]), len(labels[val_mask]), len(labels[test_mask]), num_labels))
    return g, features, labels, num_labels, train_mask, val_mask, test_mask


def evaluate(model, g, features, labels, train_mask, val_mask, test_mask):
    model.eval()
    with th.no_grad():
        res = model(g, features)
        _, indices = th.max(res, dim=1)
        isEqual = (indices == labels)
        train_acc = th.sum(isEqual[train_mask]).item() * \
            1.0 / len(labels[train_mask])
        val_acc = th.sum(isEqual[val_mask]).item() * \
            1.0 / len(labels[val_mask])
        test_acc = th.sum(isEqual[test_mask]).item() * \
            1.0 / len(labels[test_mask])

        return train_acc, val_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='Cite Graph')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--model', type=str, default='AFFN')
    args = parser.parse_known_args()[0]
    print(args)

    g, features, labels, num_labels, train_mask, val_mask, test_mask = load_data(
        args.dataset)

    device = f'cuda:{args.device}' if th.cuda.is_available() else 'cpu'
    device = th.device(device)

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    net = Net(features.size(1), args.hidden_channels,
              num_labels, args.model, args.dropout).to(device)
    print(net)

    optimizer = th.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=args.wd)
    final_train, final_val, final_test = 0, 0, 0
    for epoch in range(args.epochs):
        net.train()
        res = net(g, features)
        loss = F.nll_loss(res[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, val_acc, test_acc = evaluate(
            net, g, features, labels, train_mask, val_mask, test_mask)
        if val_acc > final_val:
            final_train, final_val, final_test = train_acc, val_acc, test_acc
        print("Epoch {:05d} | Loss {:.4f} | Train Acc {:.4f}| Val Acc {:.4f}| Test Acc {:.4f}".format(
            epoch, loss.item(), train_acc, val_acc, test_acc))
    print("Final: Train Acc {:.4f}| Val Acc {:.4f}| Test Acc {:.4f}".format(
        train_acc, val_acc, test_acc))


if __name__ == "__main__":
    main()
