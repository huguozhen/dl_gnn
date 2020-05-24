import argparse

import torch
from torch.nn import Parameter
from torch.nn import init
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GATConv

from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from logger import Logger  # logger.py  用于输出结果


class CoNet(torch.nn.Module):
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

        # 特征融合权向量
        self.w = Parameter(torch.tensor([1, 1, 1], dtype=torch.float))

    def reset_parameters(self):

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()

        init.uniform_(self.w)

    def forward(self, g, x):

        x1 = self.layer1(g, x)
        x2 = self.layer2(g, x)
        x3 = self.layer3(g, x)

        # 消除DGL内置的GAT使用多头注意力机制输出的多余维度
        if self.model == 'AFFN':
            x3 = x3.squeeze(1)
        elif self.model == 'GAT':
            x1 = x1.squeeze(1)
            x2 = x2.squeeze(1)
            x3 = x3.squeeze(1)

        # 权向量标准化
        weights = self.w / torch.sum(self.w, 0)

        return weights[0] * x1 + weights[1] * x2 + weights[2] * x3


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model, dropout):
        super(Net, self).__init__()

        self.layer1 = CoNet(in_channels, hidden_channels)
        self.layer2 = torch.nn.BatchNorm1d(hidden_channels)
        self.layer3 = CoNet(hidden_channels, out_channels)

        self.dropout = dropout

    def reset_parameters(self):

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()

    def forward(self, g, x):

        x = self.layer1(g, x)
        x = self.layer2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer3(g, x)

        return x.log_softmax(dim=-1)


def train(model, g, x, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(g, x)[train_idx]
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, g, x, y_true, split_idx, evaluator):
    model.eval()

    out = model(g, x)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (Full-Batch)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--model', type=str, default='AFFN')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = DglNodePropPredDataset(name='ogbn-arxiv')

    split_idx = dataset.get_idx_split()
    graph, label = dataset[0]

    # 有向图转无向图，添加反向边
    g = dgl.DGLGraph((graph.edges()[0], graph.edges()[1]))
    g.add_edges(graph.edges()[1], graph.edges()[0])

    x = graph.ndata['feat'].to(device)
    y_true = label.to(device)

    train_idx = split_idx['train'].to(device)

    model = Net(x.size(-1), args.hidden_channels, dataset.num_classes,
                args.model, args.dropout).to(device)
    print(model)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, g, x, y_true, train_idx, optimizer)
            result = test(model, g, x, y_true, split_idx, evaluator)
            logger.add_result(run, result)

            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
