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

from logger import Logger

# from radam import RAdam


class CoNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, f_drop=0):
        super(CoNet, self).__init__()

        self.layer1 = SAGEConv(
            in_channels, out_channels, 'mean', feat_drop=f_drop)
        self.layer2 = SAGEConv(
            in_channels, out_channels, 'pool', feat_drop=f_drop)
        self.layer2 = SAGEConv(
            in_channels, out_channels, 'gcn', feat_drop=f_drop)
        # self.layer4 = GATConv(
        #     in_channels, out_channels, 1, feat_drop=f_drop)
        # self.layer5 = GraphConv(
        #     in_channels, out_channels)

        self.w = Parameter(torch.tensor([0, 0, 0], dtype=torch.float))

    def reset_parameters(self):

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()
        # self.layer4.reset_parameters()
        # self.layer5.reset_parameters()

        init.uniform_(self.w)

    def forward(self, g, x):

        x1 = self.layer1(g, x)
        x2 = self.layer2(g, x)
        x3 = self.layer3(g, x)
        # x4 = self.layer4(g, x)
        # x4 = x4.squeeze(1)
        # x5 = self.layer5(g, x)

        weights = F.softmax(self.w, dim=0)

        return weights[0] * x1 + weights[1] * x2 + weights[2] * x3


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.layer1 = CoNet(in_channels, hidden_channels)
        self.layer2 = torch.nn.BatchNorm1d(hidden_channels)
        self.layer3 = CoNet(hidden_channels, hidden_channels, f_drop=0.5)
        self.layer4 = torch.nn.BatchNorm1d(hidden_channels)
        self.layer5 = CoNet(hidden_channels, out_channels, f_drop=0.5)

    def reset_parameters(self):

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()
        self.layer4.reset_parameters()
        self.layer5.reset_parameters()

    def forward(self, g, x):

        x = self.layer1(g, x)
        # x = x.view(x.size(0), 1, -1).squeeze(1)
        x = self.layer2(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.layer3(g, x)
        # x = torch.mean(x, 1)
        x = self.layer4(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.layer5(g, x)
        # x = torch.mean(x, 1)

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
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = DglNodePropPredDataset(name='ogbn-arxiv')

    split_idx = dataset.get_idx_split()
    # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    graph, label = dataset[0]
    # data = dataset[0].to(device)

    g = dgl.DGLGraph((graph.edges()[0], graph.edges()[1]))
    g.add_edges(graph.edges()[1], graph.edges()[0])
    # g.ndata['feat'] = graph.ndata['feat']
    # print(g.edges()[1].size())

    x = graph.ndata['feat'].to(device)
    y_true = label.to(device)

    train_idx = split_idx['train'].to(device)

    model = GCN(x.size(-1), args.hidden_channels, dataset.num_classes,
                args.num_layers, args.dropout).to(device)
    print(model)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, g, x, y_true, train_idx, optimizer)
            result = test(model, g, x, y_true, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
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
