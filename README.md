## Implementation of adaptive feature fusion network (AFFN)

### Content

This repository includes the following three python scripts:

* `run.py` : AFFN test code for the following datasets: cora, citeseer, pubmed, amazon co-purchase, coauthor.
* `run_arxiv.py` : AFFN test code for the ogbn-arxiv dataset.
* `logger.py` : Utils for logging outputs.

### Requirements

- Python >= 3.5.0

- Pyorch >= 1.5.0
- DGL >= 0.4.0. To install DGL, run `pip install dgl` .
- For CUDA builds, require CUDA >= 9.0. To install DGL with CUDA, run `pip install dgl-${CUDA}` ,  replace `${CUDA}` with your CUDA version, i.e. `cu90` , `cu92` , `cu100` or `cu101` 

### Training & Evaluation

```python
# Run with default config
python run.py
python run_arxiv.py

# Run with custom config
python run.py --runs=5 --epochs=200 --hidden_channels=256 --dataset=amazon-computers --model=AFFN
python run_arxiv.py --hidden_channels=256 --dropout=0.65 --lr=1e-3 --wd=5e-4 --model=GCN
```

##### Args list:

|        args         |                             type                             |                meaning                 | default |
| :-----------------: | :----------------------------------------------------------: | :------------------------------------: | :-----: |
|     `--device`      |                            `int`                             |           GPU device number            |   `0`   |
| `--hidden_channels` |                            `int`                             |    Number of nodes in hidden layers    |  `256`  |
|     `--dropout`     |                           `float`                            |     Dropout rate in dropout layer      |  `0.5`  |
|       `--lr`        |                           `float`                            |             Learning rate              | `0.01`  |
|       `--wd`        |                           `float`                            |    _L2_ regularization coefficient     |   `0`   |
|     `--epochs`      |                            `int`                             |      Number of epochs in each run      |  `500`  |
|      `--runs`       |                            `int`                             | Number of runs with current parameters |  `10`   |
|    `--dataset` *    | `string: {cora, pubmed, citeseer, amazon-computers, amazon-photo, coauthor-cs, coauthor-physics}` |          Which dataset to use          | `cora`  |
|      `--model`      |               `string: {AFFN, GCN, SAGE, GAT}`               |           Which model to use           | `AFFN`  |

*Script `run_arxiv.py` has **NO** arg `--dataset` .

**Note**: when running for the first time, it will take sometime to download the corresponding dataset automatically.