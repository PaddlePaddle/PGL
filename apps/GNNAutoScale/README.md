# GNNAutoScale: Auto-Scaling GNNs in PaddlePaddle

Refer to [PyGAS](https://github.com/rusty1s/pyg_autoscale), we implement a similar GNNAutoScale framework, which can scale arbitrary message-passing GNNs to large graphs.

Following PyGAS, we use history embedding to store the embeddings from prior training iterations, which can lead to smaller GPU memory consumption.

## Requirements

- paddlepaddle-gpu==2.2.0
- pgl==2.2

## Models

- GCN
- GAT
- To be continued...

## Commands & Results
