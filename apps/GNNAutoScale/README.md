# GNNAutoScale: Auto-Scaling GNNs in PaddlePaddle

Refer to paper [*GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings*](https://arxiv.org/abs/2106.05609), we reproduce the GNNAutoScale framework using PGL, which can scale arbitrary message-passing GNNs to large graphs.

By using history embeddings on CPU to store updated node-embeddings from prior training iterations, and pulling neighboring node-embeddings in history embeddings to participate in training, we can have smaller GPU memory consumption.

## Requirements

- paddlepaddle-gpu>=2.2.0
- pgl

## Models

- GCN
- GAT
- APPNP
- GCNII

## Commands

```shell
cd examples/

# GCN
python run.py --conf config/cora/gcn.yaml
python run.py --conf config/pubmed/gcn.yaml
python run.py --conf config/citeseer/gcn.yaml
python run.py --conf config/reddit/gcn.yaml
python run.py --conf config/arxiv/gcn.yaml

# GAT
python run.py --conf config/cora/gat.yaml
python run.py --conf config/pubmed/gat.yaml
python run.py --conf config/citeseer/gat.yaml

# APPNP
python run.py --conf config/cora/appnp.yaml
python run.py --conf config/pubmed/appnp.yaml
python run.py --conf config/citeseer/appnp.yaml

# GCNII
python run.py --conf config/cora/gcnii.yaml
python run.py --conf config/pubmed/gcnii.yaml
python run.py --conf config/citeseer/gcnii.yaml
python run.py --conf config/reddit/gcnii.yaml
python run.py --conf config/arxiv/gcnii.yaml
```

## Results

### Citation Network
<table>
   <tr align="center">
      <th></th>
      <th colspan="4" align="center">Cora</th>
   </tr>
   <tr align="center">
      <td>Accuracy</td>
      <td colspan="2" align="center">GNNAutoScale</td>
      <td>PyGAS</td>
      <td>PGL</td>
   </tr>
   <tr align="center">
      <td>Partition Method</td>
      <td>Metis</td>
      <td>Random </td>
      <td>Metis</td>
      <td>-</td>
   </tr>
   <tr align="center">
      <td>GCN</td>
      <td>81.4(1.15)</td>
      <td>78.12(1.28)</td>
      <td>82.29(0.76)</td>
      <td>80.7(1.0)</td>
   </tr>
   <tr align="center">
      <td>GAT</td>
      <td>82.93(1.27)</td>
      <td>82.79(0.99)</td>
      <td>83.32(0.62)</td>
      <td>83.4(0.4)</td>
   </tr>
   <tr align="center">
      <td>APPNP</td>
      <td>83.94(0.55)</td>
      <td>79.62(1.69)	</td>
      <td>83.19(0.58)</td>
      <td>84.6(0.3)</td>
   </tr>
   <tr align="center">
      <td>GCNII</td>
      <td>83.91(0.54)</td>
      <td>81.36(1.26)</td>
      <td>85.52(0.39)</td>
      <td>84.6(0.3)</td>
   </tr>
</table>

<table>
   <tr align="center">
      <th></th>
      <th colspan="4" align="center">Citeseer</th>
   </tr>
   <tr align="center">
      <td>Accuracy</td>
      <td colspan="2" align="center">GNNAutoScale</td>
      <td>PyGAS</td>
      <td>PGL</td>
   </tr>
   <tr align="center">
      <td>Partition Method</td>
      <td>Metis</td>
      <td>Random </td>
      <td>Metis</td>
      <td>-</td>
   </tr>
   <tr align="center">
      <td>GCN</td>
      <td>70.86(0.93)</td>
      <td>71.05(0.98)</td>
      <td>71.18(0.97)</td>
      <td>71.0(0.7)</td>
   </tr>
   <tr align="center">
      <td>GAT</td>
      <td>71.56(0.9)</td>
      <td>71.5(0.65)</td>
      <td>71.86(1.00)</td>
      <td>70.0(0.6)</td>
   </tr>
   <tr align="center">
      <td>APPNP</td>
      <td>71.58(0.53)</td>
      <td>70.35(1.3)</td>
      <td>72.63(0.82)</td>
      <td>71.9(0.3)</td>
   </tr>
   <tr align="center">
      <td>GCNII</td>
      <td>70.06(1.11)</td>
      <td>70.96(1.08)</td>
      <td>73.89(0.48)</td>
      <td>72.4(0.6)</td>
   </tr>
</table>

<table>
   <tr align="center">
      <th></th>
      <th colspan="4" align="center">Pubmed</th>
   </tr>
   <tr align="center">
      <td>Accuracy</td>
      <td colspan="2" align="center">GNNAutoScale</td>
      <td>PyGAS</td>
      <td>PGL</td>
   </tr>
   <tr align="center">
      <td>Partition Method</td>
      <td>Metis</td>
      <td>Random</td>
      <td>Metis</td>
      <td>-</td>
   </tr>
   <tr align="center">
      <td>GCN</td>
      <td>79.13(0.59)</td>
      <td>78.47(0.44)</td>
      <td>79.23(0.62)</td>
      <td>79.4(0.3)</td>
   </tr>
   <tr align="center">
      <td>GAT</td>
      <td>77.76(0.45)</td>
      <td>77.4(0.56)</td>
      <td>78.42(0.56)</td>
      <td>77.2(0.4)</td>
   </tr>
   <tr align="center">
      <td>APPNP</td>
      <td>80.4(0.28)</td>
      <td>78.96(0.58)</td>
      <td>79.82(0.52)</td>
      <td>80.3(0.2)</td>
   </tr>
   <tr align="center">
      <td>GCNII</td>
      <td>79.93(0.48)</td>
      <td>78.51(0.49)</td>
      <td>80.19(0.49)</td>
      <td>79.8(0.3)</td>
   </tr>
</table>

### Reddit Dataset

<table>
   <tr align="center">
      <th>Accuracy</th>
      <th colspan="2" align="center">GNNAutoScale</th>
      <th>PyGAS</th>
   </tr>
   <tr align="center">
      <td>Partition Method</td>
      <td>Metis</td>
      <td>Random </td>
      <td>Metis</td>
   </tr>
   <tr align="center">
      <td>GCN</td>
      <td>95.27</td>
      <td>95.21</td>
      <td>95.45</td>
   </tr>
   <tr align="center">
      <td>GCNII</td>
      <td>96.72</td>
      <td>96.71</td>
      <td>96.77</td>
   </tr>
</table>

### Ogbn-arxiv Dataset

<table>
   <tr align="center">
      <th>Accuracy</th>
      <th colspan="2" align="center">GNNAutoScale</th>
      <th>PyGAS</th>
   </tr>
   <tr align="center">
      <td>Partition Method</td>
      <td>Metis</td>
      <td>Random </td>
      <td>Metis</td>
   </tr>
   <tr align="center">
      <td>GCN</td>
      <td>71.8</td>
      <td>70.7</td>
      <td>71.68</td>
   </tr>
   <tr align="center">
      <td>GCNII</td>
      <td>72.4</td>
      <td>71.5</td>
      <td>73.0</td>
   </tr>
</table>

## Coming new features

- Multi-GPUs support.
- Large dataset ogbn-papers100m support.

## Reference

```
@inproceedings{Fey/etal/2021,
  title={{GNNAutoScale}: Scalable and Expressive Graph Neural Networks via Historical Embeddings},
  author={Fey, M. and Lenssen, J. E. and Weichert, F. and Leskovec, J.},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2021},
}

Github link: https://github.com/rusty1s/pyg_autoscale

```
