# GNNAutoScale: Auto-Scaling GNNs in PaddlePaddle

Refer to [PyGAS](https://github.com/rusty1s/pyg_autoscale), we implement a similar GNNAutoScale framework, which can scale arbitrary message-passing GNNs to large graphs.

Following PyGAS, we use history embedding to store the embeddings from prior training iterations, which can lead to smaller GPU memory consumption.

## Requirements

- paddlepaddle-gpu==develop
- pgl

## Models

- GCN
- GAT
- APPNP
- GCNII

## Commands


## Results

### Citation Network


### Reddit Dataset

<table>
   <tr>
      <td>Accuracy</td>
      <td>GNNAutoScale</td>
      <td></td>
      <td>PyGAS</td>
   </tr>
   <tr>
      <td>partition</td>
      <td>Metis</td>
      <td>Random </td>
      <td>Metis</td>
   </tr>
   <tr>
      <td>GCN</td>
      <td>95.27</td>
      <td>95.21</td>
      <td>95.45</td>
   </tr>
   <tr>
      <td>GCNII</td>
      <td>96.72</td>
      <td>96.71</td>
      <td>96.77</td>
   </tr>
</table>
