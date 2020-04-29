# ERNIESage in PGL

## Introduction
In many industrial applications, there is often a special graph shown below: Text Graph. As the name implies, the node attributes of a graph consist of text, and the construction of edges provides structural information. For example, in Text Graph in search scenarios, nodes can be expressed by search terms, web page titles, and web page text, and user feedback and hyperlink information can form edge relationships.

<img src="./docs/source/_static/text_graph.png" alt="Text Graph" width="800">

**ERNIESage**, a model proposed by the PGL team, effectively improves the performance on Text Graph by simultaneously modeling text semantics and graph structure information. [ERNIE](https://github.com/PaddlePaddle/ERNIE) is a continual pre-training framework for language understanding launched by Baidu. It surpasses 16 tasks in Chinese and English.
When ERNIE meets Graph, it comes to the birth of ErnieSAGE(abbreviation of ERNIE SAmple aggreGatE). Its structure is shown in the figure below. The main idea is to use ERNIE as an aggregation function (Aggregators) to model the semantic and structural relationship between its own nodes and neighbor nodes. In addition, for the position-independent characteristics of neighbor nodes, Attention Mask and independent Position Embedding mechanism for neighbor blindness are designed.

<img src="./docs/source/_static/ernie_aggregator.png" alt="ERNIESage" width="800">

Thanks to the flexibility and usability of PGL, ERNIESage can be quickly implemented under PGL's Message Passing paradigm. 

## Dependencies
- paddlepaddle>=1.7
- pgl>=1.1

## Dataformat

## How to run

We adopt [PaddlePaddle Fleet](https://github.com/PaddlePaddle/Fleet) as our distributed training frameworks ```config/*.yaml``` are some example config files for hyperparameters.

```sh
# train ERNIESage in distributed gpu mode.
sh local_run.sh config/erniesage_v2_gpu.yaml

# train ERNIESage in distributed cpu mode.
sh local_run.sh config/erniesage_v2_cpu.yaml
```

## Hyperparamters
- learner_type: `gpu` or `cpu`; gpu use fleet Collective mode, cpu use fleet Transpiler mode.


