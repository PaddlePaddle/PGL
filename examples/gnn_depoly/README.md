# GNN Paddle Inference

## 节点分类python部署（无采样）

此处提供了共计三个模型，包括GCN、GAT、GraphSage，示例数据集采用Cora。

以GAT为例，模型训练导出和Python部署复现步骤如下：

``` python
# 进入``node_classification_with_full_graph``文件夹

python train.py --model GAT  # 通过此步可以得到动态图保存的模型 "gat.pdparam"

python export_model.py --model GAT  # 用paddle.jit.save保存模型，以供后续预测部署，模型位于export文件夹

python python_deploy.py --model GAT --model_dir export/  # Python部署预测示例

```

## 节点分类python部署（需采样）

以GraphSage模型为例，示例数据集采用Reddit。
其中，训练采用GPU高性能采样，推理采用PGL原有CPU采样函数。推理时的重点是获取子图信息，子图采样方式可根据需要自行修改。

``` python
# 进入``node_classification_with_sampling``文件夹

python train.py --model GraphSage  # 通过此步可以得到动态图保存的模型 "graphsage.pdparam" 

python export_model.py --model GraphSage  # 用paddle.jit.save保存模型，以供后续预测部署，模型位于export文件夹

python python_deploy.py --model GraphSage --model_dir export/  # Python部署预测示例

```
