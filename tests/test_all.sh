python -m unittest test_bigraph.py
python -m unittest test_conv.py
python -m unittest test_dataloader.py
python -m unittest test_graph.py
python -m unittest test_graph_op.py
python -m unittest test_hetergraph.py
python -m unittest test_math.py
python -m unittest test_partition.py
python -m unittest test_sample.py
python -m unittest test_static_graph.py

# Distributed
python -m unittest test_dist_cpu_graph.py
CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch test_dist_graph.py
