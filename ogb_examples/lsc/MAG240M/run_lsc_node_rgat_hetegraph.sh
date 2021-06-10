## training
CUDA_VISIBLE_DEVICES=0,1,2,3  python -m paddle.distributed.launch --log_dir ./output/rgat_output/ lsc_node_hetegraph.py --conf configs/rgat.yaml  
