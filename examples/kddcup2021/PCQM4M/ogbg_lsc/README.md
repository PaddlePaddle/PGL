# LiteGEM: Solution to KDDCUP 2021 PCQM4M-LSC


## Installation Requirements

```
rdkit==2019.03.1
ogb==1.3.0
paddle==2.0.0
pgl==2.1.4
```

## Data Preprocessing

To preprocess the data, run the following commands:

```
cd ./features
python mol_tree.py --data_path ../dataset
```

The processed data will be saved in `../dataset/processed_data`

## Training

All the configuration can be found in `./src/config.yaml` file.

To train the model using single GPU, run the following commands:

```
cd ./src
export CUDA_VISIBLE_DEVICES=0
python main.py --config config.yaml
```

The training log will be saved in `../logs/task_name/` and  
the checkpoint will be saved in `../checkpoints/task_name`, 
where `task_name` can be found in `config.yaml` file.


Users can also train the model using multi-GPU:

```
cd ./src
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fleetrun --log_dir ../fleet_logs/task_name main.py --config config.yaml
```

## Testing

After training the model, open the `./src/config.yaml` file, 
write down the model saved path for `infer_from` hyper-parameter,
and then run the following commands:

```
cd ./src

python test.py --config config.yaml --output_path ./test_result
```

The test result will be saved in `./src/test_result`.
