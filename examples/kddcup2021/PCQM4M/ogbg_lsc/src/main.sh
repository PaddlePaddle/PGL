cd "$(dirname "$0")"

cd ../features
python mol_tree.py         # takes about 30 min

cd ../src
. ./cross_run.sh 0 1        # training on the whole dataset will take about 10 days

cd ../outputs
rsync -av * ../ensemble/model_pred/new_run

cd ../ensemble
python ensemble.py
