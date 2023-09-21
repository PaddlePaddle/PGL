mkdir data
cd data

# FB15k-237
wget https://github.com/TimDettmers/ConvE/raw/master/FB15k-237.tar.gz;mkdir FB15k-237;tar xvf FB15k-237.tar.gz -C FB15k-237;rm FB15k-237.tar.gz

# WN18RR
wget https://github.com/TimDettmers/ConvE/raw/master/WN18RR.tar.gz;mkdir WN18RR;tar xvf WN18RR.tar.gz -C WN18RR;rm WN18RR.tar.gz

# FB15k
wget https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz
mkdir FB15k;mv fetch.php\?media\=en\:fb15k.tgz* fb15k.tgz;tar xvf fb15k.tgz -C FB15k;rm fb15k.tgz
mv FB15k/FB15k/freebase_mtr100_mte100-train.txt FB15k/train.txt
mv FB15k/FB15k/freebase_mtr100_mte100-valid.txt FB15k/valid.txt
mv FB15k/FB15k/freebase_mtr100_mte100-test.txt FB15k/test.txt

# WN18
wget https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz;
mkdir wn18;mv fetch.php\?media\=en\:wordnet-mlj12.tar.gz* wn18.tgz;tar xvf wn18.tgz -C wn18;rm wn18.tgz
mv wn18/wordnet-mlj12/wordnet-mlj12-train.txt wn18/train.txt
mv wn18/wordnet-mlj12/wordnet-mlj12-valid.txt wn18/valid.txt
mv wn18/wordnet-mlj12/wordnet-mlj12-test.txt wn18/test.txt
