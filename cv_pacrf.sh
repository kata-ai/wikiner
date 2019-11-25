#!/bin/bash

data_dir=$1
dataset=$2
pacrf=$3
fold=$4
gold=${5:-$data_dir/goldstandard-0811.conll}
window=${6:-0}

echo "test using ${gold}"
# Prepare the data by extracting features from raw files into crfsuite input format

python partial_crfprep.py --window-size $window \
    $data_dir/$dataset/train-$fold.conll > temp/$dataset-train-$fold.crfsuite
python partial_crfprep.py --window-size $window \
    $data_dir/$dataset/test-$fold.conll > temp/$dataset-test-$fold.crfsuite

python partial_crfprep.py --window-size $window \
    $gold > temp/gold-$dataset.crfsuite

# Train partial-crfsuite (using pacrf variable) model

$pacrf learn -m temp/cv-crf-pa-$dataset-$fold.model \
    -a lbfgs -e 2 temp/$dataset-train-$fold.crfsuite \
    temp/$dataset-test-$fold.crfsuite

# Tag the gold dataset from trained model (model will not read true label)

$pacrf tag -m temp/cv-crf-pa-$dataset-$fold.model \
    temp/gold-$dataset.crfsuite > temp/cv-crf-pa-$dataset-$fold-pred.out

# Evaluate the tag from trained model to gold tag

python eval_output.py with test_corpus=$gold col_ref=1 \
    model_output=temp/cv-crf-pa-$dataset-$fold-pred.out col_hyp=0
