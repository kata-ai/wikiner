#!/bin/bash

input_file=$1
input_gold=$2
output_dir=$3
base_model_dir=$4
model_name=$5
tokenizer=$6

filename=$(basename -- "$input_gold")
extension="${filename##*.}"
filename="${filename%.*}"
input_name="${filename%.*}"
echo $input_name

model_dirname="$base_model_dir/$model_name"
echo $model_dirname

output_name="$model_name.$input_name.pred.conll"
output_file="$output_dir/$output_name"
echo $output_file

[ -f $output_file ] && rm $output_file

# python run_allennlp.py conll-output \
#     $model_dirname/model.tar.gz $input_file \
#     --output-file $output_file \
#     --batch-size 32 --predictor sentence-tagger

if [ -z "$tokenizer" ]
then
    echo "\$tokenizer is empty"
    python run_allennlp.py conll-output \
    $model_dirname/model.tar.gz $input_file \
    --output-file $output_file \
    --batch-size 32 --predictor sentence-tagger
else
    echo "\$tokenizer is NOT empty"
    python run_allennlp.py conll-output \
    $model_dirname/model.tar.gz $input_file \
    --output-file $output_file \
    --batch-size 32 --predictor sentence-tagger \
    --tokenizer $tokenizer
fi

python eval_output.py test with test_corpus=$input_gold \
    model_output=$output_file col_ref=1 col_hyp=1

# rm $output_file
