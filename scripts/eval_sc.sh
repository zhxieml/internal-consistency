for model in "llama2-7b" "llama2-13b" "mixtral-7b" "mixtral-8x7b"; do
    for dataset in "coinflip" "boolq" "prontoqa" "proofwriter"; do
        echo "Running for model: ${model}, dataset: ${dataset}"
        python eval_sc.py \
        --dataset ${dataset} \
        --model ${model}
    done
done