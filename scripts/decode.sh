declare -A data_map=(
    ["boolq"]="data/boolq.jsonl"
    ["coinflip"]="data/coin_flip.json"
    ["prontoqa"]="data/prontoqa.json"
    ["proofwriter"]="data/proofwriter.json"
)
declare -A bs_map=(
    ["mixtral-7b"]="12"
    ["mixtral-8x7b"]="1"
    ["llama2-7b"]="12"
    ["llama2-13b"]="4"
)
base_out_dir=res

tensor_parallel=2
for dataset in "coinflip" "boolq" "prontoqa" "proofwriter"; do
    for seed in 111 222; do
        for model in "llama2-7b" "llama2-13b" "mixtral-7b" "mixtral-8x7b"; do
            echo "model: ${model}, dataset: ${dataset}, seed: ${seed}"
            out_dir=${base_out_dir}/${dataset}-${model}
            mkdir -p $out_dir

            # decode parameters
            temperature=0.7
            top_p=0.95
            run_prefix=temp${temperature}-topp${top_p}-seed${seed}

            output_prefix=${out_dir}/${run_prefix}
            echo "output_prefix: ${output_prefix}"

            python decode.py \
            --dataset ${dataset}:${data_map[${dataset}]} \
            --model $model \
            --batch_size 20 \
            --num_return_sequences 20 \
            --tensor_parallel $tensor_parallel \
            --output_prefix ${output_prefix} \
            --seed ${seed} \
            --temperature ${temperature} \
            --top_p ${top_p}
        done
    done
done
