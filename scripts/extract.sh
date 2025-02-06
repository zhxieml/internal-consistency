declare -A data_map=(
    ["boolq"]="data/boolq.jsonl"
    ["coinflip"]="data/coin_flip.json"
    ["prontoqa"]="data/prontoqa.json"
    ["proofwriter"]="data/proofwriter.json"
)
declare -A bs_map=(
    ["mixtral-7b"]="4"
    ["mixtral-8x7b"]="2"
    ["llama2-7b"]="4"
    ["llama2-13b"]="2"
)
base_out_dir="res"

for no_icl in true; do
    for model in "llama2-7b" "llama2-13b" "mixtral-7b" "mixtral-8x7b"; do
        for dataset in "coinflip" "boolq" "prontoqa" "proofwriter"; do
            for seed in 111; do
                for no_context in false; do
                    use_gt_cot=false
                    no_icl_cot=true
                    echo "model: ${model}, dataset: ${dataset}, seed: ${seed}"
                    out_dir=${base_out_dir}/${dataset}-${model}
                    mkdir -p $out_dir

                    run_prefix=gtcot${use_gt_cot}-noicl${no_icl}-nocontext${no_context}
                    output_prefix=${out_dir}/${run_prefix}
                    echo "output_prefix: ${output_prefix}"

                    more_args=""
                    if [ "$use_gt_cot" = true ] ; then
                        more_args="--use_gt_cot"
                    fi
                    if [ "$no_icl" = true ] ; then
                        more_args="$more_args --no_icl"
                    fi
                    if [ "$no_context" = true ] ; then
                        more_args="$more_args --no_context"
                    fi
                    if [ "$no_icl_cot" = true ] ; then
                        more_args="$more_args --no_icl_cot"
                    fi

                    python extract.py \
                    --dataset ${dataset}:${data_map[${dataset}]} \
                    --model $model \
                    --batch_size ${bs_map[${model}]} \
                    --output_prefix ${output_prefix} \
                    --seed ${seed} \
                    --analyze_logits \
                    $more_args
                done
            done
        done
    done
done
