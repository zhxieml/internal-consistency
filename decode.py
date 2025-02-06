import argparse
import ast
import json
import multiprocessing

import torch

# Solve an error on my server: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
multiprocessing.set_start_method("spawn", force=True)
torch.multiprocessing.set_start_method("spawn", force=True)

from vllm import LLM, SamplingParams

from src.datasets.fewshot_dataset import format_icl, setup_dataset
from src.models.hf_model import MODEL_CONFIGS
from src.utils import set_seed

LLAMA2_ANSWER_END_TOKENS = [13, 13]  # "\n\n"
MIXTRAL_ANSWER_END_TOKENS = [13, 13]  # "\n\n"


def main(args):
    set_seed(args.seed)

    # load dataset
    dataset, path = args.dataset.split(":")
    ds = setup_dataset(dataset, path, least_to_most=args.least_to_most)

    prompts = [
        format_icl(
            ast.literal_eval(d),
            use_cot=True,
            use_icl=True,
            use_context=True,
            answer_formatter=lambda ans: ans,
        )
        for d in ds
    ]

    # load model
    if args.model.startswith("llama2"):
        raw_answer_end_tokens = LLAMA2_ANSWER_END_TOKENS
    elif args.model.startswith("mixtral"):
        raw_answer_end_tokens = MIXTRAL_ANSWER_END_TOKENS
    else:
        raise NotImplementedError

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        stop_token_ids=raw_answer_end_tokens,
        max_tokens=args.max_new_length,
        n=args.num_return_sequences,
    )

    # Create an LLM.
    # llm = LLM(model="TheBloke/Llama-2-70b-Chat-AWQ", quantization="AWQ", tensor_parallel_size=4)
    engine_kwargs = {
        "model": MODEL_CONFIGS[args.model]["model_name"],
        "dtype": "float16",
        "tensor_parallel_size": args.tensor_parallel,
        "max_num_seqs": args.batch_size,
        "seed": args.seed,
        "max_model_len": 4096,
    }

    if args.model == "mixtral-8x7b":
        engine_kwargs["model"] = "TheBloke/mixtral-8x7b-v0.1-AWQ"
        engine_kwargs["quantization"] = "AWQ"

    llm = LLM(**engine_kwargs)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )
    # Print the outputs.
    all_outputs = []
    for sample, output in zip([ast.literal_eval(d) for d in ds], outputs):
        prompt = output.prompt
        generated_texts = [out.text.strip() for out in output.outputs]
        all_outputs.append(dict(sample=sample, model_output=generated_texts))

    # save results
    with open(f"{args.output_prefix}_answers.jsonl", "w") as f:
        for line in all_outputs:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="something like prontoqa:prontoqa.json",
    )
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_length", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--least_to_most", action="store_true")

    # decode parameters
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--tensor_parallel", type=int, default=4)

    args = parser.parse_args()

    main(args)
