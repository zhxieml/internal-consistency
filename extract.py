"""Inspect hidden states & attentions of model outputs."""

import argparse
import ast
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from src.datasets.fewshot_dataset import format_icl, setup_data_loader, setup_dataset
from src.models import HFModel
from src.utils import set_seed

LLAMA2_FALSE_TOKENS = [7700]  # `False`
LLAMA2_TRUE_TOKENS = [5852]  # `True`
MIXTRAL_FALSE_TOKENS = [8250]  # `False`
MIXTRAL_TRUE_TOKENS = [6110]  # `True`

LLAMA2_CHOICES_TOKENS = [[319], [350], [315], [360], [382]]  # `A`, `B`, `C`, `D`， `E`
MIXTRAL_CHOICES_TOKENS = [[330], [365], [334], [384], [413]]  # `A`, `B`, `C`, `D`, `E`


def main(args):
    set_seed(args.seed)

    # load dataset
    dataset, path = args.dataset.split(":")
    ds = setup_dataset(dataset, path, least_to_most=args.least_to_most)
    if dataset == "answer":
        print("Using decoded response for extraction.")

    dataloader = setup_data_loader(
        ds, seed=args.seed, batch_size=args.batch_size, shuffle=args.shuffle
    )

    # load model
    mt = HFModel(
        args.model,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    mt.model.eval()
    norm_fn, head_fn = mt.get_norm_and_head(mt.model)

    if args.model.startswith("llama2"):
        TRUE_TOKENS = LLAMA2_TRUE_TOKENS
        FALSE_TOKENS = LLAMA2_FALSE_TOKENS
        CHOICES_TOKENS = LLAMA2_CHOICES_TOKENS
    elif args.model.startswith("mixtral"):
        TRUE_TOKENS = MIXTRAL_TRUE_TOKENS
        FALSE_TOKENS = MIXTRAL_FALSE_TOKENS
        CHOICES_TOKENS = MIXTRAL_CHOICES_TOKENS
    else:
        raise NotImplementedError

    if ast.literal_eval(ds[0])["answer"] in ["True", "False"]:
        target_tokens = [TRUE_TOKENS, FALSE_TOKENS]
        label_map = {"True": 0, "False": 1}
        task = "prontoqa"
    elif ast.literal_eval(ds[0])["answer"] in ["A", "B", "C", "D", "E"]:
        target_tokens = CHOICES_TOKENS
        label_map = {chr(65 + i): i for i in range(len(target_tokens))}
        task = "swag"
    else:
        raise NotImplementedError

    answer_trigger = ""
    answer_formatter = lambda ans: ans

    # forward arguments
    forward_kwargs = dict(
        output_hidden_states=args.analyze_hidden or args.analyze_logits,
        output_attentions=args.analyze_attn,
    )

    # set analysis flags
    if args.analyze_logits:
        prompt_to_idxs = defaultdict(list)
        all_norm_probs = []
        all_gt_preds = []
        all_correct = []

    # start generation
    pbar = tqdm(dataloader)
    for i, data in enumerate(pbar):
        queries = [
            format_icl(
                d,
                use_cot=not args.no_icl_cot,
                use_context=not args.no_context,
                answer_formatter=answer_formatter,
            )
            for d in data
        ]
        prompts = deepcopy(queries)

        if dataset == "answer":
            if args.use_gt_cot:
                answers = [
                    " ".join(d["chain_of_thought"]) + " " + answer_trigger for d in data
                ]
            else:
                separator = ". "
                decoded_cot = [d["model_output"].split(separator)[:-1] for d in data]
                answers = [
                    separator.join(cot) if len(cot) else data[i]["model_output"]
                    for i, cot in enumerate(decoded_cot)
                ]
                answers = [a + ("." if not a.endswith(".") else "") for a in answers]
            texts = [p + " " + a for p, a in zip(prompts, answers)]
        else:
            texts = prompts

            if args.use_gt_cot:
                if dataset.startswith("aqua"):
                    answers = [
                        "\n".join((" ".join(d["chain_of_thought"])).split("\n")[:-1])
                        + " "
                        + answer_trigger
                        for d in data
                    ]
                else:
                    answers = [
                        " ".join(d["chain_of_thought"]) + " " + answer_trigger
                        for d in data
                    ]
                texts = [p + " " + a for p, a in zip(prompts, answers)]
            else:
                texts = [p + " " + answer_trigger for p in prompts]

        if args.no_icl:
            texts = [t.split("\n\n")[-1] for t in texts]

        texts = [t.strip() for t in texts]

        # forward
        with torch.no_grad():
            inputs = mt.tokenizer(
                texts, padding=True, return_tensors="pt", return_offsets_mapping=True
            ).to(mt.model.device)
            forward_dict = mt.model.forward(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **forward_kwargs,
            )

        # extract internal information
        if args.analyze_logits:
            hidden = forward_dict["hidden_states"]
            norm_probs = []
            gt_preds = [label_map[d["answer"]] for d in data]

            with torch.no_grad():
                for layer, hidden_layer in enumerate(hidden[:-1]):
                    layer_logits = []
                    all_layer_logits = head_fn(norm_fn(hidden_layer[:, -1]))
                    for tokens in target_tokens:
                        logits = all_layer_logits[:, tokens].sum(-1)
                        layer_logits.append(logits)
                    layer_logits = torch.stack(layer_logits, dim=-1)
                    layer_norm_probs = layer_logits.softmax(dim=-1)
                    norm_probs.append(layer_norm_probs.cpu())

                # final layer
                final_logits = []
                all_final_logits = forward_dict["logits"][:, -1]
                top1_tokens = mt.tokenizer.batch_decode(
                    all_final_logits.argmax(dim=-1), skip_special_tokens=False
                )
                print("Top 1 tokens:", top1_tokens)
                for i in range(len(top1_tokens)):
                    if top1_tokens[i] not in ["A", "B", "C", "D", "E", "True", "False"]:
                        print("Text: ", texts[i])

                for tokens in target_tokens:
                    logits = all_final_logits[:, tokens].sum(-1)
                    final_logits.append(logits)
                final_logits = torch.stack(final_logits, dim=-1)
                final_norm_probs = final_logits.softmax(dim=-1)
                norm_probs.append(final_norm_probs.cpu())

            for sample_idx, p in enumerate(prompts):
                prompt_to_idxs[p].append(len(all_gt_preds) + sample_idx)

            norm_probs = torch.stack(norm_probs, dim=1)
            all_norm_probs.append(norm_probs)
            all_gt_preds.extend(gt_preds)
            all_correct.extend(
                [
                    p.lower() == gt.lower()
                    for p, gt in zip(top1_tokens, [d["answer"] for d in data])
                ]
            )

    # save results
    if args.analyze_logits:
        all_norm_probs = torch.cat(all_norm_probs, dim=0)
        all_gt_preds = torch.LongTensor(all_gt_preds)
        all_analyze_res = dict(
            all_norm_probs=all_norm_probs,
            all_gt_preds=all_gt_preds,
            prompt_to_idxs=prompt_to_idxs,
        )
        torch.save(
            all_analyze_res,
            f"{args.output_prefix}_analyze_logits_res_acc{np.mean(all_correct).item()}.pt",
        )


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
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--least_to_most", action="store_true")

    # analysis flags
    parser.add_argument("--analyze_attn", action="store_true")
    parser.add_argument("--analyze_hidden", action="store_true")
    parser.add_argument("--analyze_logits", action="store_true")
    parser.add_argument("--use_gt_cot", action="store_true")
    parser.add_argument("--no_icl", action="store_true")
    parser.add_argument("--no_icl_cot", action="store_true")
    parser.add_argument("--no_context", action="store_true")

    args = parser.parse_args()

    main(args)
