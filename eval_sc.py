import argparse
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sns.set_style("whitegrid")

NUM_PERMS = 100


def cal_aoc_of_accs(accs, layers=None):
    """

    accs: (num_samples, num_layers, 1)
    """

    if layers is not None:
        selected_accs = accs[:, layers]
    else:
        selected_accs = accs

    return (selected_accs).sum(1)


def normalize(scores):
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return scores

    return [(s - min_score) / (max_score - min_score) for s in scores]


def select_data(folder):
    all_files = os.listdir(folder)
    selected_file = None
    acc = None

    for file in all_files:
        selected_prefix = "gtcot"
        if file.startswith(selected_prefix):
            selected_file = file
            acc = float(file.split("acc")[-1].split(".pt")[0])
            break

    if selected_file is None:
        print("No file found for folder", folder)
        return None

    data = torch.load(os.path.join(folder, selected_file))
    data["acc"] = acc
    return data


def select_data_2(folder, seed, least_to_most=False):
    all_files = os.listdir(folder)
    selected_file = None
    acc = None

    for file in all_files:
        selected_prefix = "gtcot"
        if (
            file.startswith(selected_prefix)
            and (f"seed{seed}" in file)
            and (
                (least_to_most and "least_to_most" in file)
                or (not least_to_most and "least_to_most" not in file)
            )
        ):
            selected_file = file
            acc = float(file.split("acc")[-1].split(".pt")[0])
            break

    if selected_file is None:
        print("No file found for folder", folder)
        return None

    data = torch.load(os.path.join(folder, selected_file))
    data["acc"] = acc
    return data


def cal_agreement_accs(all_norm_probs, all_gt_preds):
    """Calculate the agreement accuracy of the model.

    Namely, we calculate the agreement rate between the calibrated predictions and the final ones.
    all_norm_probs: (num_samples, num_layers, num_labels)
    """
    all_norm_probs = all_norm_probs.float()
    num_layers, num_labels = all_norm_probs.shape[1:]
    quantiles = torch.quantile(
        all_norm_probs, (num_labels - 1) / num_labels, dim=0, keepdim=True
    )
    calibrated_scores = (all_norm_probs > quantiles).float() + (
        (1 / num_labels) * (all_norm_probs == quantiles).float()
    )
    # calibrated
    all_final_preds = (
        calibrated_scores[:, -1]
        .max(-1)
        .indices.unsqueeze(1)
        .unsqueeze(1)
        .repeat(1, num_layers, 1)
    )
    calibrated_consistency = torch.gather(calibrated_scores, 2, all_final_preds)

    # for faithfulness calculation
    all_gt_preds = (
        torch.LongTensor(all_gt_preds)
        .unsqueeze(1)
        .unsqueeze(1)
        .repeat(1, num_layers, 1)
    )
    calibrated_accs = torch.gather(calibrated_scores, 2, all_gt_preds)[:, -1]
    aocs = cal_aoc_of_accs(calibrated_consistency)

    return calibrated_accs, aocs


def weighed_mode_rand_ties(a, w=None):
    if w is None:
        w = [1.0] * len(a)

    cum_weights = defaultdict(float)

    for ai, wi in zip(a, w):
        cum_weights[ai] += wi

    max_w = max(cum_weights.values())
    ties = [k for k, v in cum_weights.items() if v == max_w]
    return np.random.choice(ties)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="boolq")
parser.add_argument("--model", type=str, default="mixtral-8x7b")
parser.add_argument("--aggregate", type=str, default="mean")
parser.add_argument("--least_to_most", action="store_true")
args = parser.parse_args()

# candidate_seeds = [111, 222, 333, 444, 555]
candidate_seeds = [111, 222]
dataset, model = args.dataset, args.model
max_votes = 20


work_dir = f"res/{dataset}-{model}"
if not args.least_to_most:
    text_files = [
        f"res/{dataset}-{model}/temp0.7-topp0.95-seed{seed}_answers.jsonl"
        for seed in candidate_seeds
    ]
else:
    text_files = [
        f"res/{dataset}-{model}/least_to_most-temp0.7-topp0.95-seed{seed}_answers.jsonl"
        for seed in candidate_seeds
    ]

all_model_outputs = defaultdict(list)
all_samples = defaultdict(list)
for f in text_files:
    df = pd.read_json(f, lines=True)
    model_outputs = df["model_output"]
    samples = df["sample"]
    for i, o in enumerate(model_outputs):
        all_model_outputs[i] += o
    for i, s in enumerate(samples):
        all_samples[i].append(s)
all_data = defaultdict(list)

all_norm_probs_concat = []
all_gt_preds_concat = []
for seed in candidate_seeds:  # every seed
    prob_data = select_data_2(work_dir, seed, least_to_most=args.least_to_most)
    if prob_data is None:
        continue

    all_norm_probs = prob_data["all_norm_probs"]
    all_norm_probs = all_norm_probs.reshape(-1, max_votes, *all_norm_probs.shape[1:])
    all_gt_preds = prob_data["all_gt_preds"]
    all_gt_preds = all_gt_preds.reshape(-1, max_votes, *all_gt_preds.shape[1:])

    all_norm_probs_concat.append(all_norm_probs)
    all_gt_preds_concat.append(all_gt_preds)

all_norm_probs = torch.cat(all_norm_probs_concat, dim=1)
all_gt_preds = torch.cat(all_gt_preds_concat, dim=1)

all_vote_accs = defaultdict(list)
all_vote_aocs = defaultdict(list)
all_vote_conf = defaultdict(list)
all_vote_soft = defaultdict(list)
for vote_idx in range(max_votes * len(candidate_seeds)):
    norm_probs = all_norm_probs[:, vote_idx]
    gt_preds = all_gt_preds[:, vote_idx]
    calibrated_accs, aocs = cal_agreement_accs(norm_probs, gt_preds)

    # calculate quantile (for `conf` baseline)
    num_labels = norm_probs.shape[-1]
    quantiles = torch.quantile(
        norm_probs, (num_labels - 1) / num_labels, dim=0, keepdim=True
    )
    threshold = quantiles[0, -1, -1]

    for i in range(len(calibrated_accs)):
        all_vote_accs[i].append(calibrated_accs[i].item())  # final layer
        all_vote_aocs[i].append(aocs[i].item())
        # all_vote_conf[i].append((norm_probs[i, -1, 0] - norm_probs[i, -1, 1]).abs().item())
        all_vote_conf[i].append(
            (norm_probs[i, -1, 0] - threshold).abs().item()
            + (norm_probs[i, -1, 1] - threshold).abs().item()
        )
        all_vote_soft[i].append(
            norm_probs[i, -1][(norm_probs[i, -1, -1] > threshold).int()].item()
        )

# NOTE: get argmax and argmin of aocs
#########################################################
mins = {i: np.min(aocs) for i, aocs in all_vote_aocs.items()}
maxs = {i: np.max(aocs) for i, aocs in all_vote_aocs.items()}
all_argmins = {
    i: [j for j, a in enumerate(aocs) if a == mins[i]]
    for i, aocs in all_vote_aocs.items()
}
all_argmaxs = {
    i: [j for j, a in enumerate(aocs) if a == maxs[i]]
    for i, aocs in all_vote_aocs.items()
}

minmax_gaps = {i: maxs[i] - mins[i] for i in all_vote_aocs.keys()}

label_mismatches = [
    (
        weighed_mode_rand_ties(accs) != weighed_mode_rand_ties(accs, aocs)
        and weighed_mode_rand_ties(accs, aocs)
    )
    for accs, aocs in zip(all_vote_accs.values(), all_vote_aocs.values())
]
mismatch_idxs = [i for i, mismatch in enumerate(label_mismatches) if mismatch]

# for each seed, also randomly permute n times
all_sc_accs = []
all_weighted_sc_accs = []
all_conf_weighted_sc_accs = []
all_soft_weighted_sc_accs = []
for _ in range(NUM_PERMS):
    for i in all_vote_accs.keys():
        random_idxs = random.sample(
            list(range(max_votes * len(candidate_seeds))),
            max_votes * len(candidate_seeds),
        )
        all_vote_accs[i] = [all_vote_accs[i][ri] for ri in random_idxs]
        all_vote_aocs[i] = [all_vote_aocs[i][ri] for ri in random_idxs]
        all_vote_conf[i] = [all_vote_conf[i][ri] for ri in random_idxs]
        all_vote_soft[i] = [all_vote_soft[i][ri] for ri in random_idxs]

    all_sc_accs_perm = []
    all_weighted_sc_accs_perm = []
    all_conf_weighted_sc_accs_perm = []
    all_soft_weighted_sc_accs_perm = []
    num_valid = len(all_vote_accs[0])

    all_vote_aocs_flatten = []
    for i in all_vote_aocs.keys():
        all_vote_aocs_flatten.extend(all_vote_aocs[i])
    all_model_outputs_flatten = []
    for i in all_model_outputs.keys():
        all_model_outputs_flatten.extend(all_model_outputs[i])

    end_idxs = list(range(1, 41))
    for end_idx in end_idxs:
        vote_accs = {i: accs[:end_idx] for i, accs in all_vote_accs.items()}
        vote_aocs = {i: aocs[:end_idx] for i, aocs in all_vote_aocs.items()}
        vote_conf = {i: conf[:end_idx] for i, conf in all_vote_conf.items()}
        vote_soft = {i: soft[:end_idx] for i, soft in all_vote_soft.items()}

        # calculate accuracy
        sc_accs = [
            weighed_mode_rand_ties(votes, [1.0] * len(votes))
            for votes in vote_accs.values()
        ]

        weighted_min_sc_accs = [
            weighed_mode_rand_ties(vote_accs[i], vote_aocs[i]).item()
            for i in vote_accs.keys()
        ]
        weighted_conf_sc_accs = [
            weighed_mode_rand_ties(vote_accs[i], vote_conf[i]).item()
            for i in vote_accs.keys()
        ]
        weighted_soft_sc_accs = [
            weighed_mode_rand_ties(vote_accs[i], vote_soft[i]).item()
            for i in vote_accs.keys()
        ]

        all_sc_accs_perm.append(np.mean(sc_accs))
        all_weighted_sc_accs_perm.append(np.mean(weighted_min_sc_accs))
        all_conf_weighted_sc_accs_perm.append(np.mean(weighted_conf_sc_accs))
        all_soft_weighted_sc_accs_perm.append(np.mean(weighted_soft_sc_accs))

    # aggregate permutations
    all_sc_accs.append(all_sc_accs_perm)
    all_weighted_sc_accs.append(all_weighted_sc_accs_perm)
    all_conf_weighted_sc_accs.append(all_conf_weighted_sc_accs_perm)
    all_soft_weighted_sc_accs.append(all_soft_weighted_sc_accs_perm)

if args.aggregate == "mean":
    aggregate_func = lambda x: np.array(x).mean(0).tolist()
elif args.aggregate == "std":
    aggregate_func = lambda x: np.array(x).std(0).tolist()
else:
    raise NotImplementedError

all_sc_accs = aggregate_func(all_sc_accs)
all_weighted_sc_accs = aggregate_func(all_weighted_sc_accs)
all_conf_weighted_sc_accs = aggregate_func(all_conf_weighted_sc_accs)
all_soft_weighted_sc_accs = aggregate_func(all_soft_weighted_sc_accs)

all_data["num_votes"].extend(end_idxs)
all_data["value"].extend(all_sc_accs)
all_data["type"].extend(["sc"] * len(end_idxs))
all_data["folder"].extend([work_dir] * len(end_idxs))

all_data["num_votes"].extend(end_idxs)
all_data["value"].extend(all_weighted_sc_accs)
all_data["type"].extend(["weighted_sc"] * len(end_idxs))
all_data["folder"].extend([work_dir] * len(end_idxs))

all_data["num_votes"].extend(end_idxs)
all_data["value"].extend(all_conf_weighted_sc_accs)
all_data["type"].extend(["conf_weighted_sc"] * len(end_idxs))
all_data["folder"].extend([work_dir] * len(end_idxs))

"""
all_data["num_votes"].extend(end_idxs)
all_data["value"].extend(all_soft_weighted_sc_accs)
all_data["type"].extend(["soft_weighted_sc"] * len(end_idxs))
all_data["folder"].extend([prob_folder] * len(end_idxs))
"""

print(
    all_sc_accs[0],
    all_sc_accs[-1],
    all_conf_weighted_sc_accs[-1],
    all_weighted_sc_accs[-1],
)

all_data = pd.DataFrame(all_data)
all_data.to_json(
    f"res/{dataset}_{model}.json",
    indent=4,
    orient="records",
)
fig, ax = plt.subplots()
sns.lineplot(all_data, x="num_votes", y="value", hue="type")
plt.legend()
plt.title(f"{dataset}-{model}")
plt.tight_layout()
plt.savefig(f"res/{dataset}_{model}.pdf")
