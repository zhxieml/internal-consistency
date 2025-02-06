import numpy as np
import seaborn as sns
import torch


def cal_aoc_of_accs(accs):
    """

    accs: (num_samples, num_layers, 1)
    """

    return (accs).sum(1)


def cal_agreement_accs(all_norm_probs, all_gt_preds, verbose=False):
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
    # un-calibrated
    # all_final_preds = all_norm_probs[:,-1].max(-1).indices.unsqueeze(1).unsqueeze(1).repeat(1, num_layers, 1)
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
    vanilla_preds = all_norm_probs[:, -1].max(-1).indices
    vanilla_accs = vanilla_preds == all_gt_preds
    calibrated_preds = calibrated_scores[:, -1].max(-1).indices
    calibrated_accs = calibrated_preds == all_gt_preds
    aocs = cal_aoc_of_accs(calibrated_consistency)

    ref_accs = calibrated_accs

    # get palette's first and second color
    correct_color = "grey"
    wrong_color = sns.color_palette("colorblind")[3]

    if verbose:
        print(aocs[ref_accs].mean(), aocs[~ref_accs].mean(), aocs.mean())
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(6, 3))
        axs[0].hist(
            aocs[ref_accs].numpy() / num_layers,
            alpha=0.5,
            color=correct_color,
            label="Correct",
            density=True,
        )
        axs[0].hist(
            aocs[~ref_accs].numpy() / num_layers,
            alpha=0.5,
            color=wrong_color,
            label="Wrong",
            density=True,
        )

        # draw mean line
        axs[0].axvline(
            aocs[ref_accs].mean() / num_layers,
            color=correct_color,
            linestyle="dashed",
            linewidth=3,
        )
        axs[0].axvline(
            aocs[~ref_accs].mean() / num_layers,
            color=wrong_color,
            linestyle="dashed",
            linewidth=3,
        )

        axs[0].legend()
        axs[0].set_xlabel("Inner Consistency (%)")
        axs[0].set_ylabel("Density")

        # set xticks format
        axs[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x * 100)))

        # plt.tight_layout()
        # # plt.savefig("tmp/hist.pdf")
        # plt.show()

        # more temporary results
        # fig, ax = plt.subplots()
        axs[1].plot(
            calibrated_consistency[ref_accs].mean(0).flatten(),
            color=correct_color,
            label="Correct",
        )
        axs[1].plot(
            calibrated_consistency[~ref_accs].mean(0).flatten(),
            color=wrong_color,
            label="Wrong",
        )
        axs[1].set_xlabel("Layer")
        axs[1].set_ylabel("Agreement (%)")

        # set yticks format
        axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x * 100)))

        axs[1].legend()
        plt.tight_layout()
        plt.savefig("tmp/hist.pdf")
        plt.show()

    # calculate original accuracy vs accuracy by aoc
    true_consistency = calibrated_scores[:, :, [0]]
    false_consistency = calibrated_scores[:, :, [1]]
    true_aocs = cal_aoc_of_accs(true_consistency).flatten()
    false_aocs = cal_aoc_of_accs(false_consistency).flatten()

    ref_preds = calibrated_preds
    aoc_preds = torch.clone(ref_preds)
    sorted_idxs = torch.sort(true_aocs - false_aocs).indices
    aoc_preds[sorted_idxs[: len(sorted_idxs) // 2]] = 0
    aoc_preds[sorted_idxs[len(sorted_idxs) // 2 :]] = 1

    aoc_accs = aoc_preds == all_gt_preds

    other_stats = {
        "aoc_accs": aoc_accs,
        "aocs": aocs,
        "ref_accs": ref_accs,
        "calibrated_accs": calibrated_accs,
        "calibrated_consistency": calibrated_consistency,
    }

    return calibrated_consistency, other_stats
