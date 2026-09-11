import matplotlib.pyplot as plt
import numpy as np
import argparse

import json
import os
import tabulate

# this file is to construct the plots for measurement

def plot_pte(runs_data, plot_config): # periodic test eval
    # two figures, one for loss one for acc
    fig, (ax_h, ax_n) = plt.subplots(1, 2,figsize=(12, 4))

    for results, config, label in runs_data:
        pte_acc = results["test_periodic_accuracy"]
        pte_loss = results["test_periodic_loss"]
        ax_h.plot(results["global_steps"], pte_acc, label=label)
        ax_n.plot(results["global_steps"], pte_loss, label=label)

    ax_h.set_title("Periodic Test Accuracy")
    ax_h.set_xlabel("Steps")
    ax_h.set_ylabel("Accuracy")
    ax_h.legend()

    ax_n.set_title("Periodic Test Loss")
    ax_n.set_xlabel("Steps")
    ax_n.set_ylabel("Loss")
    ax_n.legend()

    plt.tight_layout()
    plt.savefig(f"{plot_config['output_dir']}/periodic_test_eval.png")
    plt.close()

def plot_re(runs_data, plots_config, smooth_flag): # rotational equilibrium or relative update
    # this one requires a bit of additional computation to determine the rotational equilibrium or relative update
    for results, config, label in runs_data:
        results["train_hidden_re"] = np.array(results["train_hidden_update_norms"]) / np.array(results["train_hidden_weight_norms"])
        results["train_nonhidden_re"] = np.array(results["train_nonhidden_update_norms"]) / np.array(results["train_nonhidden_weight_norms"])

    plot_two_panels(runs_data, plots_config, "train_hidden_re", "train_nonhidden_re", "Rotational Equilibrium", "rotational_equilibrium.png", "Steps", "RE", smooth_flag)

def smooth(data, w):
    if w <= 1:
        return data
    smoothed = np.convolve(data, np.ones(w)/w, mode='valid')
    return smoothed

def plot_ls(runs_data, plot_config): # plotting loss/steps
    plt.figure()
    for results, config, label in runs_data:
        loss_steps = results["train_loss_step"]
        smoothed_loss = smooth(loss_steps, plot_config["w"])
        plt.plot(range(plot_config["w"], len(loss_steps)+1), smoothed_loss, label=label)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{plot_config['output_dir']}/loss_steps.png")
    plt.close()

def plot_two_panels(runs_data, plot_config, hidden_key, nonhidden_key, tilte, filename, xlabel, ylabel, smooth_flag): # this is used for plotting hidden and non-hidden variables side by side
    # this plot will be slightly different since there will be hidden and nonhidden params
    fig, (ax_h, ax_n) = plt.subplots(1, 2, figsize=(12, 4))

    for results, config, label in runs_data:
        hidden_values = results[hidden_key]
        nonhidden_values = results[nonhidden_key]
        if smooth_flag:
            hidden_values = smooth(hidden_values, plot_config["w"])
            nonhidden_values = smooth(nonhidden_values, plot_config["w"])
            ax_h.plot(range(plot_config["w"], len(hidden_values)+plot_config["w"]), hidden_values, label=label)
            ax_n.plot(range(plot_config["w"], len(nonhidden_values)+plot_config["w"]), nonhidden_values, label=label)
        else:
            ax_h.plot(hidden_values, label=label)
            ax_n.plot(nonhidden_values, label=label)

    ax_h.set_title(f"Hidden {tilte}")
    ax_h.set_xlabel(xlabel)
    ax_h.set_ylabel(ylabel)
    ax_h.legend()

    ax_n.set_title(f"Non-Hidden {tilte}")
    ax_n.set_xlabel(xlabel)
    ax_n.set_ylabel(ylabel)
    ax_n.legend()

    plt.tight_layout()
    plt.savefig(f"{plot_config['output_dir']}/{filename}")
    plt.close()

def summary_table(runs_data, plot_config):
    table_data = []

    # need best report vals
    for results, config, label in runs_data:
        # getting best periodic test acc and step at best
        best_periodic_test_acc = max(results["test_accuracy"]) if results["test_accuracy"] else None
        step_at_best_periodic_test_acc = results["test_accuracy"].index(best_periodic_test_acc) if best_periodic_test_acc is not None else None
        table_data.append({
            "Label": label,
            "Best Periodic Test Accuracy": best_periodic_test_acc,
            "Step at Best Periodic Test Accuracy": step_at_best_periodic_test_acc
        })

        # getting the lowest periodic test loss and step at lowest
        lowest_periodic_test_loss = min(results["test_loss"]) if results["test_loss"] else None
        step_at_lowest_periodic_test_loss = results["test_loss"].index(lowest_periodic_test_loss) if lowest_periodic_test_loss is not None else None
        table_data.append({
            "Label": label,
            "Lowest Periodic Test Loss": lowest_periodic_test_loss,
            "Step at Lowest Periodic Test Loss": step_at_lowest_periodic_test_loss
        })

        # getting final test accuracy and step at final
        final_periodic_test_acc = results["test_accuracy"][-1] if results["test_accuracy"] else None
        step_at_final_periodic_test_acc = results["test_accuracy"].index(final_periodic_test_acc) if final_periodic_test_acc is not None else None
        table_data.append({
            "Label": label,
            "Final Periodic Test Accuracy": final_periodic_test_acc,
            "Step at Final Periodic Test Accuracy": step_at_final_periodic_test_acc
        })

        # final train acc
        final_periodic_train_acc = results["train_accuracy"][-1] if results["train_accuracy"] else None
        step_at_final_periodic_train_acc = results["train_accuracy"].index(final_periodic_train_acc) if final_periodic_train_acc is not None else None
        table_data.append({
            "Label": label,
            "Final Periodic Train Accuracy": final_periodic_train_acc,
            "Step at Final Periodic Train Accuracy": step_at_final_periodic_train_acc
        })

        # final hidden distance from pretrained
        final_hidden_pretrain_dist = results["train_hidden_pretrain_distances"][-1] if results["train_hidden_pretrain_distances"] else None
        step_at_final_hidden_pretrain_dist = results["train_hidden_pretrain_distances"].index(final_hidden_pretrain_dist) if final_hidden_pretrain_dist is not None else None
        table_data.append({
            "Label": label,
            "Final Hidden Pretrain Distance": final_hidden_pretrain_dist,
            "Step at Final Hidden Pretrain Distance": step_at_final_hidden_pretrain_dist
        })

        # final hidden weight norm
        final_hidden_weight_norm = results["train_hidden_weight_norms"][-1] if results["train_hidden_weight_norms"] else None
        step_at_final_hidden_weight_norm = results["train_hidden_weight_norms"].index(final_hidden_weight_norm) if final_hidden_weight_norm is not None else None
        table_data.append({
            "Label": label,
            "Final Hidden Weight Norm": final_hidden_weight_norm,
            "Step at Final Hidden Weight Norm": step_at_final_hidden_weight_norm
        })

    table_str = tabulate.tabulate(table_data, headers="keys", tablefmt="github")

    # saving the summary table as a markdown file
    with open(f"{plot_config['output_dir']}/summary_table.md", "w") as f:
        f.write(table_str)

def load_run(run_dir): 
    with open(f"{run_dir}/results.json", "r") as f:
        results = json.load(f)
    with open(f"{run_dir}/config.json", "r") as f:
        config = json.load(f)
    label = ""

    # this is the naming convention being used since i dont have enough time to test things apart from lr
    if config["muon"]:
        label = "Muon " + str(config["muon_lr"]) + " (aux " + str(config["adamw_lr"]) + ")"
    else :
        label = "AdamW " + str(config["adamw_lr"])

    return results, config, label

if __name__ == "__main__":

    # hyperparameters for plotting
    plot_config = {
        "w": 50, # this is the width for the smoothening for loss/steps
        "output_dir": "figures" # where to output the graphs
    }

    # arguments must be passed so that the plots can be stacked, this is for clean comparisons for sweeps, or for optimizer differences
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args() # gets true path of folders

    runs_data = [load_run(run_dir) for run_dir in args.runs]

    # making the plots directory
    os.makedirs(plot_config["output_dir"], exist_ok=True)

    # plotting the loss/steps
    plot_ls(runs_data, plot_config)

    # plotting hidden/nonhidden graphs
    plot_two_panels(runs_data, plot_config, "train_hidden_pretrain_distances", "train_nonhidden_pretrain_distances", "Pretrain Distances", "distance_pretrain_weights.png", "Steps", "Distance", smooth_flag=False)
    plot_two_panels(runs_data, plot_config, "train_hidden_grad_norms", "train_nonhidden_grad_norms", "Gradient Norms", "gradient_norms.png", "Steps", "Norm", smooth_flag=True)
    plot_two_panels(runs_data, plot_config, "train_hidden_update_norms", "train_nonhidden_update_norms", "Update Norms", "update_norms.png", "Steps", "Norm", smooth_flag=True)
    plot_two_panels(runs_data, plot_config, "train_hidden_weight_norms", "train_nonhidden_weight_norms", "Weight Norms", "weight_norms.png", "Steps", "Norm", smooth_flag=False)
    plot_re(runs_data, plot_config, smooth_flag=True)
    plot_pte(runs_data, plot_config)
    summary_table(runs_data, plot_config)