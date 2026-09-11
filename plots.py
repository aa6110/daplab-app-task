import matplotlib.pyplot as plt
import numpy as np
import argparse

import json
import os

# this file is to construct the plots for measurement

def plot_pte(): # periodic test eval
    pass

def plot_re(): # rotational equilibrium or relative update
    pass

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

def plot_acc(): # plotting accuracy (can be used for training or testing)
    pass

def plot_le(): # plotting loss/epoch
    pass

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