import matplotlib.pyplot as plt
import os

def plot_perturbation(sharpness_config, all_stats):
    for key, values in all_stats.items():
        if "_stats_" not in key:
            continue
        sigmas = [float(s) for s in values]            # keys are strings after JSON
        means  = [values[s]["mean"] for s in values]
        stds   = [values[s]["std"]  for s in values]
        plt.errorbar(sigmas, means, yerr=stds, label=key)
    plt.xlabel("Relative Perturbation")
    plt.xscale("log")
    plt.ylabel("Loss Increase")
    plt.yscale("log")
    plt.title("Relative Perturbation vs Loss Increase")
    plt.legend()
    plt.savefig(f"{sharpness_config['output_dir']}/perturbation_sharpness.png")
    plt.close()

if __name__ == "__main__":

    sharpness_plots_config = {
        "output_dir": "sharpness"
    }

    # making the plots directory
    os.makedirs(sharpness_plots_config["output_dir"], exist_ok=True)

    # loading the json file for sharpness evals
    import json
    with open(f"{sharpness_plots_config['output_dir']}/results.json", "r") as f:
        all_stats = json.load(f)

    plot_perturbation(sharpness_plots_config, all_stats)