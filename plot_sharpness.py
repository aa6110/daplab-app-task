import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tabulate import tabulate


GROUPS = ["hidden", "nonhidden", "all"]
OPTS = {"adamw": ("AdamW", "tab:orange"), "muon": ("Muon", "tab:blue")}


def plot_perturbation(cfg, all_stats):
    """Loss increase vs relative perturbation size, one panel per parameter group.
    Error bars are the standard error of the mean over draws (std / sqrt(n))."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, group in zip(axes, GROUPS):
        for opt, (label, color) in OPTS.items():
            stats = all_stats[f"{group}_stats_{opt}"]
            sigmas, means, ses = [], [], []
            for s, v in stats.items():
                if v["mean"] > 0:  # log axis can't show noise around zero
                    sigmas.append(float(s))
                    means.append(v["mean"])
                    ses.append(v["std"] / np.sqrt(len(v["losses"])))
            ax.errorbar(sigmas, means, yerr=ses, marker="o", capsize=3, label=label, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{group} parameters")
        ax.set_xlabel("relative perturbation σ")
        ax.legend()
    axes[0].set_ylabel("mean loss increase")
    fig.suptitle("Random-perturbation sharpness (train subset, 10 draws per σ)")
    plt.tight_layout()
    plt.savefig(f"{cfg['output_dir']}/perturbation_sharpness.png", dpi=150)
    plt.close(fig)


def plot_interpolation(cfg, all_stats):
    """Loss along the straight line θ_A + t(θ_M − θ_A). t=0 is AdamW, t=1 is Muon."""
    it = all_stats["interpolate_stats"]
    t = np.array(it["t"])
    fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(12, 4))
    ax_tr.plot(t, it["train_loss"], marker="o", color="tab:green")
    ax_tr.set_yscale("log")
    ax_tr.set_title("Train loss (512-sentence subset)")
    ax_te.plot(t, it["test_loss"], marker="o", color="tab:red")
    ax_te.set_title("Test loss (full validation set)")
    for ax in (ax_tr, ax_te):
        ax.axvline(0, color="tab:orange", linestyle="--", label="AdamW (t=0)")
        ax.axvline(1, color="tab:blue", linestyle="--", label="Muon (t=1)")
        ax.set_xlabel("t")
        ax.set_ylabel("loss")
        ax.legend()
    fig.suptitle("Linear interpolation between the two solutions")
    plt.tight_layout()
    plt.savefig(f"{cfg['output_dir']}/interpolation.png", dpi=150)
    plt.close(fig)


def summary_table(cfg, all_stats, sigma="0.05"):
    """One row per parameter group: top Hessian eigenvalue and perturbation loss rise
    for each optimizer, plus the paired difference (same noise directions for both)."""
    rows = []
    for group in GROUPS:
        lam_a = all_stats[f"power_method_{group}_adamw"][-1]
        lam_m = all_stats[f"power_method_{group}_muon"][-1]
        pa = np.array(all_stats[f"{group}_stats_adamw"][sigma]["losses"])
        pm = np.array(all_stats[f"{group}_stats_muon"][sigma]["losses"])
        diff = pa - pm
        rows.append({
            "Group": group,
            "Top eigenvalue AdamW": round(lam_a, 3),
            "Top eigenvalue Muon": round(lam_m, 3),
            f"Loss rise @σ={sigma} AdamW": f"{pa.mean():.5f}",
            f"Loss rise @σ={sigma} Muon": f"{pm.mean():.5f}",
            "Paired diff (A−M) ± SE": f"{diff.mean():+.5f} ± {diff.std(ddof=1)/np.sqrt(len(diff)):.5f}",
            "A>M draws": f"{int((diff > 0).sum())}/{len(diff)}",
        })

    it = all_stats["interpolate_stats"]
    t = np.array(it["t"])
    i0, i1 = int(np.argmin(np.abs(t))), int(np.argmin(np.abs(t - 1)))
    gap_rows = [
        {"Solution": "AdamW (t=0)", "Train loss": round(it["train_loss"][i0], 4),
         "Test loss": round(it["test_loss"][i0], 4),
         "Gap": round(it["test_loss"][i0] - it["train_loss"][i0], 4)},
        {"Solution": "Muon (t=1)", "Train loss": round(it["train_loss"][i1], 4),
         "Test loss": round(it["test_loss"][i1], 4),
         "Gap": round(it["test_loss"][i1] - it["train_loss"][i1], 4)},
        {"Solution": f"Test-loss minimum (t={t[np.argmin(it['test_loss'])]:.2f})",
         "Train loss": round(min(it["train_loss"]), 4),
         "Test loss": round(min(it["test_loss"]), 4), "Gap": ""},
    ]

    md = "## Sharpness estimates\n\n" + tabulate(rows, headers="keys", tablefmt="github")
    md += "\n\n## Generalization gap along the interpolation line\n\n" + tabulate(gap_rows, headers="keys", tablefmt="github")
    with open(f"{cfg['output_dir']}/summary_table.md", "w") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    cfg = {"output_dir": "sharpness"}
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(f"{cfg['output_dir']}/results.json") as f:
        all_stats = json.load(f)

    plot_perturbation(cfg, all_stats)
    plot_interpolation(cfg, all_stats)
    summary_table(cfg, all_stats)