from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

from transformers import AutoModelForSequenceClassification

import torch

from train import split_params

import os
import json

from tqdm import tqdm
import numpy as np

from train import group_norm

# this function wont have too many comments because it's similar to load_data() in train.py
def load_eval_set(sharpness_config, split):
    ds = load_dataset("stanfordnlp/sst2")
    eval_set = ds[split].shuffle(seed=sharpness_config["seed"])
    if split == "train" : # only for train
        eval_set = eval_set.select(range(sharpness_config["eval_set_size"])) # it's important to do for train because we are measuring the sharpness of the "valley" in the training landscape

    tokenizer = AutoTokenizer.from_pretrained(sharpness_config["model_name"])

    def tokenize(examples):
        return tokenizer(examples["sentence"], truncation=True)

    eval_set = eval_set.map(tokenize, batched=True)
    eval_set = eval_set.remove_columns(["idx", "sentence", "token_type_ids"])

    # data collator 
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # dynamic padding

    eval_dataloader = DataLoader(eval_set, batch_size=sharpness_config["batch_size"], shuffle=False, collate_fn=data_collator)

    return eval_dataloader

def load_model(model_dir, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model

# there wont be too many comments here either since this is about the same as test_loop() in train.py
def loss_fn(model, dataloader, device):
    num_batches = len(dataloader)
    losses = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(**batch)
            loss = pred.loss

            losses += loss.item()
            
    return losses / num_batches

# this function is to see the areas around the two models in weight space for their losses to identify sharpness levels
def pertubation_sharpness(model, params, sharpness_config, dataloader, device, sigmas, n_draws):
    base_loss = loss_fn(model, dataloader, device)

    snapshot_params = [p.clone() for p in params]

    stats = {}

    for sigma in tqdm(sigmas, desc="PERTURBATION SHARPNESS"):
        perturbed_losses = []
        for i in range(n_draws):
            torch.manual_seed(sharpness_config["seed"] + i) # this is adding randomization and would be reproducible since in a loop

            for p in params: # for every hidden / nonhidden parameter
                with torch.no_grad():
                    # steps to create and add noise
                    noise = torch.randn_like(p)
                    noise *= sigma * p.norm() / noise.norm()
                    p.add_(noise)

            # generating the loss
            perturbed_loss = loss_fn(model, dataloader, device) - base_loss 
            perturbed_losses.append(perturbed_loss)

            with torch.no_grad():
                for p, snapshot_p in zip(params, snapshot_params):
                    p.copy_(snapshot_p)

        stats[sigma] = {
            "mean": torch.tensor(perturbed_losses).mean().item(),
            "std": torch.tensor(perturbed_losses).std().item(),
            "losses": perturbed_losses
        }

    return stats

# this is to see the landscape of weight space in terms of the loss "altitude" from one trained model to another
def interpolate(model_adamw, model_muon, train_loader, test_loader, device, ts):
    params_adamw = list(model_adamw.parameters())
    params_muon = list(model_muon.parameters())

    snapshot_params_adamw = [p.clone() for p in params_adamw]
    snapshot_params_muon = [p.clone() for p in params_muon]

    train_loss = []
    test_loss = []

    for t in tqdm(ts, desc="INTERPOLATION"):
        with torch.no_grad():
            for p_adamw, snapshot_p_adamw, snapshot_p_muon in zip(params_adamw, snapshot_params_adamw, snapshot_params_muon):
                p_adamw.copy_(snapshot_p_adamw * (1 - t) + snapshot_p_muon * t)

            # recording loss
            train_loss_adamw = loss_fn(model_adamw, train_loader, device)
            test_loss_adamw = loss_fn(model_adamw, test_loader, device)

            train_loss.append(train_loss_adamw)
            test_loss.append(test_loss_adamw)

            for p_adamw, snapshot_p_adamw in zip(params_adamw, snapshot_params_adamw):
                p_adamw.copy_(snapshot_p_adamw)

    stats = {
        "t": ts.tolist(), # needs .tolist() because it is being constructed from np.linspace()
        "train_loss": train_loss,
        "test_loss": test_loss
    }

    return stats

# Hessian-vector product helper function
def hvp(model, params, dataloader, device, v): # in all honesty, i didn't have too much time to read the paper, but i had the help of claude to implement the math for this function
    Hv = [torch.zeros_like(p) for p in params]

    for batch in dataloader:
        batch = {k: val.to(device) for k, val in batch.items()}
        pred = model(**batch)
        loss = pred.loss
        g = torch.autograd.grad(loss, params, create_graph=True) # the gradient where the graph is kept
        gv = sum(torch.sum(g_i * v_i) for g_i, v_i in zip(g, v)) # scalar
        hv = torch.autograd.grad(gv, params)
        Hv = [a + b.detach() for a, b in zip(Hv, hv)]

    Hv = [h / len(dataloader) for h in Hv]

    return Hv

# to get the greatest descent slop surrounding the weights of the models
def top_eigenvalue(model, params, dataloader, device, n_iters, seed): # this fucntion is also hard to understand mathematically
    torch.manual_seed(seed)
    v = [torch.randn_like(p) for p in params]
    norm_v = group_norm(v)
    v = [vi / norm_v for vi in v]

    lam_prev = None
    lams = []

    for i in range(n_iters):
        Hv = hvp(model, params, dataloader, device, v)
        lam = sum(torch.sum(v_i * Hv_i) for v_i, Hv_i in zip(v, Hv))
        lams.append(lam.item())
        norm_Hv = group_norm(Hv)
        v = [Hv_i / norm_Hv for Hv_i in Hv]
        if lam_prev is not None and torch.abs(lam - lam_prev) < 1e-3:
            break
        lam_prev = lam

    return lams

if __name__ == "__main__":

    sharpness_config = {
        "model_name": "distilbert/distilbert-base-uncased",
        "eval_set_size": 512, # arbitrary value
        "batch_size": 32, # fixed batch size for evaluation
        "seed": 42, # random number for reproducibility, what was used in training
        "model_dir_adamw": "artifacts/adamw/20260911_110518/model",
        "model_dir_muon": "artifacts/muon/20260911_134357/model",
        "sigmas": [0.001, 0.005, 0.01, 0.02, 0.05], # suggested by Claude, i didn't really know what values to put
        "n_draws": 10, # arbitrarily chosen; more draws means better std
        "output_dir": "sharpness", # folder to save sharpness results
        "ts": np.linspace(-0.5, 1.5, 25) # trying to see both sides of the basins from a side view
    }

    train_eval_loader = load_eval_set(sharpness_config, "train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_adamw = load_model(sharpness_config["model_dir_adamw"], device)
    model_muon = load_model(sharpness_config["model_dir_muon"], device)

    hidden_params_adamw, nonhidden_params_adamw = split_params(model_adamw)
    hidden_params_muon, nonhidden_params_muon = split_params(model_muon)

    # compute sharpness for both models
    hidden_stats_adamw = pertubation_sharpness(model_adamw, hidden_params_adamw, sharpness_config, train_eval_loader, device, sharpness_config["sigmas"], sharpness_config["n_draws"])
    nonhidden_stats_adamw = pertubation_sharpness(model_adamw, nonhidden_params_adamw, sharpness_config, train_eval_loader, device, sharpness_config["sigmas"], sharpness_config["n_draws"])
    all_stats_adamw = pertubation_sharpness(model_adamw, list(model_adamw.parameters()), sharpness_config, train_eval_loader, device, sharpness_config["sigmas"], sharpness_config["n_draws"])

    hidden_stats_muon = pertubation_sharpness(model_muon, hidden_params_muon, sharpness_config, train_eval_loader, device, sharpness_config["sigmas"], sharpness_config["n_draws"])
    nonhidden_stats_muon = pertubation_sharpness(model_muon, nonhidden_params_muon, sharpness_config, train_eval_loader, device, sharpness_config["sigmas"], sharpness_config["n_draws"])
    all_stats_muon = pertubation_sharpness(model_muon, list(model_muon.parameters()), sharpness_config, train_eval_loader, device, sharpness_config["sigmas"], sharpness_config["n_draws"])

    # interpolate (side view of landscape)
    test_eval_loader = load_eval_set(sharpness_config, "validation")
    interpolate_stats = interpolate(model_adamw, model_muon, train_eval_loader, test_eval_loader,  device, sharpness_config["ts"])

    # saving the results as one json for plots.py compatibility
    os.makedirs(sharpness_config["output_dir"], exist_ok=True)
    all_stats = {
        "hidden_stats_adamw": hidden_stats_adamw,
        "nonhidden_stats_adamw": nonhidden_stats_adamw,
        "all_stats_adamw": all_stats_adamw,
        "hidden_stats_muon": hidden_stats_muon,
        "nonhidden_stats_muon": nonhidden_stats_muon,
        "all_stats_muon": all_stats_muon,
        "interpolate_stats": interpolate_stats
    }
    with open(os.path.join(sharpness_config["output_dir"], "results.json"), "w") as f:
        json.dump(all_stats, f)