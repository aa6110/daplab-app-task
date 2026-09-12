from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding

from torch.utils.data import DataLoader

import torch

from tqdm import tqdm 

import os
from datetime import datetime

import json

from muon import SingleDeviceMuonWithAuxAdam

def load_data(mn, bs, seed): # model_name and batch_size
    ds = load_dataset("stanfordnlp/sst2") # the dataset task asks for

    # since test has hidden labels, we use two way split: train and test
    ds_train = ds["train"]
    ds_test = ds["validation"]

    # defining tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mn)

    def tokenize(examples): # we aren't going to pad here, but do dynamic padding
        return tokenizer(examples["sentence"], truncation=True) # truncation for edge cases

    # batch tokenizing
    ds_train = ds_train.map(tokenize, batched=True) # need to drop 'idx', 'sentence', and 'token_type_ids' because unnecessary
    ds_train = ds_train.remove_columns(["idx", "sentence", "token_type_ids"])
    ds_test = ds_test.map(tokenize, batched=True) # same thing here
    ds_test = ds_test.remove_columns(["idx", "sentence", "token_type_ids"])

    # data collator (for dynamic padding from earlier)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # using dataloader
    generator=torch.Generator().manual_seed(seed) # need to do this because otherwise the RNG is not reproducible with others
    train_dataloader = DataLoader(ds_train, batch_size=bs, shuffle=True, collate_fn=data_collator, generator=generator) # shuffle=True to remove order bias
    test_dataloader = DataLoader(ds_test, batch_size=bs, shuffle=False, collate_fn=data_collator) # shuffle=False to maintain order for evaluation

    return train_dataloader, test_dataloader

def split_params(model): # this is because build optimizer and gradient_sum needs this
    hidden_weights = [p for p in model.distilbert.transformer.parameters() if p.ndim >= 2]
    hidden_gains_biases = [p for p in model.distilbert.transformer.parameters() if p.ndim < 2]
    nonhidden_params = [*model.pre_classifier.parameters(), *model.classifier.parameters(), *model.distilbert.embeddings.parameters()]

    nonhidden_weights = hidden_gains_biases + nonhidden_params

    return hidden_weights, nonhidden_weights

def build_optimizer(hidden_weights, nonhidden_weights, config): # learning_rate and weight_decay
    optimizer = torch.optim.AdamW(params=hidden_weights+nonhidden_weights, lr=config["adamw_lr"], betas=config["adamw_betas"], eps=config["adamw_eps"], weight_decay=config["adamw_wd"])

    if config["muon"]: # the muon side of things
        # this part was basically copied from the muon example, just the calling of the parameters have been changed for distilbert
        # this was further simplified to split the parameters between hidden and nonhidden for multiple parts

        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=config["muon_lr"], weight_decay=config["muon_wd"], momentum=config["muon_momentum"]),
            dict(params=nonhidden_weights, use_muon=False, lr=config["adamw_lr"], weight_decay=config["adamw_wd"], betas=config["adamw_betas"], eps=config["adamw_eps"])
        ] # need to change the betas and eps for AdamW because base code uses Keller Jordan's rather than AdamW defaults, which would not be testing purely Muon

        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    return optimizer

def group_norm(tensors):
    return sum(t.norm()**2 for t in tensors).sqrt().item()

# training loop WARNING FOR TRAINING LOOP: metrics are logged while weights are changing. 
def train_loop(train_dataloader, eval_dataloader, model, hidden_weights, nonhidden_weights, hidden_pretrain_weights, nonhidden_pretrain_weights, optimizer, epoch, eval_freq, device): # loss function is cross entropy, chosen automatically
    model.train() # train mode
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    train_loss, correct = 0, 0
    losses = []
    hidden_grad_norms = []
    nonhidden_grad_norms = []
    hidden_update_norms = []
    nonhidden_update_norms = []
    hidden_pretrain_distances = []
    nonhidden_pretrain_distances = []
    hidden_weight_norms = []  
    nonhidden_weight_norms = []  
    eval_freq_losses = []
    eval_freq_accs = []
    global_steps = []

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="TRAINING")):
        batch = {k: v.to(device) for k, v in batch.items()} # loading batch onto device to train

        pred = model(**batch) # forward pass
        loss = pred.loss # cross-entropy (chosen automatically)
        loss.backward() # backward pass to compute gradients

        hidden_grad_norm, nonhidden_grad_norm = group_norm([p.grad for p in hidden_weights]), group_norm([p.grad for p in nonhidden_weights]) # required to measure the gradient norms
        
        with torch.no_grad():
            hidden_snapshot_params, nonhidden_snapshot_params = [p.clone() for p in hidden_weights], [p.clone() for p in nonhidden_weights] # needed for update norm
            hidden_weight_norm, nonhidden_weight_norm = group_norm(hidden_snapshot_params), group_norm(nonhidden_snapshot_params)

        hidden_weight_norms.append(hidden_weight_norm)
        nonhidden_weight_norms.append(nonhidden_weight_norm)

        optimizer.step() # update model parameters

        with torch.no_grad():
            diff_hidden_update_params, diff_nonhidden_update_params = [h - s for h, s in zip(hidden_weights, hidden_snapshot_params)], [nh - s for nh, s in zip(nonhidden_weights, nonhidden_snapshot_params)]
            hidden_update_norm, nonhidden_update_norm = group_norm(diff_hidden_update_params), group_norm(diff_nonhidden_update_params)
        
        optimizer.zero_grad() # reset gradients for the next step

        losses.append(loss.item())
        hidden_grad_norms.append(hidden_grad_norm)
        nonhidden_grad_norms.append(nonhidden_grad_norm)
        hidden_update_norms.append(hidden_update_norm)
        nonhidden_update_norms.append(nonhidden_update_norm)
        train_loss += loss.item()
        correct += (pred.logits.argmax(dim=-1) == batch["labels"]).sum().item()

        with torch.no_grad():
            hidden_pretrain_distance = group_norm([h - hp for h, hp in zip(hidden_weights, hidden_pretrain_weights)])
            nonhidden_pretrain_distance = group_norm([nh - nhp for nh, nhp in zip(nonhidden_weights, nonhidden_pretrain_weights)])
        hidden_pretrain_distances.append(hidden_pretrain_distance)
        nonhidden_pretrain_distances.append(nonhidden_pretrain_distance)

        if (epoch * len(train_dataloader) + batch_idx) % eval_freq == 0:
            model.eval() # switch to eval mode
            eval_freq_acc, eval_freq_loss = test_loop(eval_dataloader, model, device, verbose=False)
            eval_freq_accs.append(eval_freq_acc)
            eval_freq_losses.append(eval_freq_loss)
            global_steps.append(epoch * len(train_dataloader) + batch_idx)
            model.train() # switch back to train mode

    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

    return correct, train_loss, losses, hidden_grad_norms, nonhidden_grad_norms, hidden_update_norms, nonhidden_update_norms, hidden_pretrain_distances, nonhidden_pretrain_distances, hidden_weight_norms, nonhidden_weight_norms, eval_freq_accs, eval_freq_losses, global_steps

# test loop. similar to train loop but with some changes
def test_loop(dataloader, model, device, verbose=True): 
    model.eval() # evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): # no gradient computation during evaluation
        for batch in tqdm(dataloader, desc="TESTING", disable=not verbose): # the verbose thing is something i picked up from Claude for styling
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(**batch)
            loss = pred.loss

            test_loss += loss.item()
            correct += (pred.logits.argmax(dim=-1) == batch["labels"]).sum().item()
    
    test_loss /= num_batches
    correct /= size
    if verbose :
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct, test_loss

def save_run(artifact_dir, results, config, model):
    # first saving the config information
    with open(f"{artifact_dir}/config.json", "w") as f:
        json.dump(config, f)

    # now saving the actual data into a separate file
    with open(f"{artifact_dir}/results.json", "w") as f:
        json.dump(results, f)

    # saving the model itself
    model.save_pretrained(f"{artifact_dir}/model")

if __name__ == "__main__":

    # hyperparameters (easier for saving run)
    config = {
        "model_name": "distilbert/distilbert-base-uncased",
        "num_labels": 2, # for binary classification tasks like SST-2
        "batch_size": 32, # bert on sst2 paper
        "adamw_lr": 5e-5, # adjusted from what was used in the original BERT paper
        "adamw_eps": 1e-8, # default epsilon value for AdamW optimizer
        "adamw_betas": (0.9, 0.999), # default beta values for AdamW optimizer 
        "adamw_wd": 0.01, # default weight decay for AdamW optimizer
        "muon": False, # toggle for incorporating muon for 2d optimization
        "muon_lr": 2e-4, # sweeped and changed from the initial value used in the Muon blog
        "muon_momentum": 0.95, # default momentum value for Muon optimizer
        "muon_wd": 0.01, # default weight decay for Muon optimizer
        "epochs": 3, # fine-tuning, and also what was used on bert paper
        "eval_freq": 100, # evaluate the model every 100 steps (because fine-tuning, normally get very few datapoints)
        "seed": 42 # random number
    }

    torch.manual_seed(config["seed"]) # so that the only knob that is changing when testing agaist muon is the optimizer itself

    train_dataloader, test_dataloader = load_data(config["model_name"], config["batch_size"], config["seed"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=config["num_labels"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    hidden_weights, nonhidden_weights = split_params(model)

    optimizer = build_optimizer(hidden_weights, nonhidden_weights, config)

    tr_acc = []
    tr_loss_epoch = []
    tr_loss_step = []
    tr_hidden_grad_norms = []
    tr_nonhidden_grad_norms = []
    tr_hidden_pretrain_distances = []
    tr_nonhidden_pretrain_distances = []
    tr_hidden_update_norms = []
    tr_nonhidden_update_norms = []
    tr_hidden_weight_norms = []
    tr_nonhidden_weight_norms = []

    global_steps = []
    tst_periodic_accs = []
    tst_periodic_losses = []
    tst_acc = []
    tst_loss = []

    # making a snapshot of the current state of the model before training (for distance from pre-train params; another metric)
    hidden_pretrain_weights, nonhidden_pretrain_weights = [w.clone().detach() for w in hidden_weights], [w.clone().detach() for w in nonhidden_weights]

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_acc, train_loss_epoch, train_loss_step, hidden_grad_norms, nonhidden_grad_norms, hidden_update_norms, nonhidden_update_norms, hidden_pretrain_distances, nonhidden_pretrain_distances, hidden_weight_norms, nonhidden_weight_norms, eval_freq_accs, eval_freq_losses, steps = train_loop(train_dataloader, test_dataloader, model, hidden_weights, nonhidden_weights, hidden_pretrain_weights, nonhidden_pretrain_weights, optimizer, epoch, config["eval_freq"], device)
        tr_acc.append(train_acc)
        tr_loss_epoch.append(train_loss_epoch)
        tr_loss_step.extend(train_loss_step)
        tr_hidden_grad_norms.extend(hidden_grad_norms)
        tr_nonhidden_grad_norms.extend(nonhidden_grad_norms)
        tr_hidden_update_norms.extend(hidden_update_norms)
        tr_nonhidden_update_norms.extend(nonhidden_update_norms)
        tr_hidden_pretrain_distances.extend(hidden_pretrain_distances)
        tr_nonhidden_pretrain_distances.extend(nonhidden_pretrain_distances)
        tr_hidden_weight_norms.extend(hidden_weight_norms)
        tr_nonhidden_weight_norms.extend(nonhidden_weight_norms)
        tst_periodic_accs.extend(eval_freq_accs)
        tst_periodic_losses.extend(eval_freq_losses)
        global_steps.extend(steps)
    
        test_acc, test_loss = test_loop(test_dataloader, model, device)
        tst_acc.append(test_acc)
        tst_loss.append(test_loss)

    # compiling all the lists (data) into results dictionary 
    results = {
        "train_accuracy": tr_acc,
        "train_loss_epoch": tr_loss_epoch,
        "train_loss_step": tr_loss_step, # done
        "train_hidden_grad_norms": tr_hidden_grad_norms, # done
        "train_nonhidden_grad_norms": tr_nonhidden_grad_norms, # done
        "train_hidden_update_norms": tr_hidden_update_norms, # done
        "train_nonhidden_update_norms": tr_nonhidden_update_norms, # done
        "train_hidden_pretrain_distances": tr_hidden_pretrain_distances, # done
        "train_nonhidden_pretrain_distances": tr_nonhidden_pretrain_distances, # done
        "train_hidden_weight_norms": tr_hidden_weight_norms, # done
        "train_nonhidden_weight_norms": tr_nonhidden_weight_norms, # done
        "test_periodic_accuracy": tst_periodic_accs, # done
        "test_periodic_loss": tst_periodic_losses, # done
        "test_accuracy": tst_acc,
        "test_loss": tst_loss,
        "global_steps": global_steps # done
    }

    # getting the artiffact directory and making a subfolder per run based on time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = f"artifacts/adamw/{timestamp}"
    if config["muon"]:
        artifact_dir = f"artifacts/muon/{timestamp}"
    os.makedirs(artifact_dir, exist_ok=True)

    # saving all the information for the model, plots can be generated from different file
    save_run(artifact_dir, results, config, model) 