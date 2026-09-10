from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding

from torch.utils.data import DataLoader
import torch

from tqdm import tqdm 

import matplotlib.pyplot as plt

import os
from datetime import datetime

import json

from muon import SingleDeviceMuonWithAuxAdam

# CONFIGS
MODEL_NAME = "distilbert/distilbert-base-uncased"
BATCH_SIZE = 32 # bert on sst2 paper
ADAMW_LR = 5e-5 # what was used for bert on sst2 
MUON_LR = 2e-3 # starting point for the muon github page
EPS = 3 # fine-tuning, and also what was used on bert paper
WEIGHT_DECAY = 0.01 # default of PyTorch's AdamW
MUON = True # toggle displaying whether muon is on or not; AdamW is just being used in the background
SEED = 42 # eh, random ig

torch.manual_seed(SEED) # so that the only knob that is changing when testing agaist muon is the optimizer itself

# loading the dataset
ds = load_dataset("stanfordnlp/sst2") 

# since test has hidden labels, we use two way split: train and test
ds_train = ds["train"]
ds_test = ds["validation"]

# preparing the data 

# defining tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples): # we aren't going to pad here, but do dynamic padding
    return tokenizer(examples["sentence"], truncation=True) # truncation for edge cases

# batch tokenizing
ds_train = ds_train.map(tokenize, batched=True) # need to drop 'idx', 'sentence', and 'token_type_ids' because unnecessary
ds_train = ds_train.remove_columns(["idx", "sentence", "token_type_ids"])
ds_test = ds_test.map(tokenize, batched=True) # same thing here
ds_test = ds_test.remove_columns(["idx", "sentence", "token_type_ids"])

# loading model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) # need to specify two labels for SST-2

# data collator (for dynamic padding from earlier)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# using dataloader
train_dataloader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator) # shuffle=True to remove order bias
test_dataloader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator) # shuffle=False to maintain order for evaluation

# choosing the device - currently a RTX 4070 Laptop GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# setting up the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=ADAMW_LR, weight_decay=WEIGHT_DECAY)

if MUON: # the muon side of things
    # this part was basically copied from the muon example, just the calling of the parameters have been changed for distilbert
    hidden_weights = [p for p in model.distilbert.transformer.parameters() if p.ndim >= 2]
    hidden_gains_biases = [p for p in model.distilbert.transformer.parameters() if p.ndim < 2]
    nonhidden_params = [*model.pre_classifier.parameters(), *model.classifier.parameters(), *model.distilbert.embeddings.parameters()]

    param_groups = [
        dict(params=hidden_weights, use_muon=True, lr=MUON_LR, weight_decay=WEIGHT_DECAY),
        dict(params=hidden_gains_biases+nonhidden_params, use_muon=False, lr=ADAMW_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-08)
    ] # need to change the betas and eps for AdamW because base code uses Keller Jordan's rather than AdamW defaults, which would not be testing purely Muon

    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

# training loop WARNING FOR TRAINING LOOP: metrics are logged while weights are changing. 
def train_loop(dataloader, model, optimizer): # loss function is cross entropy, chosen automatically
    model.train() # train mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0

    for batch in tqdm(dataloader, desc="TRAINING"):
        batch = {k: v.to(device) for k, v in batch.items()} # loading batch onto device to train

        pred = model(**batch) # forward pass
        loss = pred.loss # cross-entropy (chosen automatically)
        loss.backward() # backward pass to compute gradients
        optimizer.step() # update model parameters
        optimizer.zero_grad() # reset gradients for the next step

        train_loss += loss.item()
        correct += (pred.logits.argmax(dim=-1) == batch["labels"]).sum().item()

    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

    return correct, train_loss

# test loop. similar to train loop but with some changes
def test_loop(dataloader, model): 
    model.eval() # evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): # no gradient computation during evaluation
        for batch in tqdm(dataloader, desc="TESTING"):
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(**batch)
            loss = pred.loss
            test_loss += loss.item()
            correct += (pred.logits.argmax(dim=-1) == batch["labels"]).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct, test_loss

# using the training and testing loops
tr_acc = []
tr_loss = []
tst_acc = []
tst_loss = []

for epoch in range(EPS):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_acc, train_loss = train_loop(train_dataloader, model, optimizer)
    tr_acc.append(train_acc)
    tr_loss.append(train_loss)

    test_acc, test_loss = test_loop(test_dataloader, model)
    tst_acc.append(test_acc)
    tst_loss.append(test_loss)

# plotting the results into six different graphs

# making a specific folder in artifacts in adamw based on time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
artifact_dir = f"artifacts/adamw/{timestamp}"
if MUON :
    artifact_dir = f"artifacts/muon/{timestamp}"
os.makedirs(artifact_dir, exist_ok=True)

with open(f"{artifact_dir}/results.json", "w") as f:
    json.dump({
        "tr_acc": tr_acc,
        "tr_loss": tr_loss,
        "tst_acc": tst_acc,
        "tst_loss": tst_loss
    }, f)

# storing the config in a separate json
if MUON :
    with open(f"{artifact_dir}/config.json", "w") as f:
        json.dump({
            "MODEL_NAME": MODEL_NAME,
            "EPS": EPS,
            "MUON_LR": MUON_LR,
            "ADAMW_LR": ADAMW_LR,
            "BATCH_SIZE": BATCH_SIZE,
            "OPTIMIZER": "SingleDeviceMuonWithAuxAdam",
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "SEED": SEED
        }, f)
else:
    with open(f"{artifact_dir}/config.json", "w") as f:
        json.dump({
            "MODEL_NAME": MODEL_NAME,
            "EPS": EPS,
            "ADAMW_LR": ADAMW_LR,
            "BATCH_SIZE": BATCH_SIZE,
            "OPTIMIZER": "AdamW",
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "SEED": SEED
        }, f)

# graph 1: training accuracy
plt.figure()
plt.plot(range(1, EPS+1), tr_acc, label="Training Accuracy") # making sure range is from 1, eps
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy over Epochs")
plt.legend()
plt.savefig(f"{artifact_dir}/training_accuracy.png")
    
# graph 2: training loss
plt.figure()
plt.plot(range(1, EPS+1), tr_loss, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.savefig(f"{artifact_dir}/training_loss.png")

# graph 3: testing accuracy
plt.figure()
plt.plot(range(1, EPS+1), tst_acc, label="Testing Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Testing Accuracy over Epochs")
plt.legend()
plt.savefig(f"{artifact_dir}/testing_accuracy.png")

# graph 4: testing loss
plt.figure()
plt.plot(range(1, EPS+1), tst_loss, label="Testing Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Testing Loss over Epochs")
plt.legend()
plt.savefig(f"{artifact_dir}/testing_loss.png")

# graph 5: combined accuracy
plt.figure()
plt.plot(range(1, EPS+1), tr_acc, label="Training Accuracy")
plt.plot(range(1, EPS+1), tst_acc, label="Testing Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Testing Accuracy over Epochs")
plt.legend()
plt.savefig(f"{artifact_dir}/combined_accuracy.png")

# graph 6: combined loss
plt.figure()
plt.plot(range(1, EPS+1), tr_loss, label="Training Loss")
plt.plot(range(1, EPS+1), tst_loss, label="Testing Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Testing Loss over Epochs")
plt.legend()
plt.savefig(f"{artifact_dir}/combined_loss.png")
