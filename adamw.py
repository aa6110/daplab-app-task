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

# CONFIGS
MODEL_NAME = "distilbert/distilbert-base-uncased"
BATCH_SIZE = 32
LR = 5e-5
EPS = 3
SEED = 42

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
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

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
os.makedirs(artifact_dir, exist_ok=True)

with open(f"{artifact_dir}/results.json", "w") as f:
    json.dump({
        "tr_acc": tr_acc,
        "tr_loss": tr_loss,
        "tst_acc": tst_acc,
        "tst_loss": tst_loss
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
