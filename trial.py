import pandas as pd
# import streamlit as st
import os,time
from datasets import Dataset
import gc
STATS={}

 
#from main
split=80
num_epochs = 1
STATS,dataset = get_dataset(r"E:\Python\Personal Project\NoCodeMLdash\data\sequence\Sarcasm_train.json")
model_name = "distilbert-base-uncased"


#%%
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from transformers import get_scheduler
from tqdm.auto import tqdm


def tokenize_function(examples):
    return tokenizer(examples["input"], padding="max_length", truncation=True)



tokenized_datasets = download_n_tokenize()
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=8)

print("here")
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

print("here81")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

#%%
from tqdm import trange
from time import sleep
t = trange(100, desc='Bar desc', leave=True)
for i in t:
    t.set_description("Bar desc (file %i)" % i, refresh=True)
    t.set_postfix(loss=i, gen=i+1, str='h',
                      lst=[1, 2])
    sleep(0.01)
    
