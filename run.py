import pandas as pd
import numpy as np
import math
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data.dataloader import DataLoader
from params import *
from utils import *
from model import *

import pandas as pd


data_path = 'data/click_bait_full.csv'
checkpoint = "model_checkpoints/bert-base-uncased"


datasets_train_test = load_data(data_path)
tokenizer = load_tokenizer(checkpoint)


# def tokenize_function(record):
#     return tokenizer(record['headline'], truncation=True)

def tokenize_function(record):
    return tokenizer(record['headline'], truncation=True, max_length=block_size)

tokenized_datasets = datasets_train_test.map(tokenize_function, batched=True,)

train_data_iter = tokenized_datasets['train'].remove_columns(['id', 'headline', 'body',])
val_data_iter = tokenized_datasets['val'].remove_columns(['id', 'headline', 'body'])
test_data_iter = tokenized_datasets['test'].remove_columns(['id', 'headline', 'body'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

train_dataloader = DataLoader(train_data_iter, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
val_dataloader = DataLoader(val_data_iter, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
test_dataloader = DataLoader(test_data_iter, shuffle=False, batch_size=batch_size, collate_fn=data_collator)


## Params
vocab_size = tokenizer.vocab_size
torch.manual_seed(1337)

print("vocab_size", vocab_size)
enc_model = Encoder(vocab_size)
enc_model = enc_model.to(device)
optimizer = torch.optim.AdamW(enc_model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
EPOCHS = epochs  
total_accu = None
log_data = pd.DataFrame(columns=["epoch","batch_id", "split", "pres", "recal", "f1", "acc", "loss", "batch_size", "block_size", "learning_rate", "n_embd", "n_head", "n_layer", "dropout"])

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    log_data = train(train_dataloader, enc_model, optimizer, criterion, epoch, device, log_data)
    cm, pres, recal, f1, acc = evaluate(val_dataloader, enc_model, criterion, device)
    
    log_data = append_log(log_data, epoch, 0, "val", 
                                  pres, recal, f1, acc, torch.tensor(0.0))
    
    if total_accu is not None and total_accu > acc:
        scheduler.step()
    else:
        total_accu = acc
    print("-" * 70)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} "
        "valid f1-score {:8.3f}".format(
            epoch, time.time() - epoch_start_time, acc.item(), f1.item()
        )
    )
    print_cf(*cm)
    print("-" * 70)


print("Checking the results of test dataset.")
test_cm, test_pres, test_recal, test_f1, test_acc = evaluate(test_dataloader, enc_model, criterion, device)
print("| accuracy {:8.3f} "
      "| f1-score {:8.3f} ".format(test_acc,test_f1))

log_data = append_log(log_data, 0, 0, "test", 
                      test_pres, test_recal, test_f1, test_acc, torch.tensor(0.0))

time_stamp = datetime.now().strftime("%m_%d_%HH_%MM_%SS")
log_data.to_csv("logs/log_data_"+time_stamp+".csv", index=False)
print_cf(*test_cm)
torch.save(enc_model, "model_checkpoints/custom_models/model_"+time_stamp+".pt")

with open("logs/model_summary_"+time_stamp+".txt", "w") as f:
    f.write(repr(enc_model))