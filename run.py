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
# from model import *
from cross_model import *



import pandas as pd


datasets_train_test = load_data(data_path)
tokenizer = load_tokenizer(checkpoint)

# def tokenize_function(record):
#     return tokenizer(record['headline'], truncation=True, max_length=block_size)

# def tokenize_function(record):
#     return tokenizer(record['body'], truncation=True, max_length=block_size)

def tokenize_function(example):
    encoded_seq1 = tokenizer(example["headline"], return_tensors='pt', padding='max_length', truncation=True, max_length=120)
    encoded_seq2 = tokenizer(example["body"], return_tensors='pt', padding='max_length', truncation=True, max_length=640)
    return {"input_ids_seq1": encoded_seq1['input_ids'],
            "attention_mask_seq1": encoded_seq1['attention_mask'],
            "input_ids_seq2": encoded_seq2['input_ids'],
            "attention_mask_seq2": encoded_seq2['attention_mask'],
            "label": torch.tensor(example["label"])}

tokenized_datasets = datasets_train_test.map(tokenize_function, batched=True)

train_data_iter = tokenized_datasets['train'].remove_columns(['id', 'headline', 'body'])
test_data_iter = tokenized_datasets['test'].remove_columns(['id', 'headline', 'body'])

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
def custom_collate_fn(batch):
    input_ids_seq1 = torch.tensor([item["input_ids_seq1"] for item in batch])
    attention_masks_seq1 = torch.tensor([item["attention_mask_seq1"] for item in batch])
    input_ids_seq2 = torch.tensor([item["input_ids_seq2"] for item in batch])
    attention_masks_seq2 = torch.tensor([item["attention_mask_seq2"] for item in batch])
    label = torch.tensor([item["label"] for item in batch])

    padded_seq1 = torch.nn.utils.rnn.pad_sequence(input_ids_seq1, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_mask_seq1 = torch.nn.utils.rnn.pad_sequence(attention_masks_seq1, batch_first=True)

    padded_seq2 = torch.nn.utils.rnn.pad_sequence(input_ids_seq2, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_mask_seq2 = torch.nn.utils.rnn.pad_sequence(attention_masks_seq2, batch_first=True)

    return {
        "input_ids_seq1": padded_seq1,
        "attention_mask_seq1": padded_mask_seq1,
        "input_ids_seq2": padded_seq2,
        "attention_mask_seq2": padded_mask_seq2,
        "label": label
    }

# train_dataloader = DataLoader(train_data_iter, shuffle=True, batch_size=batch_size, collate_fn=data_collator, 
#                               num_workers=2, pin_memory=True)
# test_dataloader = DataLoader(test_data_iter, shuffle=False, batch_size=batch_size, collate_fn=data_collator, 
#                              num_workers=2, pin_memory=True)
train_dataloader = DataLoader(train_data_iter, shuffle=True, batch_size=batch_size, collate_fn=custom_collate_fn, 
                              num_workers=2, pin_memory=True)
test_dataloader = DataLoader(test_data_iter, shuffle=False, batch_size=batch_size, collate_fn=custom_collate_fn, 
                             num_workers=2, pin_memory=True)


## Params
vocab_size = tokenizer.vocab_size
torch.manual_seed(1337)

print("vocab_size", vocab_size)
# enc_model = Encoder(vocab_size)
enc_model = CrossEncoder(vocab_size)
enc_model = enc_model.to(device)
optimizer = torch.optim.AdamW(enc_model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
EPOCHS = epochs  
scheduler_metric = None
log_data = pd.DataFrame(columns=["epoch","batch_id", "split", "pres", "recall", "f1", "acc", "loss", "batch_size", "block_size", "learning_rate", "n_embd", "n_head", "n_layer", "dropout"])
print("Device Count",torch.cuda.device_count())
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    # print("************************************************")
    # print("Current Lr", scheduler.get_last_lr())
    # print("GPU USAGE", torch.cuda.memory_summary())
    # print("************************************************")
    log_data = train(train_dataloader, enc_model, optimizer, criterion, epoch, device, log_data)
    # break
    test_cm, test_pres, test_recall, test_f1, test_acc, test_loss = evaluate(test_dataloader, enc_model, criterion, device)
    
    log_data = append_log(log_data, epoch, 0, "test", 
                                  test_pres, test_recall, test_f1, test_acc, test_loss)
    
    if scheduler_metric is not None and scheduler_metric > test_f1.item():
        scheduler.step()
    else:
        scheduler_metric = test_f1.item()
    print("-" * 70)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s "
        "| test accuracy {:8.3f} "
        "| test f1-score {:8.3f} "
        "| test loss {:8.3f}".format(
            epoch, time.time() - epoch_start_time, test_acc.item(), test_f1.item(), test_loss.item()
        )
    )
    print_cf(*test_cm)
    print("=" * 70)
    

# print("Checking the results of test dataset.")
# test_cm, test_pres, test_recall, test_f1, test_acc = evaluate(test_dataloader, enc_model, criterion, device)
# print("| accuracy {:8.3f} "
#       "| f1-score {:8.3f} ".format(test_acc,test_f1))

# log_data = append_log(log_data, 0, 0, "test", 
#                       test_pres, test_recall, test_f1, test_acc, torch.tensor(0.0))

time_stamp = datetime.now().strftime("%m_%d_%HH_%MM_%SS")
log_data.to_csv("logs/log_data_"+time_stamp+".csv", index=False)
# print_cf(*test_cm)
torch.save(enc_model, "model_checkpoints/custom_models/model_"+time_stamp+".pt")

with open("logs/model_summary_"+time_stamp+".txt", "w") as f:
    f.write(repr(enc_model))