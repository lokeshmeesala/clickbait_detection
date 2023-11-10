import pandas as pd
import numpy as np
import math
import json
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data.dataloader import DataLoader
from params import *
from model import *

custom_random_seed = 1337

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, return_tensors='pt')
    return tokenizer

def load_data(data_path):
    click_bait_data = pd.read_csv(data_path)
    # click_bait_data.rename({'truthClass': 'label'}, inplace=True, axis=1)
    train_data, val_test_data = train_test_split(click_bait_data, test_size=0.25, random_state = custom_random_seed, stratify=click_bait_data['label'])
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state = custom_random_seed, stratify=val_test_data['label'])
    print("Train\n",train_data.label.value_counts(normalize=True))
    print("Val\n",val_data.label.value_counts(normalize=True))
    print("Test\n",test_data.label.value_counts(normalize=True))
    
    datasets_train_test = DatasetDict({
    "train": Dataset.from_pandas(train_data),
    "val": Dataset.from_pandas(val_data),
    "test": Dataset.from_pandas(test_data),
    })
    return datasets_train_test

def train(dataloader, model, optimizer, criterion, epoch, device, log_data):
    model.train()
    # total_acc, total_count, 
    total_loss = 0
    total_label = 0
    total_label_ones = 0
    total_label_zeros = 0
    total_predictions = torch.tensor([]).to(device)
    total_actuals = torch.tensor([]).to(device)
    log_interval = 200
    start_time = time.time()
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        text, label = batch['input_ids'], batch['labels']
        text, label = text.to(device), label.to(device)
        total_label_ones += label.sum()
        total_label += len(label)
        total_label_zeros += len(label) - label.sum()
        predicted_label = model(text)
        # print("Actual:", label)
        # print("Predicted:", predicted_label.argmax(1))
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        # total_acc += (predicted_label.argmax(1) == label).sum().item()
        # total_count += label.size(0)
        
        total_predictions = torch.cat([total_predictions, predicted_label.argmax(1)])
        total_actuals = torch.cat([total_actuals, label])

        total_loss += loss
        if idx % log_interval == 0 and idx > 0:
            cm, pres, recal, f1, acc = get_metrics(total_actuals, total_predictions)
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                # "| Ones Count {:3d} "
                # "| Zeros Count {:3d} "
                # "| Total Count {:3d} "
                "| accuracy {:8.3f} "
                "| f1-score {:8.3f} "
                "| loss {:8.3f}".format(
                    epoch, idx, len(dataloader), 
                    # total_label_ones,
                    # total_label_zeros,
                    # total_label,
                    acc,
                    f1,
                    total_loss / log_interval
                )
            )
            # print_cf(*cm)
            log_data = append_log(log_data, epoch, idx, "train", 
                                  pres, recal, f1, acc, total_loss / log_interval)
                            #    block_size, learning_rate, n_embd, n_head, n_layer, dropout)
            # total_acc, total_count, 
            total_loss = 0
            total_label = 0
            total_label_ones = 0
            total_label_zeros = 0
            total_predictions = torch.tensor([]).to(device)
            total_actuals = torch.tensor([]).to(device)
            start_time = time.time()
    return log_data       

def evaluate(dataloader, model, criterion, device):
    model.eval()
    # total_acc, total_count = 0, 0
    total_predictions = torch.tensor([]).to(device)
    total_actuals = torch.tensor([]).to(device)
    
    with torch.no_grad():
        # for idx, (label, text, offsets) in enumerate(dataloader):
        for idx, batch in enumerate(dataloader):
            text, label = batch['input_ids'], batch['labels']
            text, label = text.to(device), label.to(device)
            predicted_label = model(text)
            # print("Actual:", label)
            # print("Predicted:", predicted_label.argmax(1))
            loss = criterion(predicted_label, label)
            # total_acc += (predicted_label.argmax(1) == label).sum().item()
            # total_count += label.size(0)
            total_predictions = torch.cat([total_predictions, predicted_label.argmax(1)])
            total_actuals = torch.cat([total_actuals, label])

    # cm, pres, recal, f1, acc = get_metrics(total_actuals, total_predictions)

    return get_metrics(total_actuals, total_predictions)


def get_metrics(act, pred):
    tp = ((act==1) & (pred ==1)).sum()
    tn = ((act==0) & (pred ==0)).sum()
    fp = ((act==0) & (pred ==1)).sum()
    fn = ((act==1) & (pred ==0)).sum()
    pres = tp/(tp+fp)
    recal = tp/(tp+fn)
    f1 = 2*((pres*recal)/(pres+recal))
    acc = (act==pred).sum()/len(pred)
    
    if pres.isnan(): pres = torch.tensor([0])
    if recal.isnan(): recal = torch.tensor([0])
    if f1.isnan(): f1 = torch.tensor([0])
    if acc.isnan(): acc = torch.tensor([0])

    return [tn,fp,fn,tp], pres, recal, f1, acc

def print_cf(tn,fp,fn,tp):
    print(f"""         Pred -ve       Pred +ve
Act -ve  {tn}              {fp} 
Act +ve  {fn}              {tp}""")
    
    
def append_log(log_data, epoch, batch_id, split, pres, recal, f1, acc, loss):
    curr_log = {'epoch': epoch,
                'batch_id': batch_id,
                'split': split,
                'pres': round(float(pres.cpu().data.numpy()), 3),
                'recal': round(float(recal.cpu().data.numpy()), 3),
                'f1': round(float(f1.cpu().data.numpy()), 3),
                'acc': round(float(acc.cpu().data.numpy()), 3),
                'loss': round(float(loss.cpu().data.numpy()), 3),
                'batch_size': batch_size,
                'block_size': block_size,
                'learning_rate': learning_rate,
                'n_embd': n_embd,
                'n_head': n_head,
                'n_layer': n_layer,
                'dropout': dropout
                }
    log_data = log_data.append(pd.Series(curr_log, index=log_data.columns), ignore_index=True)
    return log_data