import pandas as pd
import numpy as np
import math
import json
import mlflow
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data.dataloader import DataLoader

import src.utils as utility
from src.models.cross_model import CrossEncoder

import pandas as pd
torch.manual_seed(1337)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


params = utility.load_yaml('./params.yaml')
tokenizer = utility.load_tokenizer(params["checkpoint"])
# tokenize_function = utility.tokenize_function
vocab_size = tokenizer.vocab_size

print("device", device)

mlflow.login()
mlflow.set_experiment("/mlflow-clickbait-tracker")


print("vocab_size", vocab_size)
enc_model = CrossEncoder(vocab_size)
enc_model = enc_model.to(device)
optimizer = torch.optim.AdamW(enc_model.parameters(), lr=params["learning_rate"])
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
EPOCHS = params["epochs"]  
scheduler_metric = None
log_data = pd.DataFrame(columns=["epoch","batch_id", "split", "pres", "recall", "f1", "acc", "loss", "batch_size", "block_size", "learning_rate", "n_embd", "n_head", "n_layer", "dropout"])





data_handler = utility.DataHandler(params, tokenizer)

train_dataloader, test_dataloader = data_handler.prepare_data_loader()




with mlflow.start_run() as run:
    mlflow_params = {
        "epochs": EPOCHS,
        "learning_rate": params["learning_rate"],
        "batch_size": params["batch_size"],
        "loss_function": loss_fn.__class__.__name__,
        "metric_function": "Accuracy",
        "optimizer": "AdamW",
    }

    mlflow.log_params(mlflow_params)

    time_stamp = datetime.now().strftime("%m_%d_%HH_%MM_%SS")


    with open("logs/model_summary_"+time_stamp+".txt", "w") as f:
        f.write(repr(enc_model))
    mlflow.log_artifact("logs/model_summary_"+time_stamp+".txt")


    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        log_data = utility.train(train_dataloader, enc_model, optimizer, loss_fn, epoch, device, log_data)
        test_cm, test_pres, test_recall, test_f1, test_acc, test_loss = utility.evaluate(test_dataloader, enc_model, loss_fn, device)
        
        log_data = utility.append_log(log_data, epoch, 0, "test", 
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
        utility.print_cf(*test_cm)
        print("=" * 70)
        

    # time_stamp = datetime.now().strftime("%m_%d_%HH_%MM_%SS")
    log_data.to_csv("logs/log_data_"+time_stamp+".csv", index=False)
    # print_cf(*test_cm)
    torch.save(enc_model, "model_checkpoints/custom_models/model_"+time_stamp+".pt")

    mlflow.pytorch.log_model(enc_model, "model")

# with open("logs/model_summary_"+time_stamp+".txt", "w") as f:
#     f.write(repr(enc_model))