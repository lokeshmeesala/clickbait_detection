import mlflow
import time
import logging
import torch
from datetime import datetime
from torchmetrics import ConfusionMatrix
import src.utils as utility
from src.models.cross_model import CrossEncoder

import pandas as pd
torch.manual_seed(1337)

logging.basicConfig(filename=datetime.now().strftime('logs/run_%Y_%m_%d_%H_%M_%S.log'), 
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = utility.load_yaml('./params.yaml')
tokenizer = utility.load_tokenizer(params["checkpoint"])
vocab_size = tokenizer.vocab_size
print("device", device)

logging.info(f'device {device}')
logging.info(f'params {params}')
logging.info(f'vocab_size {vocab_size}')

mlflow.login()
mlflow.set_experiment("/mlflow-clickbait-tracker")


enc_model = CrossEncoder(vocab_size)
enc_model = enc_model.to(device)
optimizer = torch.optim.AdamW(enc_model.parameters(), lr=params["learning_rate"])
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
EPOCHS = params["epochs"]  
metric_fn = ConfusionMatrix(task="binary", num_classes=2).to(device)
data_handler = utility.DataHandler(params, tokenizer)
train_dataloader, test_dataloader = data_handler.prepare_data_loader()

with mlflow.start_run() as run:
    mlflow_params = {
        "epochs": EPOCHS,
        "learning_rate": params["learning_rate"],
        "batch_size": params["batch_size"],
        "loss_function": loss_fn.__class__.__name__,
        "metric_function": metric_fn.__class__.__name__,
        "optimizer": "AdamW",
    }

    mlflow.log_params(mlflow_params)

    time_stamp = datetime.now().strftime("%m_%d_%HH_%MM_%SS")


    with open("logs/model_summary_"+time_stamp+".txt", "w") as f:
        f.write(repr(enc_model))
    mlflow.log_artifact("logs/model_summary_"+time_stamp+".txt")


    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        utility.train(train_dataloader, enc_model, optimizer, loss_fn, metric_fn, epoch, device)
        utility.evaluate(test_dataloader, enc_model, loss_fn, metric_fn, device, epoch)

    time_stamp = datetime.now().strftime("%m_%d_%HH_%MM_%SS")
    torch.save(enc_model, "model_checkpoints/custom_models/model_"+time_stamp+".pt")
    mlflow.pytorch.log_model(enc_model, "model")