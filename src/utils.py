import pandas as pd
import yaml
import os
import mlflow
import torch
import logging
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader


custom_random_seed = 1337

class DataHandler:
    def __init__(self, params, tokenizer) -> None:
        """
        Initialize the DataHandler class with parameters and tokenizer.

        Args:
        - params (dict): dictionary of config parameters.
        - tokenizer: tokenizer instance.
        """
        self.params = params
        self.tokenizer = tokenizer
        
    def custom_collate_fn(batch):
        """
        Custom collate function for DataLoader to handle batch sequence padding.

        Args:
        - batch (list): A batch of inputs, each sample contains a sequence and label.

        Returns:
        - dict: padded inputs and labels.
        """
        input_ids_seq1 = [torch.tensor(item["input_ids_seq1"]) for item in batch]
        attention_masks_seq1 = [torch.tensor(item["attention_mask_seq1"]) for item in batch]
        input_ids_seq2 = [torch.tensor(item["input_ids_seq2"]) for item in batch]
        attention_masks_seq2 = [torch.tensor(item["attention_mask_seq2"]) for item in batch]
        label = torch.tensor([item["label"] for item in batch])

        padded_seq1 = torch.nn.utils.rnn.pad_sequence(input_ids_seq1, batch_first=True)
        padded_mask_seq1 = torch.nn.utils.rnn.pad_sequence(attention_masks_seq1, batch_first=True)

        padded_seq2 = torch.nn.utils.rnn.pad_sequence(input_ids_seq2, batch_first=True)
        padded_mask_seq2 = torch.nn.utils.rnn.pad_sequence(attention_masks_seq2, batch_first=True)

        return {
            "input_ids_seq1": padded_seq1,
            "attention_mask_seq1": padded_mask_seq1,
            "input_ids_seq2": padded_seq2,
            "attention_mask_seq2": padded_mask_seq2,
            "label": label
        }
    
    def tokenize_function(example):
        """
        Tokenize the given example.

        Args:
        - example: A dictionary containing "headline", "body", and "label" keys.

        Returns:
        - dict: A dictionary containing tokenized sequences and label.
        """
        encoded_seq1 = DataHandler.tokenizer(example["headline"], truncation=True, max_length=640)
        encoded_seq2 = DataHandler.tokenizer(example["body"], truncation=True, max_length=640)
        return {
            "input_ids_seq1": encoded_seq1['input_ids'],
            "attention_mask_seq1": encoded_seq1['attention_mask'],
            "input_ids_seq2": encoded_seq2['input_ids'],
            "attention_mask_seq2": encoded_seq2['attention_mask'],
            "label": torch.tensor(example["label"])}

    def prepare_data_loader(self):
        """
        Prepare data loader for training and testing datasets.

        Returns:
        - tuple: A tuple containing train and test DataLoader objects.
        """
        tokenized_datasets_path = self.params["tokenized_datasets_path"]
        if os.path.exists(tokenized_datasets_path):
            logging.info(f"The file at {tokenized_datasets_path} exists.")

            tokenized_datasets = torch.load(tokenized_datasets_path)
        else:
            datasets_train_test = load_data(self.params["data_path"])
            logging.info(f"The file at {tokenized_datasets_path} does not exists.")

            tokenized_datasets = datasets_train_test.map(self.tokenize_function, batched=True)
            torch.save(tokenized_datasets, tokenized_datasets_path)

        train_data_iter = tokenized_datasets['train'].remove_columns(['id', 'headline', 'body'])
        test_data_iter = tokenized_datasets['test'].remove_columns(['id', 'headline', 'body'])

        logging.info(f"Creating Dataloaders")
        train_dataloader = DataLoader(train_data_iter, shuffle=True, batch_size=self.params["batch_size"], collate_fn=DataHandler.custom_collate_fn)
        test_dataloader = DataLoader(test_data_iter, shuffle=False, batch_size=self.params["batch_size"], collate_fn=DataHandler.custom_collate_fn)

        return train_dataloader, test_dataloader
    
def load_tokenizer(model_path):
    """Loads the tokenizer from hugging face checkpoint"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, return_tensors='pt', use_fast=False)
    return tokenizer

def load_data(data_path):
    """Reads the csv from the given path and creates dataset dicts"""
    logging.info(f"Reading Data from the path {data_path}")
    click_bait_data = pd.read_csv(data_path)
    train_data, test_data = train_test_split(click_bait_data, test_size=0.20, random_state = custom_random_seed, stratify=click_bait_data['label'])
    
    datasets_train_test = DatasetDict({
    "train": Dataset.from_pandas(train_data),
    "test": Dataset.from_pandas(test_data),
    })
    return datasets_train_test

def train(dataloader, model, optimizer, loss_fn, metrics_fn, epoch, device):
    """Trains the given model using the inputs from dataloader
    
    Returns:
        - dataloader: A dataset loader, it creates batches.
        - model: A Cross Encoder Model
        - optimizer: To perform gradient descent.
        - loss_fn: Calculate loss
        - metrics_fn: Calculate metrics
        - epoch: current epoch
        - device: cuda or cpu
    """
    logging.info(f"Training the model")
    model.train()
    log_interval = 100
    for idx, batch in enumerate(dataloader):

        # Gather Features and Labels
        idx_head = batch['input_ids_seq1']
        idx_head  = idx_head.to(device)

        idx_body = batch['input_ids_seq2']
        idx_body  = idx_body.to(device)

        label =  batch['label']
        label = label.to(device)

        ## Run Model
        predicted_label = model(idx_head, idx_body)
        loss = loss_fn(predicted_label, label)
        [tn,fp,fn,tp], pres, recall, f1, acc_metric = get_cm_metrics(metrics_fn(predicted_label.argmax(1), label))

        ## Propagate Loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        if idx % log_interval == 0:
            loss = loss.item()
            mlflow.log_metric("train_loss", f"{loss:2f}", step=idx)
            mlflow.log_metric("train_accuracy", f"{acc_metric:2f}", step=idx)
            mlflow.log_metric("train_pres", f"{pres:2f}", step=epoch)
            mlflow.log_metric("train_recall", f"{recall:2f}", step=epoch)

            logging.info("| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f} "
                "| f1-score {:8.3f} "
                "| loss {:8.3f}".format(
                    epoch, idx, len(dataloader), 
                    acc_metric,
                    f1,
                    loss
                ))

def evaluate(dataloader, model, loss_fn, metrics_fn, device, epoch):
    """Evaluates the given model using the inputs from dataloader
    
    Returns:
        - dataloader: A dataset loader, it creates batches.
        - model: A Cross Encoder Model
        - loss_fn: Calculate loss
        - metrics_fn: Calculate metrics
        - device: cuda or cpu
        - epoch: current epoch
    """
    logging.info(f"Evaluating the model")
    model.eval()
    total_loss = torch.tensor([]).to(device)
    total_predictions = torch.tensor([]).to(device)
    total_actuals = torch.tensor([]).to(device)
    
    with torch.no_grad():            
        for idx, batch in enumerate(dataloader):
            idx_head = batch['input_ids_seq1']
            idx_head  = idx_head.to(device)

            idx_body = batch['input_ids_seq2']
            idx_body  = idx_body.to(device)

            label =  batch['label']
            label = label.to(device)
            
            predicted_label = model(idx_head, idx_body)

            loss = loss_fn(predicted_label, label)
            total_loss = torch.cat([total_loss, torch.tensor([loss]).to(device)])

            total_predictions = torch.cat([total_predictions, predicted_label.argmax(1)])
            total_actuals = torch.cat([total_actuals, label])
                
    [tn,fp,fn,tp], pres, recall, f1, acc_metric = get_cm_metrics(metrics_fn(total_predictions, total_actuals))
    avg_loss = torch.mean(total_loss)
    mlflow.log_metric("eval_loss", f"{avg_loss.cpu().data.numpy():2f}", step=epoch)
    mlflow.log_metric("eval_accuracy", f"{acc_metric:2f}", step=epoch)
    mlflow.log_metric("eval_pres", f"{pres:2f}", step=epoch)
    mlflow.log_metric("eval_recall", f"{recall:2f}", step=epoch)

    logging.info("| epoch {:3d} "
                "| accuracy {:8.3f} "
                "| f1-score {:8.3f} "
                "| loss {:8.3f}".format(
                    epoch, 
                    acc_metric,
                    f1,
                    avg_loss
                ))

def get_cm_metrics(confusion_matrix):
    """Calculate classification metrics from the given confusion matrix array."""
    tp = confusion_matrix[1][1]
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]

    pres = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((pres*recall)/(pres+recall))
    acc = (tp+tn) / (tp+tn+fp+fn)
    
    if pres.isnan(): pres = torch.tensor(0)
    if recall.isnan(): recall = torch.tensor(0)
    if f1.isnan(): f1 = torch.tensor(0)
    if acc.isnan(): acc = torch.tensor(0)

    return [tn,fp,fn,tp], pres, recall, f1, acc

def print_cf(tn,fp,fn,tp):
    print(f"""         Pred -ve       Pred +ve
Act -ve  {tn}              {fp} 
Act +ve  {fn}              {tp}""")
 
def load_yaml(config_file_name):
    """Read the yaml file"""
    logging.info(f"Reading yaml from {config_file_name}")
    with open(config_file_name, 'r') as file:
        config = yaml.safe_load(file)
    return config
