import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json
import pickle
from datahandler import DataHandler
from train_and_evaluate import Trainer
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Transformer


from torch.utils.tensorboard import SummaryWriter


# Create an instance of the data handler
data_handler = DataHandler()

# Load the dataset from the file and get the tokenized datasets
tokenized_datasets = data_handler.load_dataset('spa.csv')

# Prepare the data loaders
train_loader, valid_loader = data_handler.prepare_dataloader(tokenized_datasets)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

max_lens = [512]
model_sizes =[128]
n_heads =[4]
n_layers =[4]
dropout_prob = [0]
epochs = 1


results = {}

for max_len in max_lens:
    for model_size in model_sizes:
        for n_head in n_heads:
            for n_layer in n_layers:
                for dropout in dropout_prob:

                    key = '_'.join(str(x) for x in (max_len,model_size,n_head,n_layer,dropout))
                    print("Running experiment with configuration:", key)



                    encoder = Encoder(vocab_size=data_handler.tokenizer.vocab_size+1,
                                    max_len = max_len,
                                    d_k = int(model_size//n_head),
                                    d_model = model_size,
                                    n_heads = n_head,
                                    n_layers = n_layer,
                                    dropout_prob = dropout)
                    encoder.to(device)

                    decoder = Decoder(vocab_size=data_handler.tokenizer.vocab_size+1,
                                    max_len =max_len,
                                    d_k=int(model_size//n_head),
                                    d_model=model_size,
                                    n_heads=n_head,
                                    n_layers=n_layer,
                                    dropout_prob=dropout)
                    decoder.to(device)

                    transformer = Transformer(encoder, decoder)

                    criterion = torch.nn.CrossEntropyLoss(ignore_index = -100)
                    optimizer = torch.optim.Adam(transformer.parameters())

                    trainer = Trainer(transformer, criterion, optimizer, device, data_handler.tokenizer, train_loader, valid_loader, key)
                    train_losses, validation_losses, train_perplexity_list, validation_perplexity_list = trainer.train(epochs=epochs)

                    results[key] = {
                        "train_losses_list":train_losses,
                        "validation_losses_list":validation_losses,
                        "model_size": model_size,
                        # "d_k": d_k,
                        "max_len": max_len,
                        "n_head": n_head,
                        "n_layer": n_layer,
                        "dropout": dropout,
                    }