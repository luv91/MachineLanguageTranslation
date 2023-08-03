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
# import matplotlib.pyplot as plt
from model import Encoder, Decoder, Transformer
from datahandler import DataHandler
from train_and_evaluate import Trainer
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu


# Create an instance of the data handler
data_handler = DataHandler()

# Load the dataset from the file and get the tokenized datasets
tokenized_datasets = data_handler.load_dataset('spa.csv')

# Prepare the data loaders
train_loader, valid_loader = data_handler.prepare_dataloader(tokenized_datasets)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the checkpoint
checkpoint_path = "checkpoint_nlp_latest_epoch.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
                    
max_lens = [512]
model_sizes =[128]
n_heads =[4]
n_layers =[4]
dropout_prob = [0.1]
epochs = 1000


results = {}

for max_len in max_lens:
    # for d_k in d_ks:
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
                    
                    transformer.load_state_dict(checkpoint)
                    transformer.to(device)
                    transformer.eval() # Set the model to evaluation mode if you are going to do inference


def translate(input_sentence, transformer, encoder, decoder, data_handler, device):
    enc_input = data_handler.tokenizer(input_sentence, return_tensors='pt').to(device)
    enc_output = encoder(enc_input['input_ids'], enc_input['attention_mask'])

    dec_input_ids = torch.tensor([[data_handler.tokenizer.cls_token_id]], device=device)
    dec_attn_mask = torch.ones_like(dec_input_ids, device=device)

    for _ in range(32):
        dec_output = decoder(
            enc_output,
            dec_input_ids,
            enc_input['attention_mask'],
            dec_attn_mask
        )
        
        prediction_id = torch.argmax(dec_output[:, -1, :], axis=-1)

        dec_input_ids = torch.hstack((dec_input_ids, prediction_id.view(1, 1)))
        dec_attn_mask = torch.ones_like(dec_input_ids)

        if prediction_id == 0:
            break

    translation = data_handler.tokenizer.decode(dec_input_ids[0, 1:])
    print(translation)
    return translation

# Test the translation function
translate("thank you", transformer, encoder, decoder, data_handler, device)


from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(input_sentence, reference_sentence, transformer, encoder, decoder, data_handler, device):
    # Translate the input sentence
    translated_sentence = translate(input_sentence, transformer, encoder, decoder, data_handler, device)
    
    # Tokenize both the translated sentence and the reference sentence
    translated_tokens = data_handler.tokenizer.tokenize(translated_sentence)
    reference_tokens = data_handler.tokenizer.tokenize(reference_sentence)
    
    # Compute the BLEU score
    bleu_score = sentence_bleu([reference_tokens], translated_tokens)
    
    return bleu_score

input_sentence = "thank you"
reference_sentence = "gracias" # the expected translation
bleu = calculate_bleu(input_sentence, reference_sentence, transformer, encoder, decoder, data_handler, device)
print("BLEU score:", bleu)

# Read the CSV file
file_path = 'spa1lakh.csv'
df = pd.read_csv(file_path)

# Variables to hold the total BLEU score and the number of sentences
total_bleu = 0
num_sentences = 100

# Loop through the first num_sentences rows of the CSV file
for index, row in df.head(num_sentences).iterrows():
    input_sentence = row['en']
    reference_sentence = row['es']
    
    # Translate the input sentence using the trained model
    translated_sentence = translate(input_sentence, transformer, encoder, decoder, data_handler, device)
    
    # Tokenize both the translated sentence and the reference sentence
    translated_tokens = data_handler.tokenizer.tokenize(translated_sentence)
    reference_tokens = data_handler.tokenizer.tokenize(reference_sentence)
    
    # Compute the BLEU score for this translation
    bleu_score = sentence_bleu([reference_tokens], translated_tokens)
    
    # Accumulate the total BLEU score
    total_bleu += bleu_score

# Calculate the average BLEU score over all the sentences
average_bleu = total_bleu / num_sentences

print("Average BLEU score over 100 sentences:", average_bleu)