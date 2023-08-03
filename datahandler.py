import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

class DataHandler:
    def __init__(self, model_checkpoint='Helsinki-NLP/opus-mt-en-es', 
                 max_input_length=128, max_target_length=128):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.tokenizer.add_special_tokens({"cls_token": "<s>"})
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def load_dataset(self, filepath):
        # df = pd.read_csv(filepath, sep="\t", header=None)
        # df = df.iloc[:100000]
        # df.columns = ['en', 'es']
        # df.to_csv('spa1lakh.csv', index=None)
        
        
        raw_dataset = load_dataset('csv', data_files='spa.csv')
        
        split = raw_dataset['train'].train_test_split(test_size=0.3, seed=42)
        tokenized_datasets = split.map(
            self.preprocess_function, batched=True,
            remove_columns=split["train"].column_names,
        )
        return tokenized_datasets

    def preprocess_function(self, batch):
        model_inputs = self.tokenizer(
            batch['en'], max_length=self.max_input_length, truncation=True)
        labels = self.tokenizer(
            batch['es'], max_length=self.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def prepare_dataloader(self, tokenized_datasets, batch_size=32):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        train_loader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=data_collator
        )
        valid_loader = DataLoader(
            tokenized_datasets["test"],
            batch_size=batch_size,
            collate_fn=data_collator
        )
        return train_loader, valid_loader
