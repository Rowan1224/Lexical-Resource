#!/usr/bin/env python
# coding: utf-8

import sys
import os
sys.path.append('../')
import pandas as pd
import torch 
from torch import cuda
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
from tqdm import tqdm
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import random
import evaluate
from util.utils import get_tag_mappings, get_data, compute_metrics
from util.args import create_arg_parser
from util.dataloader import PreDataCollator

os.environ["WANDB_DISABLED"] = "true"

device = 'cuda' if cuda.is_available() else 'cpu'

### Seed all

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ### Instructions
# 
# Set the variables in the next cell according to the experiment:
# 
# ``LANG``: Set the language. You can find the language codes in the excel file.
# 
# ``TOKENIZER_NAME`` or ``MODEL_NAME``: Huggingface Model link. Also mentioned in excel file. 
# 
# ``SET``: Select the dataset
# 
# - ``None`` --> **None Augmentation** (No Augmentation from wiki) NB: None is **not** a string value here
# - ``tags`` --> **Max Augmentation** (Maximum Augmentation from wiki)
# - ``LM`` --> **Entity Extractor** (Augmentation from wiki after extracting tags using other NER model)
#  
# ``IS_CRF``: True if you want to try the CRF model. Recommended to finish all non-CRF experiments first
# 
# 
# **Please ensure that you are saving the trained models**
# 
# [Link to Excel File](https://docs.google.com/spreadsheets/d/11LXkOBWxpWDGMsi9XC72eMNSJI14Qo2iwP8qugwjyqU/edit#gid=0)

# ### Define Variables



def main():
    
    args = create_arg_parser()

    LANG = args.language# use None for all lang
    MAX_LEN = 256
    TOKENIZER_NAME = args.model
    MODEL_NAME = args.model
    SET = args.set# or 'tags' or 'LM' or None
    IS_CRF = args.crf
    
    output_dir = f"./output/{MODEL_NAME}-{LANG}-{SET}" if SET!=None else f"./output/{MODEL_NAME}-{LANG}"
        
    
    # Load data as pandas dataframe
    
    df = get_data(LANG, SET, train=True)
        
    train_df, dev_df = train_test_split(df, test_size=0.2, random_state=SEED)
    
    
    if LANG!=None:
        train_df = train_df[train_df['lang']==LANG]
        dev_df = dev_df[dev_df['lang']==LANG]
    
    
    train_df['length'] = train_df.sent.apply(lambda x:len(x.split()))
    dev_df['length'] = dev_df.sent.apply(lambda x:len(x.split()))
    train_data = Dataset.from_pandas(train_df)
    dev_data = Dataset.from_pandas(dev_df)
    
    
    print(train_data[0]['sent'])
    print(train_data[0]['labels'])
    
    
    
    # getting the tags
    tags_to_ids, ids_to_tags = get_tag_mappings()
    number_of_labels = len(tags_to_ids)
    
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    
    
    # In[ ]:
    
    
    collator = PreDataCollator(tokenizer=tokenizer, max_len=MAX_LEN, tags_to_ids = tags_to_ids)
    
    
    # In[ ]:
    
    
    train_tokenized = train_data.map(collator, remove_columns=train_data.column_names, batch_size=4, num_proc=4, batched=True)
    
    
    # In[ ]:
    
    
    dev_tokenized = dev_data.map(collator, remove_columns=dev_data.column_names, batch_size=4, num_proc=4, batched=True)
    
    
    # ### Training
    
    # In[ ]:
    
    
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=number_of_labels)
        
    model = model.to(device)
    
    EPOCHS = 7
    LEARNING_RATE = args.learning_rate
    TRAIN_BATCH_SIZE = args.batch_size
    VALID_BATCH_SIZE = args.batch_size
    SAVE_STEPS = args.save_steps
    EVAL_STEPS = 500
    SAVE_LIMIT = 2
    WARMUP_STEPS = 100
    
    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors='pt')
    
    
    training_args = TrainingArguments(
      output_dir= output_dir,
      group_by_length=True,
      per_device_train_batch_size=TRAIN_BATCH_SIZE,
      gradient_accumulation_steps=2,
      evaluation_strategy="steps",
      num_train_epochs=EPOCHS,
      fp16=False,
      save_steps=SAVE_STEPS,
      eval_steps=EVAL_STEPS,
      logging_steps=EVAL_STEPS,
      learning_rate=LEARNING_RATE,
      warmup_steps=WARMUP_STEPS,
      save_total_limit=SAVE_LIMIT,
    )
    
    
    
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        tokenizer=tokenizer
    )
    
    
    trainer.train()
    
    
    trainer.save_model(f"{output_dir}/Final")

if __name__ == '__main__':
    main()
# %%
