#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
sys.path.append('../')
import pandas as pd
import torch 
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import random
from transformers import DataCollatorForTokenClassification
import evaluate
from util.utils import get_tag_mappings, get_data, compute_metrics
from util.dataloader import PreDataCollator
from util.args import create_arg_parser
from helper import prepare_data
os.environ["WANDB_DISABLED"] = "true"


# ### Env Setup

# In[ ]:


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print("device = ", device)
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

# In[ ]:
args = create_arg_parser()

LANG = args.language # use None for all lang
MAX_LEN = 256
TOKENIZER_NAME = args.model
MODEL_NAME = args.model
if args.set == "None":
    set = None
else:
    set = args.set
AUG = set # or 'tags' or 'LM' or None

output_dir = f"./output/{MODEL_NAME}-{LANG}-{AUG}" if AUG!=None else f"./output/{MODEL_NAME}-{LANG}"


# ### Preparing data

# In[ ]:


def get_data(lang, data_split, AUG):
    
    augmented_data_dir = '../Data-Processing/Augmented-Dataset'
    conll_data_dir = '../Dataset'
    if AUG=='LM' or AUG=='tags':
        filename = f'{augmented_data_dir}/{lang}-{data_split}-{AUG}.csv' if SET!=None else f'{augmented_data_dir}/{lang}-{data_split}.csv'
        data = pd.read_csv(filename)
        data['length'] = data.sent.apply(lambda x:len(x.split()))
        df = data.drop(columns=['sent'])
        df = test_df.rename(columns={'augmented_sen':'sent'})
        data = Dataset.from_pandas(df)

    else:
        filename = f'{conll_data_dir}/{lang}/{lang}_{data_split}.conll'
        data = prepare_data(filename)
        data['length'] = data.sent.apply(lambda x:len(x.split()))
        data = Dataset.from_pandas(data)
        
    return data


# In[ ]:


## Transform into hugginface dataset
train_data = get_data(LANG, 'train', AUG)
dev_data = get_data(LANG, 'dev', AUG)


# In[ ]:


# Check random data item

print(train_data[0]['sent'])
print(train_data[0]['labels'])


# ### Tokenization

# In[ ]:


# getting the tags
tags_to_ids, ids_to_tags = get_tag_mappings()
number_of_labels = len(tags_to_ids)


# In[ ]:


## load appropiate tokenizer for pre-trained models

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


# In[ ]:


EPOCHS = 7
LEARNING_RATE = args.learning_rate
TRAIN_BATCH_SIZE = args.batch_size
VALID_BATCH_SIZE = args.batch_size
SAVE_STEPS = args.save_steps
EVAL_STEPS = 500
SAVE_LIMIT = 2
WARMUP_STEPS = 100


# In[ ]:


data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors='pt')


# In[ ]:


from transformers import TrainingArguments

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


# In[ ]:


from transformers import Trainer


trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    tokenizer=tokenizer
)


# In[ ]:


# If you want to continue training from a checkpoint
# CHECKPOINT = 2500
# chkpt_model = f'{output_dir}/checkpoint-{CHECKPOINT}'
# trainer.train(chkpt_model)


# In[ ]:


trainer.train()


# In[ ]:


trainer.save_model(f"{output_dir}/Final")

