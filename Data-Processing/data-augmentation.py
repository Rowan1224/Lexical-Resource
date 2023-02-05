#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive/')
# %cd "/content/drive/MyDrive/Colab Notebooks/SemEval2023/Evaluation"


# In[ ]:


# !pip install transformers
# !pip install datasets
# !pip install evaluate
# !pip install colorama
# !pip install wikipedia-api
# !pip install sentencepiece


# In[ ]:


import sys
import os
sys.path.append('../')
import pandas as pd
import torch 
import numpy as np
from tqdm import tqdm
import random
from datasets import Dataset
import nltk
nltk.download('punkt')
os.environ["WANDB_DISABLED"] = "true"
from helper import prepare_data
from util.args import create_arg_parser

# In[ ]:


def write_tags():
    tags = set()
    for labels in data.labels.array:
        for tag in labels.split():
            tags.add(tag)

    with open('../util/tags.txt','w') as file:
        lines = "\n".join(list(tags))
        file.write(lines)
    
    


# In[ ]:


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

### Seed all

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# In[ ]:
args = create_arg_parser()

LANG = args.language
SET = args.set
EVAL_SET = args.dataset


# In[ ]:


# if LANG=='en':
#     !python -m spacy download en_core_web_lg


# ## Read Data

# In[ ]:


filename = f'../Dataset/{LANG}/{LANG}_{EVAL_SET}.conll'
data = prepare_data(filename)


# In[ ]:


data = data.dropna()
data.shape


# In[ ]:


#write_tags()


# ## Augment Info

# In[ ]:


from InformationExtraction import InformationExtractionPipeline
infoPipeline = InformationExtractionPipeline(SET,
                                        max_sen = 2, lang = LANG, 
                                        loadJson = False, jsonPath=f'./Wiki/{LANG}-wiki.json',
                                            saveJson=True, saveJsonpath = f'./Wiki/{LANG}-wiki.json')


# In[ ]:


augmented = infoPipeline(data[['sent','labels']].values.tolist())


# In[ ]:


data['augmented_sen'] = augmented
temp = data[data['sent']!=data['augmented_sen']]
info_percent = temp.shape[0]/data.shape[0]
print(f"Info Percentage: {info_percent*100:.2f}%")


# In[ ]:


dir = f'./Augmented-Dataset'
if not os.path.exists(dir):
    os.mkdir(dir)
    
filename = f'{dir}/{LANG}-{EVAL_SET}-{SET}.csv'
data.to_csv(filename,index=False)

