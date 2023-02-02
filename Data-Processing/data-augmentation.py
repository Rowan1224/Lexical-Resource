#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive/')
# %cd "/content/drive/MyDrive/Colab Notebooks/SemEval2023/Evaluation"


# In[ ]:


#get_ipython().system('pip install transformers')
#get_ipython().system('pip install datasets')
#get_ipython().system('pip install evaluate')
#get_ipython().system('pip install colorama')
#get_ipython().system('pip install wikipedia-api')
#get_ipython().system('pip install sentencepiece')


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

LANG = args.language # use None for all lang
SET = 'LM' # 'LM' or None
EVAL_SET = 'test'


# In[ ]:


if LANG=='en' and SET=='LM':
    get_ipython().system('python -m spacy download en_core_web_lg')


# ## Read Data

# In[ ]:


filename = f'../Dataset/{LANG}-{EVAL_SET}.conll'
data = prepare_data(filename)


# In[ ]:


data.head()


# ## Augment Info

# In[ ]:


from InformationExtraction import InformationExtractionPipeline
infoPipeline = InformationExtractionPipeline(SET,
                                        max_sen = 2, lang = LANG, 
                                        loadJson = True, jsonPath=f'./Wiki/{LANG}-wiki.json',
                                            saveJson=True, saveJsonpath = f'./Wiki/{LANG}-test-wiki.json')


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
data.to_csv(f'{dir}/{LANG}-test.csv',index=False)

