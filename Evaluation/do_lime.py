#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import lime
import torch
import torch.nn.functional as F
from helper import prepare_data
from lime.lime_text import LimeTextExplainer
from torch import cuda
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re
import os
from util.args import create_arg_parser
args = create_arg_parser()
# In[3]:


class NERExplainerGenerator(object):
    
    def __init__(self, model_dir, number_of_labels, device):
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=number_of_labels)
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        
    def clean(self, sent):
        
        sentence = sent.strip().split() 
        pattern = re.compile("\ufffd|\u200e|\u200b\u200b|\u200b|\u200c|\u200f|\xad|\u0654|\u0652|\u0651|\u0650|\u0657|\u0656|\u064e|\u064b|\u0670|\u064f|\u064f",re.UNICODE)
        sentence = [pattern.sub('-',e) for e in sentence]
        sentence = [e.replace('-','') if len(e)>1 else e for e in sentence]
        
        return sentence
    
    def tokenize(self, sent):
        
        tokenized = self.tokenizer(sent,
         is_split_into_words=True, 
         return_offsets_mapping=True, 
         padding='max_length', 
         truncation=True, 
         max_length=256)
        
        
        return tokenized
    

    def get_predict_function(self, word_index, batch_size = 4):
        def predict_func(texts):
            
            tokenized = [self.tokenize(self.clean(text)) for text in texts]
            
            probas = None
            for i in range(0,len(tokenized),batch_size):
                
                if i+batch_size > len(tokenized):
                    j = len(tokenized)
                else:
                    j = i+batch_size
                    
                batch = tokenized[i:j]
                
                inp_ids = torch.as_tensor([b['input_ids'] for b in batch]).to(device)
                mask = torch.as_tensor([b['attention_mask'] for b in batch]).to(device)
                logits = self.model(input_ids=inp_ids, attention_mask=mask).logits
                probas_batch = F.softmax(logits, dim=-1).detach().cpu().numpy()
                
                if probas is None:
                    probas = probas_batch
                else:
                    probas = np.vstack((probas,probas_batch))

#             batch = [self.tokenize(self.clean(text)) for text in texts]
#             inp_ids = torch.as_tensor([b['input_ids'] for b in batch]).to(device)
#             mask = torch.as_tensor([b['attention_mask'] for b in batch]).to(device)
#             logits = self.model(input_ids=inp_ids, attention_mask=mask).logits
#             probas = F.softmax(logits, dim=-1).detach().numpy()         

            print(probas.shape)
            return probas[:,word_index,:]
        
        return predict_func


# In[4]:


tags = ['_','I-PER','I-LOC', 'B-PROD', 'B-PER', 'B-LOC','I-CORP', 'I-CW', 'I-PROD', 'B-CW','I-GRP','O','B-CORP','B-GRP']
device = 'cuda' if cuda.is_available() else 'cpu'
print("running on", device)
# device = 'cpu'


# In[5]:


def get_token_idx(sent, labels, tags):
    tokenized = model.tokenize(sent.split())
    offset = tokenized['offset_mapping']
    index = [i for i,(a,b) in enumerate(offset) if a==0 and b!=0 and tokenized['input_ids'][i]!=6]
    tag_to_id = {t:i for i,t in enumerate(tags)}
    labels = labels.split()
    wordIds_to_tokenidx = [(ti,tag_to_id[labels[wi]]) for wi,ti in enumerate(index)]
    
    return wordIds_to_tokenidx
    


# In[19]:


def explain(model, explainer, tags, data, idx):
    
    original_sent = data.iloc[idx].sent
    augmented_sent = data.iloc[idx].augmented_sen
    labels = data.iloc[idx].predictions
    
    ids = get_token_idx(original_sent, labels, tags)
    
    for i, (word_index, label_index) in enumerate(ids):
        
        func = model.get_predict_function(word_index)
        
        exp = explainer.explain_instance(augmented_sent, func, 
                                         num_features=args.num_features, num_samples=args.num_samples, labels=(label_index,))
        NUM_SAMPLES = args.num_samples
        dir = './visualizations'
        if not os.path.exists(dir):
            os.mkdir(dir)
        if not os.path.exists(f'{dir}/{LANG}_{SET}_{NUM_SAMPLES}_{str(idx)}'):
            os.mkdir(f'{dir}/{LANG}_{SET}_{NUM_SAMPLES}_{str(idx)}')   
        
        
        filename = f'{dir}/{LANG}_{SET}_{NUM_SAMPLES}_{str(idx)}/{original_sent.split()[i]}.html'
        exp.save_to_file(filename, text=augmented_sent)


# In[1]:


LANG = args.language # use None for all lang
MAX_LEN = 256
MODEL_NAME = 'xlm-roberta-large'
if args.set == "None":
    SET = None
else:
    SET = args.set # 'LM' or None
EVAL_SET = args.dataset
augmented_data_dir = '../Data-Processing/Augmented-Dataset'
conll_data_dir = '../Dataset'


# In[16]:


if SET=='LM' or SET=='tags':
    filename = f'{augmented_data_dir}/{LANG}-{EVAL_SET}-{SET}.csv'
    data = pd.read_csv(filename)
    output = pd.read_csv(f'./{LANG}/{EVAL_SET}/outputs-{MODEL_NAME}-{SET}.csv')
    data['predictions'] = output['predictions'].array
    
else:
    filename = f'{conll_data_dir}/{LANG}/{LANG}_{EVAL_SET}.conll'
    data = prepare_data(filename)
    output = pd.read_csv(f'./{LANG}/{EVAL_SET}/outputs-{MODEL_NAME}.csv')
    data['predictions'] = output['predictions'].array


# In[18]:

if args.language == "zh":
    model = NERExplainerGenerator(f'../Experiment/output/{MODEL_NAME}-{LANG}-{SET}/Final', len(tags), device)
else:
    if SET == None:
        model = NERExplainerGenerator(f'../Experiment/output/{MODEL_NAME}-{LANG}', len(tags), device)
    else:
        model = NERExplainerGenerator(f'../Experiment/output/{MODEL_NAME}-{LANG}-{SET}', len(tags), device)
explainer = LimeTextExplainer(class_names=tags, random_state=42)


# In[21]:


explain(model, explainer, tags, data, args.index)


# In[22]:


data.iloc[173299]


# In[ ]:




