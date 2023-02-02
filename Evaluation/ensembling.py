#!/usr/bin/env python
# coding: utf-8

# In[52]:


import sys
sys.path.append('../')
import pandas as pd
from operator import add
from functools import reduce
from sklearn.metrics import classification_report
import os
from collections import Counter
import gzip
import shutil
from tqdm import tqdm


# In[2]:

def recreate_conll_format_preds(data, col='labels'):


    lines = data[col].split()
    
    return "\n".join(lines)


def write_conll_format_preds(fileName, dataframe, col='labels'):

    with open(fileName,'w') as file:
        for ind in range(dataframe.shape[0]):
            data = dataframe.iloc[ind]
            lines = recreate_conll_format_preds(data, col)
            file.write("\n"+lines+"\n\n")


import re 
def clean(text):
    p = re.compile('"|,|\[|\]|')
    cleaned = p.sub('',text)
    cleaned= cleaned.replace("'", "")
    return cleaned.split()




def model_ranking(model_scores):
    model_scores = sorted(model_scores.items(), key=lambda x:x[1])
    return {key_value[0]:rank for rank,key_value in enumerate(model_scores)}


def ensemble_majority_rank(x, scores):
    
    tags = x.values.tolist()
    model_tags_map = x.to_dict()
    scores = model_ranking(scores)
    votes = {}
    for model, tag in model_tags_map.items():
        try:
            votes[tag] += scores[model]
        except:
            votes[tag] = scores[model]
    
    mx = 0
    winner = 'O'
    for key, val in votes.items():
        if val>mx:
            winner = key
            mx = val
        
    return winner




def ensemble_majority_weighted(x, scores):
    
    tags = x.values.tolist()
    model_tags_map = x.to_dict()
    votes = {}
    for model, tag in model_tags_map.items():
        try:
            votes[tag] += scores[model]
        except:
            votes[tag] = scores[model]
    
    mx = 0
    winner = 'O'
    for key, val in votes.items():
        if val>mx:
            winner = key
            mx = val
        
    return winner


def ensemble_majority(x, scores):
    
#     print(x)
    votes = Counter(x.values.tolist())
    mx = 0
    winner = 'O'
    for key, val in votes.items():
        if val>mx:
            winner = key
            mx = val
        
    return winner




def merge_df(lang, eval_set):
    
    dir = f'./Test-Dev results/{lang}/{eval_set}'
    files = os.listdir(dir)
    files = [f for f in files if f.endswith('.csv')]
    dfs = [pd.read_csv(os.path.join(dir,f))  for f in files]
    
    merged = pd.DataFrame()
    for i in range(len(dfs)):
        col = f'preds{i}'
        merged[col] = dfs[i].predictions
        merged[col] = merged[col].apply(lambda x: x.split())
    
    if eval_set=='dev':
        merged['true'] = dfs[0].true
        merged['true'] = merged['true'].apply(lambda x: clean(x))
        true = reduce(add, merged['true'])
    else:
        true = None
    preds = [reduce(add, merged['preds'+str(i)].array) for i in range(len(dfs))]
    
    
    df = pd.DataFrame(preds)
    df = df.T
    
    cols = [f.replace('outputs-','').replace('.csv','') for f in files]
    df.columns = cols

    return df, true, cols




def merge_df_test(file):
    
    dir = f'./Test-Dev results/{lang}/{eval_set}'
    files = os.listdir(dir)
    files = [f for f in files if f.endswith('.csv')]
    dfs = [pd.read_csv(os.path.join(dir,f))  for f in files]
    
    merged = pd.DataFrame()
    for i in range(len(dfs)):
        col = f'preds{i}'
        merged[col] = dfs[i].predictions
        merged[col] = merged[col].apply(lambda x: x.split())
    
    preds = [reduce(add, merged['preds'+str(i)].array) for i in range(len(dfs))]
    
    df = pd.DataFrame(preds)
    df = df.T
    
    cols = [f.replace('outputs-','').replace('.csv','') for f in files]
    df.columns = cols

    return df, None, cols




def best_f1(df, true, cols):
    best = 0
    scores = {}
    for col in cols:
        f1 = classification_report(df[col], true, output_dict=True,zero_division=1)['macro avg']['f1-score']
        scores[col] = f1

    return scores



def ensemble(dataframe, true, func, scores, eval_set='test'):
    
    ens = dataframe.apply(lambda x: func(x, scores), axis=1)
    
    if eval_set=='dev':
        f1 = classification_report(ens, true, output_dict=True,zero_division=1)['macro avg']['f1-score']
    else:
        f1 = None
    
    return ens, f1



def get_dev_scores(lang):
    

    df, true, dfs = merge_df(lang, 'dev')
    
    scores = Counter(best_f1(df, true, dfs))
    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    
    model, best = scores[0]
    best = best*100
    
    scores = {model:f1 for model,f1 in scores}
    
    return scores, model, best


langs = ['EN', 'BN', 'ZH', 'FR', 'FA', 'ES', 'DE', 'HI', 'IT', 'PT', 'SV', 'UK']
langs = [s.lower() for s in langs]




finals = {}
eval_set = 'test'
for lang in langs:
    
    print(lang)
    
    scores, model, best = get_dev_scores(lang)
    
    
    df, true, dfs = merge_df(lang, eval_set)
    print("Done Loading")
    
    majority, ef1 = ensemble(df, true, ensemble_majority, scores, eval_set)
    print("Done majority")
    weighted, ef2 = ensemble(df, true, ensemble_majority_weighted, scores, eval_set)
    print("Done weighted")
    rank, ef3 = ensemble(df, true, ensemble_majority_rank, scores,  eval_set)
    print("Done rank")
# In[3]:
    
    if eval_set=='dev':
        print(f'{lang}: best_f1 = {best*100:.2f} majority: {ef1*100:00.2f} weighted: {ef2*100:00.2f} rank: {ef3*100:00.2f}')
    
    df['majority'] = majority
    df['weighted'] = weighted
    df['rank'] = rank
    
    finals[lang] = df
#     break



for lang in langs:
    
    if not os.path.exists(f'./Submission/{lang}'):
        os.makedirs(f'./Submission/{lang}')
        
    base = "./Submission/{LANG}"
        
    predictions = finals[lang]
    for model in ['majority', 'weighted', 'rank', 'best']:
            dir = f'{base}/{model}'
            if not os.path.exists(dir):
                os.makedirs(dir)
                
            
            fileConll = f'{dir}/{lang}.pred.conll'
            fileZip = f'{dir}/my_submission.zip'
            write_conll_format_preds(fileConll, predictions, col=model)
            

            with open(fileConll, 'rb') as f_in:
                with gzip.open(fileZip, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            

        
        

        


