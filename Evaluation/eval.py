

import sys
import os
sys.path.append('../')
import pandas as pd
import torch 
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
from tqdm import tqdm
import random
from datasets import Dataset
from util.utils import get_tag_mappings
from util.dataloader import PreDataCollator
import nltk
nltk.download('punkt')
os.environ["WANDB_DISABLED"] = "true"
from helper import prepare_data
from util.args import create_arg_parser
from torch.utils.data import DataLoader
from util.utils import compute_metrics_test
from torch import cuda
from sklearn.metrics import classification_report
from operator import add
from functools import reduce
device = 'cuda' if cuda.is_available() else 'cpu'


### Seed all

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# data directory
augmented_data_dir = '../Data-Processing/Augmented-Dataset'
conll_data_dir = '../Dataset'

def main():

    args = create_arg_parser()
    

    LANG = args.language# use None for all lang
    MAX_LEN = 256
    TOKENIZER_NAME = args.model
    MODEL_NAME = args.model
    if args.set == "None":
        SET = None
    else:
        SET = args.set
    EVAL_SET = args.dataset
    

    model_dir = f"../Experiment/output/{MODEL_NAME}-{LANG}-{SET}/Final" if SET!=None else f"../Experiment/output/{MODEL_NAME}-{LANG}/Final"



    if SET=='LM' or SET=='tags':
        filename = f'{augmented_data_dir}/{LANG}-{EVAL_SET}-{SET}.csv'
        data = pd.read_csv(filename)
        data = data.dropna()
        data['length'] = data.sent.apply(lambda x:len(x.split()))
        test_df = data.drop(columns=['sent'])
        test_df = test_df.rename(columns={'augmented_sen':'sent'})
        test_data = Dataset.from_pandas(test_df)
        
    else:
        filename = f'{conll_data_dir}/{LANG}/{LANG}_{EVAL_SET}.conll'
        data = prepare_data(filename)
        data['length'] = data.sent.apply(lambda x:len(x.split()))
        test_data = Dataset.from_pandas(data)




    tags_to_ids, ids_to_tags = get_tag_mappings()
    number_of_labels = len(tags_to_ids)


    ## load appropiate tokenizer for pre-trained models
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    collator = PreDataCollator(tokenizer=tokenizer, max_len=MAX_LEN, tags_to_ids = tags_to_ids, Set= EVAL_SET)


    test_tokenized = test_data.map(collator, remove_columns=test_data.column_names, batch_size=8, num_proc=8, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=number_of_labels)
    model = model.to(device)



    dataloader = DataLoader(test_tokenized, batch_size=args.batch_size)
    outputs = []
    preds = []
    true = []
    for batch in tqdm(dataloader):

        inp_ids = torch.stack(batch["input_ids"], axis=1).to(device)
        label_ids = torch.stack(batch["labels"], axis=1).to(device)
        mask = torch.stack(batch["attention_mask"], axis=1).to(device)
        logits = model(input_ids=inp_ids, attention_mask=mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        for i in range(inp_ids.shape[0]):
            tags, predicts = compute_metrics_test(pred_ids[i], label_ids[i])
            pred_tags = [ids_to_tags[idx] for idx in predicts if idx!=-100]
            true_tags = [ids_to_tags[idx] for idx in tags if idx!=-100]
            preds.extend(pred_tags)
            true.extend(true_tags)
            outputs.append((batch['ID'][i],batch['sents'][i], pred_tags, true_tags))


    dir = f'./{LANG}/{EVAL_SET}'
    if not os.path.exists(f'./{LANG}'):
        os.makedirs(f'./{LANG}')
    if not os.path.exists(dir):
        os.makedirs(dir)


    predictions = pd.DataFrame(outputs, columns=['ID','sent','predictions','true'])
    predictions['predictions'] = predictions['predictions'].apply(lambda x: " ".join(x))
    predictions['true'] = predictions['true'].apply(lambda x: " ".join(x))

    filename = MODEL_NAME.split('/')[-1]
    fileCsv = f'{dir}/outputs-{filename}-{SET}.csv' if SET!=None else f'{dir}/outputs-{filename}.csv'
    predictions.to_csv(fileCsv,index=False)


    # true = [label.strip().split() for label in test_data['labels']]
    # preds = predictions.predictions.array
    # predictions['true'] = true
    # preds = reduce(add, preds)
    # true = reduce(add, true)


    
    print(classification_report(preds, true, output_dict=True)['macro avg']['f1-score'])



if __name__ == '__main__':
    main()
