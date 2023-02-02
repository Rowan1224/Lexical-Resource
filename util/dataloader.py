from dataclasses import dataclass
import torch
import numpy as np
from tokenizers import AddedToken
import re

@dataclass
class PreDataCollator:
    
    def __init__(self, tokenizer, max_len, tags_to_ids, Set='train'):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tags_to_ids = tags_to_ids
        self.Set = Set
     
    
    def __call__(self, batch):
        
        input_ids = []
        attention_mask = []
        labels = []
        sents = []
        ids = []
        lengths = []
        
        for sent, tag, id, length in zip(batch['sent'], batch['labels'], batch['ID'], batch['length']):

            
            tokenized = self.tokenize(sent, tag, length)
            input_ids.append(tokenized['input_ids'])
            attention_mask.append(tokenized['attention_mask'])
            labels.append(tokenized['labels'])

            sents.append(sent)
            ids.append(id)
            lengths.append(length)
     
        
        batch = {'input_ids':input_ids,'attention_mask':attention_mask, 'labels': labels, 'sents': sents, 'ID': ids, 'length': lengths}
        

        return batch

    def tokenize(self, sentence, tags, label_count):
        
        # getting the sentences and word tags
        
        sentence = sentence.strip().split() 
        pattern = re.compile("\ufffd|\u200e|\u200b\u200b|\u200b|\u200c|\u200f|\xad|\u0654|\u0652|\u0651|\u0650|\u0657|\u0656|\u064e|\u064b|\u0670|\u064f|\u064f",re.UNICODE)
        sentence = [pattern.sub('-',e) for e in sentence]
        sentence = [e.replace('-','') if len(e)>1 else e for e in sentence]
         

        # using tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides "return_offsets_mapping" functionality for individual tokens, so we know the start and end of a token divided into subtokens
        encoding = self.tokenizer(sentence,
                             is_split_into_words=True, 
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)

        # creating token tags only for first word pieces of each tokenized word
        if self.Set!='test':
            word_tags = tags.split()
            tags = [self.tags_to_ids[tag] for tag in word_tags]
            # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
            # creating an empty array of -100 of length max_length
            encoded_tags = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

            # setting only tags whose first offset position is 0 and the second is not 0
            
            i = 0
            for idx, mapping in enumerate(encoding["offset_mapping"]):
                if mapping[0] == 0 and mapping[1] != 0 and encoding['input_ids'][idx]!=6:
                    # overwrite the tag
                    try:
                        if i>= len(word_tags):
                            continue
                        encoded_tags[idx] = tags[i]
                        i += 1
                    except:
    #                     print(encoding["offset_mapping"])
    #                     print(i)
                        print(sentence)
    #                     print(len(tags), tags[i])
                

            
            
            
        else:
            encoded_tags = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
            i = 0
            for idx, mapping in enumerate(encoding["offset_mapping"]):
                if mapping[0] == 0 and mapping[1] != 0 and encoding['input_ids'][idx]!=6:
                    # overwrite the tag
                    try:
                        if i>= label_count:
                            continue
                        encoded_tags[idx] = 0
                        i += 1
                    except:
    #                     print(encoding["offset_mapping"])
    #                     print(i)
                        print(sentence)
    #                     print(len(tags), tags[i])

        # turning everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_tags)

        return item
