import wikipediaapi
import nltk
import re
import json
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import spacy
import pandas as pd
#from langdetect import detect
#from langdetect import DetectorFactory
#DetectorFactory.seed = 0

class InformationExtractionPipeline:
    
    def __init__(self, method ='tags' , max_sen = 2, lang = 'en', loadJson = False, jsonPath=None ,saveJson=False, saveJsonpath=None):
        """
        It takes in a wikipedia extractor object, a maximum number of sentences to extract, a language, a
        boolean to load a json file, a path to the json file, and a boolean to save the json file
        
        :param extractor: the extractor object that we created in the previous section
        :param max_sen: the maximum number of sentences to extract from the wikipedia page, defaults to 2
        (optional)
        :param lang: language of the wikipedia page, defaults to en (optional)
        :param loadJson: if you have a json file with the wikipedia data, you can load it here, defaults to
        False (optional)
        :param jsonPath: the path to the json file that contains the wikipedia data
        :param saveJson: if True, the json file will be saved in the current directory, defaults to False
        (optional)
        """


        
        self.model = self.load_model(lang, method)
        self.max_sen = max_sen
        self.lang = lang
        self.method = method

        if loadJson:
            try:
                with open(jsonPath,'r') as f:
                        self.wiki_dict = json.load(f)
            except:
                print('JSON NOT FOUND: pass saved json file path')
                raise SystemExit


        else:
            self.wiki_dict = {}

        self.saveJson = saveJson

        self.saveJsonpath = saveJsonpath


    
    def load_model(self, lang, method='tags'):

        if method=='LM' and lang!='en':
            tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
            model = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
            nlp = pipeline("ner", model=model, tokenizer=tokenizer)
            self.extractor = self.tag_extraction_from_LM

            return nlp

        elif method=='LM' and lang=='en':
            
            self.extractor = self.tag_extraction_from_spacy
            return spacy.load("en_core_web_lg")
        else:
            self.extractor = self.tag_extraction_from_tags



    def tag_extraction_from_spacy(self, sen, lang=None):
        
        annotated = self.model(sen)
        extracted_names = [word.text for word in annotated.ents 
                        if word.label_=='PERSON' or word.label_=='ORG'or word.label_=='GPE']
        
        
        return extracted_names
            


    def tag_extraction_from_tags(self, sent, lang, tags):

            
        sentsWithtags = [(s,t) for s,t in zip(sent.split(),tags.split())]
            
            
        entity_list = []
        for i,item in enumerate(sentsWithtags):
            if 'B-' in item[1]:
                j = i
                entity = []
                while j<len(sentsWithtags):
                    if sentsWithtags[j][1] =='O':
                        break
                    entity.append(sentsWithtags[j][0])
                    j+=1
                i = j
                
                if lang=='zh':
                    entity_list.append("".join(entity))
                
                else:
                    entity_list.append(" ".join(entity))
            
    

        return entity_list


    def tag_extraction_from_LM(self, sen, lang):
    
        if lang=='zh':
            sen = sen.replace(" ","")
        
        ner_results = self.model(sen)
        extracted_names = []

        for idx in range(len(ner_results)):
            if ner_results[idx]['entity'][0] == 'B':
                start = ner_results[idx]['start']
                end = ner_results[idx]['end']
                j = idx+1
                while j < len(ner_results):
                    if ner_results[j]['entity'][0] == 'B':
                        break
                    elif ner_results[j]['entity'][0] == 'I':
                        end = ner_results[j]['end']
                    j+=1
                idx = j
            
                extracted_names.append(sen[start:end].strip())
        
    
        return extracted_names

    
    def __call__(self, data):
        """
        It takes a list of sentences, and returns a list of sentences with the same length, but with the
        extracted information from the wikipedia page
        
        :param data: the data to be augmented
        :return: A list of dictionaries, each dictionary containing the information extracted from the
        wikipedia page of the corresponding entity.
        """


        augmented_data = []

        for i in tqdm(range(len(data))):
            
            sent = data[i][0]
            tags = data[i][1]
            aug_sent = self.information_extraction(sent, tags)
            augmented_data.append(aug_sent)
            
        try:
            if self.saveJson:
                if self.saveJsonpath !=None:
                    with open(self.saveJsonpath, 'w') as f:
                        json.dump(self.wiki_dict, f)
                else:
                    with open(f'wiki-info-{self.lang}.json', 'w') as f:
                        json.dump(self.wiki_dict, f)
        except:
            print("JSON save failed")

        return augmented_data

            
        
        


    def clean(self, text):
        """
        It takes a string as input and returns a string with all the brackets removed
        
        :param text: The text to be cleaned
        :return: the text without the brackets.
        """


        sent_lent = len(text)
        brackets = []
        i = 0
        while i<sent_lent:
            
            if text[i] == '(':
                p = [text[i]]
                j = i+1
                if j>=sent_lent:
                    break
                while len(p)!=0:
                    if text[j]=='(':
                        p.append(text[j])
                    elif text[j]==')':
                        p.pop()
                        
                    j+=1
                    if j>=sent_lent:
                        break
                
                if len(p)==0:
                    brackets.append(text[i:j])
                    i = j
            i+=1

            

    
        for pattern in brackets:
            
            text = text.replace(pattern,'')
          
        text = re.sub('\n','',text)
        
        return text.strip()
    




    def information_extraction(self, sen, tags = None):

        """
        1. Extract entities from the sentence using the extractor.
        2. Get the wikipedia page for each entity.
        3. Extract the summary of the wikipedia page.
        4. Clean the summary.
        5. Join the summary with the original sentence
        
        :param sen: the sentence to be augmented
        :return: The augmented text is being returned.
        """

        if self.method=='tags':
            wiki_entities = self.extractor(sen, self.lang, tags)
        else:
            wiki_entities = self.extractor(sen, lang=self.lang)


        if self.lang!='zh':
            wiki_links = ["_".join([token.capitalize() for token in names.strip().split()]) for names in set(wiki_entities)]
        else:
            wiki_links = list(set(wiki_entities))


        if self.lang=='multi':
            wiki_wiki = wikipediaapi.Wikipedia(detected_lang) # just to run the multi
        else:
            wiki_wiki = wikipediaapi.Wikipedia(self.lang)

        information = []
        try: 
            for link in wiki_links:
                if link=='':
                    continue
                
                if link in self.wiki_dict:
                    summary = self.wiki_dict[link]

                else:
                    page = wiki_wiki.page(link)
                    if page.exists():
                        summary = page.summary
                        if 'may refer to:' in summary:
                            return sen

                        self.wiki_dict[link] = summary
                    else:
                        return sen
                        
                # select pre-defined number of sentences
                summary = " ".join(nltk.tokenize.sent_tokenize(summary)[:self.max_sen])
                # clean text
                summary = self.clean(summary)
                information.append(summary)
            
            if len(information)>0:
                
                info = " ".join(information)
                augmented_text = sen+info.lower()
                return augmented_text

        except Exception as err:
            print(err)
            time.sleep(120)

        return sen


    

        

    
