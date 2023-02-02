
import pandas as pd

def read_data(file):
    
    try:
        with open(file,'r') as file:
            lines = [line.strip() for line in file if len(line.strip())!=0]
            
    except:
        print(f'Courrpt File: {file}')
        return []
        
    i = 0
    sentences = []

    
    for j,line in enumerate(lines[1:]):
    
    
        if len(line)>0 and line[0] =='#':
            sentences.append(lines[i:j+1])
            i = j+1
        if len(line)==1:
            print(file)
        else:
            continue
    if i<j:
        sentences.append(lines[i:])
            
    return sentences
                
def get_id_domain(line):
    
    line = line.replace("# id",'')
    ID = line.split('\t')[0].strip()
    try:
        domain = line.split('\t')[1].replace('domain=','')
    except:
        domain = None

    return ID, domain



def prepare_data(file):
    
    mx = 0
    raw_data = read_data(file)
    
    dataframe = []
    for item in raw_data:
        
        ID, domain = get_id_domain(item[0])
        tokens = []
        label = []
        for entry in item[1:]:
            
            entry = entry.split('_ _')
            tokens.append(entry[0])
            try: 
                label.append(entry[1])
            except:
                label.append(None)
        
        assert len(label)==len(tokens) 
        
        sens= " ".join(tokens)
        labels= " ".join(label)
        mx = max(mx,len(sens))
        
        dataframe.append([ID,domain,sens,labels])

    dataframe = pd.DataFrame(dataframe, columns= ['ID','lang','sent','labels'])

    return dataframe
        