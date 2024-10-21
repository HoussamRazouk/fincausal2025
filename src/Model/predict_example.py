
import sys
sys.path.append('.')
import torch
import torch.nn as nn
from src.Model.BertTokenClassification import BertTokenClassification
from transformers import BertTokenizer
import pickle

# Load the model state
num_labels=1
LM_name='bert-base-cased'

model_weights="src/Model/trained_models/bert-base-cased/bert-base-cased_200_model.pth"

max_length=512


model=torch.load(model_weights,map_location=torch.device('cpu') )
model.eval()

tokenizer = BertTokenizer.from_pretrained(LM_name)

## data example
#"Overall, Group trading continues to be subdued in large part due to legacy issues",'What is the main reason why the Group trading continues to be subdued?','legacy issues'

text="Overall, Group trading continues to be subdued in large part due to legacy issues"
## tokonized text example
#text_tokens=['Overall', ',', 'Group', 'trading', 'continues', 'to', 'be', 'subdued', 'in', 'large', 'part', 'due', 'to', 'legacy', 'issues']
Question='What is the main reason why the Group trading continues to be subdued?'
Answer='legacy issues'
## labels on the text example
#labels=[False, False, False, False, False, False, False, False, False, False, False, False, False, True, True]

## read the data 
file='processed_reference_data_practise_en'
    
with open(f'data/{file}.pkl', 'rb') as f: ## using pickle instead of csv  because there is a problem for the labels list 
    # Use pickle.load() to deserialize the data
        data = pickle.load(f)
#Answers=[]
#for index, row in data.iterrows():
def predict_an_example(row):    
    #print(f"Row {row}")
    #break
    
    text=row['Text']
    Question=row['Question']
    #Answer=row['Answer']
    
    
    #print(text)
    text_tokens=tokenizer(text, padding='max_length',max_length=max_length, truncation=True, return_tensors="pt",add_special_tokens=False)
    Question_tokens=tokenizer(Question, padding="max_length",max_length=max_length, truncation=True, return_tensors="pt",add_special_tokens=True)
            #padding the labels tensor  
    #labels=torch.tensor(labels)
    #pad_rows =max_length - labels.size(0)
    #labels=torch.nn.functional.pad(labels,(0, pad_rows), value=0)

            #truncate the labels tensor  
    #labels = labels[:max_length]
            
    text_input_ids=text_tokens["input_ids"]
    text_attention_mask=text_tokens["attention_mask"]
            
    Question_input_ids=Question_tokens["input_ids"]
    Question_attention_mask=Question_tokens["attention_mask"]
            
    #text_input_ids.shape
    #Question_input_ids.shape
    #text_attention_mask.shape
    #Question_attention_mask.shape
    #labels.shape

    text_input_ids.unsqueeze(0).shape

    outputs=model(text_input_ids,Question_input_ids,text_attention_mask=text_attention_mask,Question_attention_mask=Question_attention_mask)
    results=outputs.squeeze()
    results=results>0.5
    results=results.tolist()
    text_input_ids_list=text_input_ids.squeeze().tolist()

    Answer_ids=[]
    for i in range(len(results)):
        
        if results[i]:
            
            Answer_ids.append(text_input_ids_list[i])


    Answer_ids=torch.tensor(Answer_ids)

    Answer = tokenizer.decode(Answer_ids, skip_special_tokens=True)
    return Answer

data['predicted_Answer']=data.apply(lambda row: predict_an_example(row), axis=1)
data[['ID','Text','Question','Answer','predicted_Answer']].to_csv('test.csv')




