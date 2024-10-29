import pandas as pd
from transformers import BertTokenizer
import pickle

def label_processing(tokenized_text,tokenized_Answer):

    label=[]

    for i in range(len(tokenized_text)):
        
        label.append(False)

    for text_index in range(len(tokenized_text)):
        
        Found=False
        
        for answer_index in range(len(tokenized_Answer)):
            
            if tokenized_text[text_index+answer_index]==tokenized_Answer[answer_index]:
                
                Found=True
            else :
                Found=False
                break  
        
        if Found:
            
            for answer_index in range(len(tokenized_Answer)):
                
                label[text_index+answer_index]= True   
            break
        
    return label    


def main(file,LM,output_file):
    df=pd.read_csv(f'data/{file}.csv',sep=';')

    df['the answer is in the text']=df.apply(lambda row: row['Answer'] in row['Text'], axis=1)## check if the answer is in the text 

    df=df[df['the answer is in the text']].reset_index(drop=True) ## remove the inconstant data 

    # Load the BERT tokenizer (bert-base-uncased)
    tokenizer = BertTokenizer.from_pretrained(LM)

    ###
    df['tokenized text'] =df.apply(lambda row:  tokenizer.tokenize(
        row['Text'],
    ), axis=1)

    df['tokenized Answer'] =df.apply(lambda row:  tokenizer.tokenize(
        row['Answer'],
    ), axis=1)

    df['Labeled Text'] =df.apply(lambda row:  label_processing(
        row['tokenized text'],
        row['tokenized Answer'],
    ), axis=1)

    df.to_csv(f'data/{output_file}_{file}.csv')
    with open(f'data/{output_file}_{file}.pkl', 'wb') as f:
        # Use pickle.dump() to serialize the data
        pickle.dump(df, f)

#main('reference_data_practise_en')

File='training_data_en'
LM='bert-base-cased'
output_file='conditional_seq_tagger'
main(File,LM,output_file)