

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


def true_false_to_bio(labels):
    bio_labels = []
    for i, label in enumerate(labels):
        if label:
                bio_labels.append('Answer')  # 
        else:
            bio_labels.append('O')  # Outside
    return bio_labels


def get_data(file,LM,output_file='Vanilla_seq_tagger', save=False):
    df=pd.read_csv(f'data/{file}.csv',sep=';')

    df['the answer is in the text']=df.apply(lambda row: row['Answer'] in row['Text'], axis=1)## check if the answer is in the text 

    df=df[df['the answer is in the text']].reset_index(drop=True) ## remove the inconstant data 

    # Load the BERT tokenizer (bert-base-uncased)
    tokenizer = BertTokenizer.from_pretrained(LM)

    ###
    df['tokenized Text'] =df.apply(lambda row:  tokenizer.tokenize(
        row['Text'],
    ), axis=1)

    df['tokenized Answer'] =df.apply(lambda row:  tokenizer.tokenize(
        row['Answer'],
    ), axis=1)
    
    df['tokenized Question'] =df.apply(lambda row:  tokenizer.tokenize(
        row['Question'],
    ), axis=1)

    df['Labeled Text'] =df.apply(lambda row:  label_processing(
        row['tokenized Text'],
        row['tokenized Answer'],
    ), axis=1)
    
    df['BIO Labeled Text'] =df.apply(lambda row:  true_false_to_bio(
        row['Labeled Text'],
    ), axis=1)
    
    df['len Question and Text'] =df.apply(lambda row: len (row['tokenized Question']) + len (row['tokenized Text']) + 3, ## considering the special tokens when added e.g. CLS SEP SEP
                                           axis=1)
    
    if save:
        df.to_csv(f'data/{output_file}_{LM}_{file}.csv')
        with open(f'data/{output_file}_{LM}_{file}.pkl', 'wb') as f:
            # Use pickle.dump() to serialize the data
            pickle.dump(df, f)
        
    max_length=max(df['len Question and Text'])
    return df,max_length


def get_test_data(file,LM,output_file='Vanilla_seq_tagger', save=False):
    df=pd.read_csv(f'data/{file}.csv',sep=';')


    # Load the BERT tokenizer (bert-base-uncased)
    tokenizer = BertTokenizer.from_pretrained(LM)

    ###
    df['tokenized Text'] =df.apply(lambda row:  tokenizer.tokenize(
        row['Text'],
    ), axis=1)
    
    df['tokenized Question'] =df.apply(lambda row:  tokenizer.tokenize(
        row['Question'],
    ), axis=1)

    df['len Question and Text'] =df.apply(lambda row: len (row['tokenized Question']) + len (row['tokenized Text']) + 3, ## considering the special tokens when added e.g. CLS SEP SEP
                                           axis=1)    
    if save:
        df.to_csv(f'data/{output_file}_{LM}_{file}.csv')
        with open(f'data/{output_file}_{LM}_{file}.pkl', 'wb') as f:
            # Use pickle.dump() to serialize the data
            pickle.dump(df, f)
            
    if 'Answer' in df.columns:
        
        df['tokenized Answer'] =df.apply(lambda row:  tokenizer.tokenize(
            row['Answer'],
        ), axis=1)
        
        df['Labeled Text'] =df.apply(lambda row:  label_processing(
            row['tokenized Text'],
            row['tokenized Answer'],
        ), axis=1)
        
        df['BIO Labeled Text'] =df.apply(lambda row:  true_false_to_bio(
            row['Labeled Text'],
        ), axis=1)
            
    max_length=max(df['len Question and Text'])
    return df,max_length

#main('reference_data_practise_en')
def test():
    File='training_data_en'
    LM='bert-base-cased'
    output_file='Vanilla_seq_tagger'
    save=True
    df,max_length=get_data(File,LM,output_file,save)

    df,max_length
    
#test()

