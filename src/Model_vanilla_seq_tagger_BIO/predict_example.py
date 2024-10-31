
import sys
sys.path.append('.')
import torch
import torch.nn as nn
from transformers import BertTokenizer
import pickle
from src.Model_vanilla_seq_tagger_BIO.config import init
from src.Model_vanilla_seq_tagger_BIO.data_pre_processing import get_test_data
# Load the model state
from scoring_program.task_evaluate import SAS,ExactMatch
import random


#def answer_from_tokens_and_labels(tokenizer,Text,labels):
def answer_from_tokens_and_labels(tokenizer,tokenized_Text,labels):
    
    #### need to handel errors by the tokenizer
    '''
    Text="some stuff  which doesn't matter."
    Text=Text.strip()
    words=Text.split(' ')

    ['some', 'stuff', 'which', 'doesn', "'", 't', 'matter', '.']
    labels=['O', 'O', 'O', 'O', "'", 'I', 'I', 'O']


    words
    tokenized_words=[]
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    for word in words:
        tokenized_words.append(tokenizer.tokenize(word))
    label_IDX=0
    Word_IDX=0
    Answer=''
    for word in words:
        Flag=False
        
        for token in tokenized_words[Word_IDX]:
            
            if labels[label_IDX]!='O':
                Flag=True
            label_IDX=label_IDX+1
        if Flag: 
            Answer=Answer+' '+word
        Word_IDX=Word_IDX+1
    
    Answer=Answer.strip()
    '''
    '''
    Text=Text.strip()
    words=Text.split(' ')
    tokenized_words=[]
    for word in words:
        tokenized_words.append(tokenizer.tokenize(word))
    label_IDX=0
    Word_IDX=0
    Answer=''
    for word in words:
        Flag=False
        
        for token in tokenized_words[Word_IDX]:
            
            if labels[label_IDX]!='O':
                Flag=True
            label_IDX=label_IDX+1
        if Flag: 
            Answer=Answer+' '+word
        Word_IDX=Word_IDX+1
    
    Answer=Answer.strip()'''
    
    tokenized_answer= [token for token, entity in zip(tokenized_Text, labels) if entity != 'O']
    
    Answer=tokenizer.convert_tokens_to_string(tokenized_answer)
    return Answer

def get_labels_occurrences(BIO_Labeled_Text):
    aggregated_tokens=[]
    for item in Test_data['BIO Labeled Text']:
        
        aggregated_tokens=aggregated_tokens+item
        
    aggregated_tokens

    annotations= ['O', 'B-Answer', 'I-Answer']
    occurrences={}
    for annotation in annotations:
        occurrences[annotation]=0
        for item in aggregated_tokens:
            if item==annotation:
                
                occurrences[annotation]=occurrences[annotation]+1
            

    for annotation in annotations:   
        occurrences[annotation]=occurrences[annotation]/len(aggregated_tokens)   

    return occurrences


def generate_random_annotations(occurrences,tokenized_Text):
    random_labels=[]
    choices = list(occurrences.keys())
    weights = list(occurrences.values())
    for i in range(len(tokenized_Text)):
        
        random_labels.append(random.choices(choices, weights=weights)[0])
    return random_labels

def predict_an_example(row,tokenizer,model,max_length=512):    
    ### predict one example at a time
    ###
    
    label_to_index ={
            'O':0,
            'B-Answer':1,
            'I-Answer':2,
        }
    
    index_to_label ={
            0:'O',
            1:'B-Answer',
            2:'I-Answer',
        }
    
    
    tokenized_Text =  row['tokenized Text']
    tokenized_Question= row['tokenized Question']
    
    #print(text)
    
    Question_ids=tokenizer.convert_tokens_to_ids(tokenized_Question)
    Text_ids=tokenizer.convert_tokens_to_ids(tokenized_Text)
    
    start_token=['[CLS]']# maybe are different for other LM than BERT 
    separator_token=['[SEP]']# maybe are different for other LM than BERT 
    pad_token=['[PAD]']# maybe are different for other LM than BERT 
    
    start_token_id=tokenizer.convert_tokens_to_ids(start_token)
    separator_token_id=tokenizer.convert_tokens_to_ids(separator_token)
    pad_token_id=tokenizer.convert_tokens_to_ids(pad_token)

    
    token_ids=start_token_id+Question_ids+separator_token_id+Text_ids+separator_token_id
    loss_attention_mask=[0]*(len(Question_ids)+2)+[1]*len(Text_ids)+[0] #[CLS][Q][SEP][T][SEP]
    text_attention_mask=[1]*(len(Question_ids)+2)+[1]*len(Text_ids)+[1] #[CLS][Q][SEP][T][SEP]

    assert(len(token_ids)<=max_length) ## no truncating just padding 

    token_ids=token_ids+pad_token_id*(max_length-len(token_ids))
    loss_attention_mask=loss_attention_mask+[0]*(max_length-len(loss_attention_mask))
    text_attention_mask=text_attention_mask+[0]*(max_length-len(text_attention_mask))
    
    token_ids=torch.tensor(token_ids)
    loss_attention_mask=torch.tensor(loss_attention_mask)
    text_attention_mask=torch.tensor(text_attention_mask)
    
    
    
    outputs=model(token_ids,text_attention_mask,loss_attention_mask=loss_attention_mask)
 
    
    processed_output=torch.argmax(outputs[0][0], dim=1, keepdim=False).tolist()
    labels=[]
    
    for idx in range(max_length):
        
        if token_ids[idx]==pad_token:
            break
        elif loss_attention_mask[idx]==0:
            continue
        else:
            labels.append(index_to_label[processed_output[idx]])
            
    predicted_answer= answer_from_tokens_and_labels(tokenizer,row['tokenized Text'], labels)        
            
    #tokenized_answer= [token for token, entity in zip(row['tokenized Text'], labels) if entity != 'O']
    
    #predicted_answer=tokenizer.convert_tokens_to_string(tokenized_answer)
            
            
    return predicted_answer 
    




def main():
    
    config=init()
    LM_name='bert-base-cased'
    Testing_file=config['input_test_data_file']

    Test_data,Test_data_maxlength=get_test_data(Testing_file,LM_name)
    for max_length in config['max_length']: ## specify the max_length based on the training data
            if Test_data_maxlength<max_length:
                Test_data_maxlength=max_length
                break

    tokenizer = BertTokenizer.from_pretrained(LM_name)

    Test_data['nuance_Answer']=Test_data.apply(lambda row: tokenizer.convert_tokens_to_string(row['tokenized Answer']), axis=1)


    occurrences=get_labels_occurrences(Test_data['BIO Labeled Text'])

    Test_data['Random BIO Labels']=Test_data.apply(lambda row: generate_random_annotations(occurrences,row['tokenized Text']), axis=1)

    Test_data['Random Answer']=Test_data.apply(lambda row: answer_from_tokens_and_labels(tokenizer,
                                                                                        row['tokenized Text'],
                                                                                        row['Random BIO Labels']), axis=1)


    for fold in config['folds']:
        
        model_weights=f"src/Model_vanilla_seq_tagger_BIO/trained_models/bert-base-cased_BIO/bert-base-cased_{fold}_model.pth"
        #src/Model_vanilla_seq_tagger_BIO/trained_models/bert-base-cased_BIO/bert-base-cased_420_model .pth
        model=torch.load(model_weights,map_location=torch.device('cpu') )
        model.eval()

        Test_data['predicted_Answer']=Test_data.apply(lambda row: predict_an_example(row,tokenizer,model,max_length=512), axis=1)



        sas=SAS(Test_data['predicted_Answer'], Test_data['Answer'])  
        exact_match=ExactMatch(Test_data['predicted_Answer'], Test_data['Answer']) 


        sas_fair=SAS(Test_data['predicted_Answer'], Test_data['nuance_Answer'])  
        exact_match_fair=ExactMatch(Test_data['predicted_Answer'], Test_data['nuance_Answer'])
        
        print(f"######### Fold: {fold} #########")
        
        print("SAS achieved")
        print(sas)

        print("Exact match achieved")
        print(exact_match)


        print("SAS achieved fair")
        print(sas_fair)

        print("Exact match achieved fair")
        print(exact_match_fair)


    max_sas=SAS(Test_data['nuance_Answer'], Test_data['Answer']) 

    max_exact_match=ExactMatch(Test_data['nuance_Answer'], Test_data['Answer']) 


    random_sas=SAS(Test_data['Random Answer'], Test_data['Answer'])  
    random_exact_match=ExactMatch(Test_data['Random Answer'], Test_data['Answer']) 



    ##TODO 
    print("SAS achieved")
    print(sas)

    print("Exact match achieved")
    print(exact_match)


    print("SAS achieved fair")
    print(sas_fair)

    print("Exact match achieved fair")
    print(exact_match_fair)


    print("MAX SAS")
    print(max_sas)

    print("MAX Exact match")
    print(max_exact_match)


    print("Random SAS")
    print(random_sas)

    print("Random_exact_match")
    print(random_exact_match)