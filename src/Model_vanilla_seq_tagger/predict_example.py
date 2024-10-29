
import sys
sys.path.append('.')
import torch
import torch.nn as nn
from transformers import BertTokenizer
import pickle
from src.Model_vanilla_seq_tagger.config import init
from src.Model_vanilla_seq_tagger.data_pre_processing import get_test_data
# Load the model state
from scoring_program.task_evaluate import SAS,ExactMatch

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
            
            
            
    tokenized_answer= [token for token, entity in zip(row['tokenized Text'], labels) if entity != 'O']
    
    predicted_answer=tokenizer.convert_tokens_to_string(tokenized_answer)
            
            
    return predicted_answer 
    

config=init()
LM_name='bert-base-cased'

model_weights="src/Model_vanilla_seq_tagger/trained_models/bert-base-cased/bert-base-cased_420_model.pth"
Testing_file=config['input_test_data_file']


Test_data,Test_data_maxlength=get_test_data(Testing_file,LM_name)

tokenizer = BertTokenizer.from_pretrained(LM_name)
model=torch.load(model_weights,map_location=torch.device('cpu') )
model.eval()

for max_length in config['max_length']: ## specify the max_length based on the training data
        if Test_data_maxlength<max_length:
            Test_data_maxlength=max_length
            break


Test_data['predicted_Answer']=Test_data.apply(lambda row: predict_an_example(row,tokenizer,model,max_length=512), axis=1)

Test_data
row=Test_data.iloc[0]

row
predict_an_example(row,tokenizer,model,max_length=512)

sas=SAS(Test_data['predicted_Answer'], Test_data['Answer'])  
exact_match=ExactMatch(Test_data['predicted_Answer'], Test_data['Answer']) 

print(sas)

print(exact_match)


