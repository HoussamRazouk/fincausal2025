

import sys
sys.path.append('.')
import torch
import torch.nn as nn
from transformers import BertTokenizer
import pickle
from src.Model_vanilla_seq_tagger_BIN.config import init
from src.Model_vanilla_seq_tagger_BIN.data_pre_processing import get_test_data
from src.Model_vanilla_seq_tagger_BIN.predict_example import predict_an_example
# Load the model state
#from scoring_program.task_evaluate import SAS,ExactMatch
import random


#config=init()
LM_names=['bert-base-multilingual-cased_en&esLast training_model','bert-base-multilingual-cased_enLast training_model','bert-base-multilingual-cased_esLast training_model','bert-base-spanish-wwm-cased_esLast training_model']
tokenizer_names=['bert-base-multilingual-cased','bert-base-multilingual-cased','bert-base-multilingual-cased','dccuchile/bert-base-spanish-wwm-cased']

Testing_files=['input_data_evaluation_es','input_data_evaluation_en']


#for max_length in config['max_length']: ## specify the max_length based on the training data
#            if Test_data_maxlength<max_length:
#                Test_data_maxlength=max_length
#                break


for Testing_file in Testing_files:
    i=0
    for LM_name in LM_names:
        Test_data,Test_data_maxlength=get_test_data(Testing_file,tokenizer_names[i])

        tokenizer = BertTokenizer.from_pretrained(tokenizer_names[i])
        if LM_name== 'bert-base-spanish-wwm-cased_esLast training_model':

            model_weights=f"src/Model_vanilla_seq_tagger_BIN/trained_models/bert-base-spanish-wwm-cased_BIN/{LM_name}.pth"
        else:
            model_weights=f"src/Model_vanilla_seq_tagger_BIN/trained_models/bert-base-multilingual-cased_BIN/{LM_name}.pth"
        #src/Model_vanilla_seq_tagger_BIN/trained_models/bert-base-cased_BIN/bert-base-cased_420_model .pth
        
        model=torch.load(model_weights,map_location=torch.device('cpu') )
        model.eval()

        Test_data['Answer']=Test_data.apply(lambda row: predict_an_example(row,tokenizer,model,max_length=512), axis=1)
        Test_data['Answer']=Test_data.apply(lambda row: row['Answer'][1], axis=1)

        Test_data[['ID','Text','Question','Answer']].to_csv(f'data/{Testing_file}_{LM_name[:-19]}.csv',index=False,sep=';')
        i=i+1

