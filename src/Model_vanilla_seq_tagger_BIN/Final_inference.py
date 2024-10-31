

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


config=init()
LM_name='bert-base-cased'
Testing_file='input_data_evaluation_en'

Test_data,Test_data_maxlength=get_test_data(Testing_file,LM_name)
for max_length in config['max_length']: ## specify the max_length based on the training data
            if Test_data_maxlength<max_length:
                Test_data_maxlength=max_length
                break

tokenizer = BertTokenizer.from_pretrained(LM_name)

model_weights=f"src/Model_vanilla_seq_tagger_BIN/trained_models/bert-base-cased_BIN/bert-base-cased_Last training_model.pth"
#src/Model_vanilla_seq_tagger_BIN/trained_models/bert-base-cased_BIN/bert-base-cased_420_model .pth
model=torch.load(model_weights,map_location=torch.device('cpu') )
model.eval()

Test_data['Answer']=Test_data.apply(lambda row: predict_an_example(row,tokenizer,model,max_length=512), axis=1)

Test_data[['ID','Text','Question','Answer']].to_csv('data/input_data_evaluation_en_bert-base-cased_BIN_annotation.csv',index=False)

