import pandas as pd


df=pd.read_csv('data/input_data_evaluation_en_bert-base-cased_BIO_annotation.csv')

df.to_csv('data/input_data_evaluation_en_bert-base-cased_BIO_annotation_corrected.csv',index=False,sep=';')