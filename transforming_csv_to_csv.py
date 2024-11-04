import pandas as pd


df=pd.read_csv('data/input_data_evaluation_en_bert-base-cased_BIN_annotation.csv')

for index,row in df.iterrows():

    if str(row['Answer'])=='nan':
        
        print (row)
        df['Answer'][index]="No Answer"

df.to_csv('data/input_data_evaluation_en_bert-base-cased_BIN_annotation_corrected.csv',index=False,sep=';')


df['ID'][228]
str(df['Answer'][228])
