import pandas as pd
import pandas as pd
import zipfile
import io

LM_names=['bert-base-multilingual-cased_en&esLast training_model','bert-base-multilingual-cased_enLast training_model','bert-base-multilingual-cased_esLast training_model','bert-base-spanish-wwm-cased_esLast training_model']
f_names=['BERT_ML_EN&ES','BERT_ML_EN','BERT_ML_ES','BERT_ES_ES']

Testing_files=['input_data_evaluation_es','input_data_evaluation_en']

for Testing_file in Testing_files:
    i=0
    
    for LM_name in LM_names:

        df=pd.read_csv(f'data/{Testing_file}_{LM_name[:-19]}.csv',sep=';')

        for index,row in df.iterrows():

            if str(row['Answer'])=='nan':
                
                print (row)
                df['Answer'][index]="No Answer"

        # Create a ZipFile object
        if Testing_file=='input_data_evaluation_es':
            file_name='Es_on_'+f_names[i]
        else:
            file_name='En_on_'+f_names[i]
        with zipfile.ZipFile(f'zips/{file_name}.zip', 'w') as zip_file:
            # Create a BytesIO object to write the CSV data to
            csv_buffer = io.BytesIO()
            
            # Write the DataFrame to the CSV buffer
            df.to_csv(csv_buffer, index=False,sep=';')
            
            # Write the CSV buffer to the zip file
            if Testing_file=='input_data_evaluation_es':
                
                zip_file.writestr('results_es.csv', csv_buffer.getvalue())
            else :
                zip_file.writestr('results_en.csv', csv_buffer.getvalue())
                
        i=i+1

