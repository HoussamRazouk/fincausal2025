import sys
sys.path.append('.')
from src.config import config

from scoring_program.task_evaluate import SAS,ExactMatch
import pandas as pd
def evaluation(result_file_name):
    

    results=pd.read_csv(result_file_name)
    #Answer,response
    #SAS(predicted_answers, reference_answers)
    sas=SAS(results['response'], results['Answer'])  
    exact_match=ExactMatch(results['response'], results['Answer'])  
    
    return sas,exact_match

if True:
    configuration=config()    
    output_path=configuration['output_path']
    models=configuration['models']
    for model in models:
        print (f'testing {model}' )    
        sas,exact_match= evaluation(output_path+f"{model}_model_prediction.csv")   
        scores = [
                        "SAS: %f\n" % sas,
                        "ExactMatch: %f\n" % exact_match
                    ]

        for s in scores:
                        print(s, end='') 