import sys
sys.path.append('.')
from src.init import init, init_lama## just sets the API key as os variable 
import json 
import pandas as pd
import os
from openai import OpenAI
from tqdm.auto import tqdm
import threading
tqdm.pandas()

def predict_one_answer(client,model,ID='',Answer='',Question='',Text='', domain='finance'):
    
    """
    This function extracts the relevant text snippet that answers a given causal question.
    """
    
    completion = client.chat.completions.create(
                model=model,
                messages=[
                  {"role": "system", 
                  "content":f"""
                  You are an expert in causality and {domain}. Your task is to help users answer their causal question.
                  """},
                  {"role": "user", "content": f"""
                    Extract the relevant text that answers the following question:

                    '''
                    {Question}
                    '''
                    
                    from the given text:

                    '''
                    {Text}
                    '''

                    Provide the exact phrase or sentence from the original text that directly answers the question in a Json format as under 'response'.
                    Respond only in a the Json"
                    """}
                  ],
                  response_format={ "type": "json_object" }
                  ,
                temperature=0, ## no creativity here 
                max_tokens=4096
              )
    response=completion.choices[0].message.content
    
    try:
      response=json.loads(response)
      response['ID']=ID
      response['Text']=Text
      response['Question']=Question
      response['Answer']=Answer
        
    except:
       with open('to_check.txt','w') as f:
            f.write(str(response)+'\n')
            f.write('Prediction Model: '+str(model)+'\n')
            f.write('Domain: '+str(domain)+'\n')
    
    
    return response


def test():
    
    client=init_lama()
    
    models=["llama3-70b",
            "mixtral-8x22b-instruct",
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "llama3-8b",
            "mixtral-8x7b-instruct",
            "mistral-7b-instruct"]
    
    model=models[4]
    ID='123'
    Text='Nationwide is in robust financial health, having achieved profits of over £1 billion for the third consecutive year. As a mutual, profits are not the only barometer of our success, but they are important because they allow us to maintain our financial strength, to invest with confidence, and to return value to you, our members, through pricing and service.'
    Question='What is the effect of achieving profits of over £1 billion for the third consecutive year?'
    Answer='Nationwide is in robust financial health'

    response=predict_one_answer(client,model,ID=ID,Answer=Answer,Question=Question,Text=Text, domain='finance')
    print(response)


def model_tread(test_data,model,client,output_path):

    print(f"running {model}")
    results=[]
    for index, row in test_data.iterrows():
        print(index)
        if True:
        #try:
            ID=row['ID']
            Answer=row['Answer']
            Question=row['Question']
            Text=row['Text']
            
            response=predict_one_answer(client,model,ID=ID,Answer=Answer,Question=Question,Text=Text, domain='finance')
            print(response)
            results.append(response)
            df=pd.DataFrame(results)
            df=df[[
                'ID',
                'Text',
                'Question',
                'Answer',
                'response'
                ]]
            df.to_csv(output_path+f"{model}_model_prediction.csv",index=False)
        #except:
        #    print("Failed")
        #    print(row)
        #    with open(output_path+f'to_check/{model}/'+f"{model}_{index}_fail.txt",'w') as f:
        #         f.write(str(row))
        

#test()
def main():
    output_path='results/Q&A_LLM/'
    test_data=pd.read_csv("data/reference_data_practise_en.csv", sep=';')
    models=["llama3-70b",
            "mixtral-8x22b-instruct",
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "llama3-8b",
            "mixtral-8x7b-instruct",
            "mistral-7b-instruct"]
    threads = []
    model=models[4]
    try:
         os.makedirs(output_path+f'to_check/{model}')
    except:
         print(output_path+f'to_check/{model} already exists')
    
    if os.path.isfile(output_path+f"{model}_model_prediction.csv"):
            print(f"{model} already tested")
            #continue
    if model in ["gpt-3.5-turbo","gpt-4-turbo"]: # deferent API key 
        #different API Are used 
        init()
        client = OpenAI()
    
    else :
        #different API Are used 
        client=init_lama()
    thread = threading.Thread(target=model_tread, args=(test_data,model,client,output_path))
    threads.append(thread)
    thread.start()
    
    
#main()
#test()
from scoring_program.task_evaluate import SAS,ExactMatch

def evaluation(result_file_name):
    results=pd.read_csv(result_file_name)
    #Answer,response
    #SAS(predicted_answers, reference_answers)
    sas=SAS(results['response'], results['Answer'])  
    exact_match=ExactMatch(results['response'], results['Answer'])  
    
    return sas,exact_match

    
output_path='results/Q&A_LLM/'
models=["llama3-70b",
            "mixtral-8x22b-instruct",
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "llama3-8b",
            "mixtral-8x7b-instruct",
            "mistral-7b-instruct"]

model=models[4]    
sas,exact_match= evaluation(output_path+f"{model}_model_prediction.csv")   
scores = [
            "SAS: %f\n" % sas,
            "ExactMatch: %f\n" % exact_match
        ]

for s in scores:
            print(s, end='') 
