def config():
    configuration={}
    configuration['output_path']='results/Q&A_LLM/'
    configuration['models']=["llama3-70b",
            #"mixtral-8x22b-instruct",
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "llama3-8b",
            #"mixtral-8x7b-instruct",
            #"mistral-7b-instruct",
            ]
    return configuration