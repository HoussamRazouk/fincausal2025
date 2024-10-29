def init():
    config={}
    config['LM']='bert-base-cased'
    config['max_length']=[64,128,256,512,1024]
    config['input_train_data_file']='training_data_en'
    config['input_test_data_file']='reference_data_practice_en'
    
    return config