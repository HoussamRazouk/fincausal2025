def init():
    config={}
    #config['LM']='bert-base-cased'
    config['LM']='bert-base-multilingual-cased'
    config['max_length']=[64,128,256,512,1024]
    #config['input_train_data_file']='training_data_en'
    #config['input_train_data_file']='training_data_es'
    config['input_train_data_file']='training_data_en&es'
    #config['input_test_data_file']='reference_data_practice_en'
    #config['input_test_data_file']='reference_data_practice_es'
    config['input_test_data_file']='reference_data_practice_en&es'
    config['folds']=[420,200,100,150,24]

    return config