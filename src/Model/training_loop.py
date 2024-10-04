
import sys
sys.path.append('.')
import logging
from src.Model.training_helpers.get_kfold_data import get_kfold_data
from src.Model.training_helpers.SeqTagDataset import SeqTagDataset
from src.Model.BertTokenClassification import BertTokenClassification

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm 
from transformers import BertTokenizer
import os
import argparse
import pandas as pd
import pickle
 


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def train(args,data,tokenizer):
    


    LM_name=args.lm_name

    # model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))

    # Set the number of training epochs
    num_epochs = 100
    folds=5
    min_delta = 0.001
    patience = 5
    num_epochs = 100
    num_labels=1

    # Initialize variables to track the best model


    folds=[420,200,100,150,24]

    
    for fold_num,random_state in enumerate(folds):

        logging.info(f"fold start {random_state},")

        train_df,val_df =get_kfold_data(data,random_state=random_state)
        #train_df_processed=data_df_to_labeled_sequence(train_df)
        #val_df_processed=data_df_to_labeled_sequence(val_df)

        

        training_DS=SeqTagDataset(tokenizer,df=train_df)
        train_dataloader=DataLoader(training_DS,batch_size=8,num_workers=2, shuffle=True,drop_last=True,pin_memory=True)

        val_DS=SeqTagDataset(tokenizer,df=val_df)
        val_dataloader=DataLoader(val_DS,batch_size=2,num_workers=2, shuffle=False,drop_last=True,pin_memory=True)


        # Initialize your custom model

        best_loss = float('inf')
        counter = 0
        model = BertTokenClassification(num_labels=num_labels,pretrained_model_name=LM_name)
        model=model.to(device)
        num_params = count_trainable_params(model)
        logging.info("Number of trainable parameters in the model: "+ str(num_params))
        # Optionally, you can load pre-trained weights if you have saved them previously
        
        # Set up optimizer and learning rate scheduler
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)

        
        for epoch in range(num_epochs):
            if True: ### training 
                model.train()
                total_loss = 0

                for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)):
                    
                        
                    text_input_ids,Question_text_input_ids, text_attention_mask, Question_attention_mask,labels = batch
                    text_input_ids,Question_text_input_ids, text_attention_mask, Question_attention_mask,labels= text_input_ids.to(device),Question_text_input_ids.to(device), text_attention_mask.to(device), Question_attention_mask.to(device),labels.to(device)
                    
                    # Clear gradients
                    model.zero_grad()

                    # Forward pass
                    outputs = model(text_input_ids,Question_text_input_ids,text_attention_mask=text_attention_mask,Question_attention_mask=Question_attention_mask,labels=labels)

                    loss = outputs[0]

                    # Backward pass and optimization
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    #scheduler.step()

                    total_loss += loss.item()


                
                avg_loss = total_loss / len(train_dataloader)
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")


            if True:### validation 

                model.eval()
                all_targets = []
                all_predictions = []
                total_eval_loss = 0

                with torch.no_grad():

                    for step, batch in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)):
                        
                        
                        
                        text_input_ids,Question_input_ids,text_attention_mask,Question_attention_mask,labels = batch
                        
                        
                        text_input_ids=text_input_ids.to(device)
                        Question_input_ids=Question_input_ids.to(device)
                        text_attention_mask=text_attention_mask.to(device)
                        Question_attention_mask=Question_attention_mask.to(device)
                        labels=labels.to(device)
                        
                        
                        outputs = model(text_input_ids,Question_input_ids,text_attention_mask=text_attention_mask,Question_attention_mask=Question_attention_mask,labels=labels)
                        
                        loss = outputs[0]

                        #print (outputs[1])

                        total_eval_loss += loss.item()


                        predicted_labels = (outputs[1][0] > 0.5).int()

                        all_targets.extend(labels.view(-1, num_labels).tolist()[:len(predicted_labels.tolist())])
                        all_predictions.extend(predicted_labels.tolist())


                all_targets = torch.tensor(all_targets, dtype=torch.float32)
                
                all_predictions = torch.tensor(all_predictions, dtype=torch.float32)
                

                val_loss = total_eval_loss / len(val_dataloader)        

                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    # Create target Directory if it doesn't exist
                    directory='src/Model/trained_models/'+LM_name.split('/')[-1]
                    if not os.path.exists(directory):
                        os.mkdir(directory)
                        print("Directory ", directory, " created.")
                    else:
                        print("Directory ", directory, " already exists.")

                    torch.save(model, 'src/Model/trained_models/'+LM_name.split('/')[-1]+'/'+LM_name.split('/')[-1]+'_'+str(random_state)+'_model.pth')

                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        logging.info(f"Early stopping triggered for fold {fold_num + 1}/{folds}")
                        break
        logging.info(f"fold ends {random_state},")
        logging.info(f"############################################")
    logging.info(f"Done successfully")       
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Causal information extraction from text')

    parser.add_argument('--lm_name', type=str, default='bert-base-cased',\
                         help=''' Choose from: { }''')
    
    args = parser.parse_args()


    LM_name=args.lm_name

    
    logging.basicConfig(
        level=logging.INFO,         # Set the minimum level of log messages to display (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Customize the format of log messages
        filename='logs/training_'+LM_name.split('/')[-1]+'.log',          # Specify a file to write log messages (optional)
        filemode='w'                 # File mode: 'w' for writing, 'a' for appending (optional)
        )

    

    tokenizer = BertTokenizer.from_pretrained(LM_name)
    
   

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.split_kernels_merge = True
        torch.backends.cuda.max_split_size_mb = 7000  # Adjust this value based on your memory requirements

    else:
        device = torch.device("cpu")
        logging.info("Using device: "+ str(device))

    #config=config_init()

    #data_path='data/row'
    
    #data=get_IFX_data(data_path, config,args.lm_name)
    file='processed_reference_data_practise_en'
    #data=pd.read_csv(f'data/{file}.csv')
    with open(f'data/{file}.pkl', 'rb') as f:
    # Use pickle.load() to deserialize the data
        data = pickle.load(f)
    
    data['Labeled Text'][0][0]
    
    train(args,data,tokenizer)
    