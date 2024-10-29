from torch.utils.data import Dataset
import torch
class SeqTagDataset(Dataset):
    # This loads the data and converts it, make data rdy
    def __init__(self,tokenizer,df,max_length=512):
        # load data
        self.data=df.reset_index(drop=True)
        self.tokenizer=tokenizer
        self.max_length=max_length
        
        # This returns the total amount of samples in your Dataset
    def __len__(self):
        
        return len(self.data)
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):

        
        text=self.tokenizer.convert_tokens_to_string(self.data['tokenized text'][idx])
        #print(text)
        text_tokens=self.tokenizer(text,padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt",add_special_tokens=False)
        Question_tokens=self.tokenizer(self.data['Question'][idx],padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt",add_special_tokens=True)
        #padding the labels tensor  
        labels=torch.tensor(self.data['Labeled Text'][idx])
        pad_rows =self.max_length - labels.size(0)
        labels=torch.nn.functional.pad(labels,(0, pad_rows), value=0)

        #truncate the labels tensor  
        labels = labels[:self.max_length]
        
        text_input_ids=text_tokens["input_ids"]
        text_attention_mask=text_tokens["attention_mask"]
        
        Question_input_ids=Question_tokens["input_ids"]
        Question_attention_mask=Question_tokens["attention_mask"]
        
        


        return text_input_ids,Question_input_ids,text_attention_mask,Question_attention_mask,labels
