from torch.utils.data import Dataset
import torch
class Extractive_seq_tagger_Dataset(Dataset):
    # This loads the data and converts it, make data rdy
    def __init__(self,tokenizer,data,max_length=512):
        # load data
        self.data=data.reset_index(drop=True)
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.start_token=['[CLS]']# maybe are different for other LM than BERT 
        self.separator_token=['[SEP]']# maybe are different for other LM than BERT 
        self.pad_token=['[PAD]']# maybe are different for other LM than BERT 
        self.start_token_id=self.tokenizer.convert_tokens_to_ids(self.start_token)
        self.separator_token_id=self.tokenizer.convert_tokens_to_ids(self.separator_token)
        self.pad_token_id=self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.label_to_index ={
            'O':0,
            'B-Answer':1,
            'I-Answer':2,
        }
        
        # This returns the total amount of samples in your Dataset
    def __len__(self):
        
        return len(self.data)
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):

        
        Text=self.data['tokenized Text'][idx]
        Question=self.data['tokenized Question'][idx]
        Label=self.data['BIO Labeled Text'][idx]
        
        Question_ids=self.tokenizer.convert_tokens_to_ids(Question)
        Text_ids=self.tokenizer.convert_tokens_to_ids(Text)
        start_token_id=self.start_token_id
        separator_token_id=self.separator_token_id
        pad_token_id=self.separator_token_id

        token_ids=start_token_id+Question_ids+separator_token_id+Text_ids+separator_token_id
        loss_attention_mask=[0]*(len(Question_ids)+2)+[1]*len(Text_ids)+[0] #[CLS][Q][SEP][T][SEP]
        text_attention_mask=[1]*(len(Question_ids)+2)+[1]*len(Text_ids)+[1] #[CLS][Q][SEP][T][SEP]
        
        
        labels=['O']*(len(Question_ids)+2)+Label+['O']#[CLS][Q][SEP][T][SEP]
        max_length=self.max_length
        assert(len(token_ids)<=max_length) ## no truncating just padding 

        token_ids=token_ids+pad_token_id*(max_length-len(token_ids))
        loss_attention_mask=loss_attention_mask+[0]*(max_length-len(loss_attention_mask))
        text_attention_mask=text_attention_mask+[0]*(max_length-len(text_attention_mask))
        
        
        labels=labels+['O']*(max_length-len(labels))
        labels = [self.label_to_index[label] for label in labels]

        token_ids=torch.tensor(token_ids)
        loss_attention_mask=torch.tensor(loss_attention_mask)
        text_attention_mask=torch.tensor(text_attention_mask)
        labels=torch.nn.functional.one_hot(torch.tensor(labels), num_classes=len(self.label_to_index))
        #labels=torch.tensor(labels)

        return token_ids,loss_attention_mask,text_attention_mask,labels


def test():
    import torch
    from transformers import BertTokenizer

    # Create a tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    start_token=['[CLS]']
    separator_token=['[SEP]']
    pad_token=['[PAD]'] 
    Text=['All', 'the', 'Directors', 'are', 'resident', 'in', 'the', 'UK', 'and', 'their', 'biographical', 'details', ',', 'which', 'are', 'set', 'out', 'on', 'page', 's', '26', 'and', '27', ',', 'demonstrate', 'the', 'wide', 'range', 'of', 'skills', 'and', 'experience', 'that', 'they', 'bring', 'to', 'the', 'Board', '.', 'In', 'view', 'of', 'the', 'Company', "'", 's', 'size', 'and', 'as', 'the', 'Board', 'is', 'comprised', 'of', 'only', 'five', 'Directors', ',', 'all', 'of', 'whom', 'are', 'independent', ',', 'the', 'Board', 'considers', 'it', 'sensible', 'for', 'all', 'the', 'Directors', 'to', 'be', 'members', 'of', 'the', 'Audi', '##t', 'Committee', 'and', 'of', 'the', 'No', '##mination', 'and', 'Re', '##mu', '##ner', '##ation', 'Committee', '.']
    Question=['What', 'is', 'the', 'impact', 'of', 'the', 'Company', "'", 's', 'small', 'size', 'and', 'having', 'a', 'Board', 'comprised', 'of', 'only', 'five', 'independent', 'Directors', '?']
    Label=['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'I-Answer', 'O']
    Answer=['the', 'Board', 'considers', 'it', 'sensible', 'for', 'all', 'the', 'Directors', 'to', 'be', 'members', 'of', 'the', 'Audi', '##t', 'Committee', 'and', 'of', 'the', 'No', '##mination', 'and', 'Re', '##mu', '##ner', '##ation', 'Committee']

    Question_ids=tokenizer.convert_tokens_to_ids(Question)
    Text_ids=tokenizer.convert_tokens_to_ids(Text)
    start_token_id=tokenizer.convert_tokens_to_ids(start_token)
    separator_token_id=tokenizer.convert_tokens_to_ids(separator_token)
    pad_token_id=tokenizer.convert_tokens_to_ids(pad_token)

    token_ids=start_token_id+Question_ids+separator_token_id+Text_ids+separator_token_id
    loss_attention_mask=[0]*len(Question_ids+2)+[1]*len(Text_ids)+[0] #[CLS][Q][SEP][T][SEP] used to ignore any miss label of the question tokens 
    text_attention_mask=[1]*len(Question_ids+2)+[1]*len(Text_ids)+[1] #[CLS][Q][SEP][T][SEP] used to 
    
    labels=['O']*len(Question_ids+2)+Label+['O']#[CLS][Q][SEP][T][SEP]
    max_length=512
    assert(len(token_ids)<=max_length) ## no truchating just pading 

    token_ids=token_ids+pad_token_id*(max_length-len(token_ids))
    attention_mask=attention_mask+[0]*(max_length-len(attention_mask))
    labels=labels+['O']*(max_length-len(labels))

    token_ids=torch.tensor(token_ids)
    attention_mask=torch.tensor(attention_mask)
    labels=torch.tensor(labels)

    return token_ids,loss_attention_mask,text_attention_mask,labels







