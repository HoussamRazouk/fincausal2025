
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import BertModel
import torch

class BertTokenClassification(nn.Module):
    def __init__(self,num_labels, pretrained_model_name):
        super(BertTokenClassification, self).__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        print(self.bert.config.hidden_size)
        print(num_labels)
        self.dropout = nn.Dropout(0.1)
        self.num_labels=num_labels
        self.classifier = nn.Linear(self.bert.config.hidden_size*2, num_labels)
        self.max_length=512
        

    def forward(self, text_input_ids,Question_input_ids, text_attention_mask=None, Question_attention_mask=None,labels=None, loss_mask=None):
        
        ## reshape the input token ids for the text 
        text_input_ids=text_input_ids.view(-1, self.max_length)
        text_attention_mask=text_attention_mask.view(-1, self.max_length)
        
        ## reshape the input token ids for the question  
        Question_input_ids=Question_input_ids.view(-1, self.max_length)
        Question_attention_mask=Question_attention_mask.view(-1, self.max_length)
        
        ## get the embeddings of the text and the question 
        outputs = self.bert(text_input_ids,attention_mask=text_attention_mask)
        sentence_embedding=self.bert(Question_input_ids,attention_mask=Question_attention_mask)
        

        ## select the CLS token embedding from the question 
        sentence_embedding = sentence_embedding[0][:, 0, :]
    
        ## concatenate each token embedding of the text with the token embedding of the question    
        concat_embeddings = torch.cat((outputs[0], sentence_embedding.unsqueeze(1).expand(-1, outputs[0].shape[1], -1)), dim=2)
        
        sequence_output = concat_embeddings

        sequence_output = self.dropout(sequence_output)
        
        logits = self.classifier(sequence_output)
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()

            active_loss = text_attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1, self.num_labels)[active_loss]
            loss = loss_fct(active_logits, active_labels.float())
            outputs = (loss,) + outputs
        else:
            if text_attention_mask is not None:
                active_loss = text_attention_mask.view(-1) == 1
                if loss_mask is not None:
                    active_loss &= loss_mask.view(-1)
                    
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                outputs=active_logits
        return outputs 