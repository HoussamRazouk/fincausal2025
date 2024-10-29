
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss 
from transformers import BertModel
import torch

class BertTokenClassification(nn.Module):
    def __init__(self,num_labels, pretrained_model_name,max_length):
        super(BertTokenClassification, self).__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        print(self.bert.config.hidden_size)
        print(num_labels)
        self.dropout = nn.Dropout(0.1)
        self.num_labels=num_labels
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.max_length=max_length
        

    def forward(self, token_ids,text_attention_mask,loss_attention_mask=None,labels=None):
        
        ## reshape the input token ids for the text 
        token_ids=token_ids.view(-1, self.max_length)
        text_attention_mask=text_attention_mask.view(-1, self.max_length)
        
        ## get the embeddings of the text and the question 
        outputs = self.bert(token_ids,attention_mask=text_attention_mask)
        #print(outputs['last_hidden_state'])

        #sequence_output = self.dropout(outputs)
        
        #logits = self.classifier(sequence_output)
        
        #outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        
        if labels is not None:
            
            sequence_output = self.dropout(outputs['last_hidden_state'])
        
            logits = self.classifier(sequence_output)
            
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
            loss_fct = CrossEntropyLoss()

            active_loss = loss_attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1, self.num_labels)[active_loss]
            loss = loss_fct(active_logits, active_labels.float())
            outputs = (loss,) + outputs
        else:
        
            logits = self.classifier(outputs['last_hidden_state'])
            
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
            #loss_fct = CrossEntropyLoss()
            
            #if text_attention_mask is not None:
            #    active_loss = text_attention_mask.view(-1) == 1
            #    if loss_mask is not None:
            #        active_loss &= loss_mask.view(-1)
                    
            #    active_logits = logits.view(-1, self.num_labels)[active_loss]
            #    outputs=active_logits
        return outputs 