import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, MultiheadAttention
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel, DistilBertPreTrainedModel, DistilBertModel

from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


class PSCBert(BertPreTrainedModel):
    def __init__(self, config, num_classes=2, feat_dim=128):
        super(PSCBert, self).__init__(config)
        print("-----Initializing PSCBert-----")
        self.bert = BertModel(config)
        self.emb_size = self.bert.config.hidden_size
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))
        
    def forward(self, input_ids, attention_mask, task_type, token_type_ids=None):        
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
        elif task_type == "distill":
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            return bert_output.hidden_states
        else: # "train"
            '''
            When both query and reponse are single-turn sentence, input_ids are in shape
            Batch_Size * 2 * Max_Sequence_Length

            When query is multi-turn dialogue and reponse is single-turn sentence, input_ids are in shape
            Batch_Size * (Num_of_turn + 1) * Max_Sequence_Length

            See 'prepare_pairwise_input_multiturn_concatenate()' and 'prepare_pairwise_input()' in training.py for more details

            The last index of the second dimension always stands for the response, the rest stands for the query
            '''
            if input_ids.shape[1] == 2:
                input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
                attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1) 
            else:
                batch_size = input_ids.shape[0]
                input_ids_1 = input_ids[:, :-1, :].view(batch_size, -1)
                input_ids_2 = input_ids[:, -1, :]
                attention_mask_1 = attention_mask[:, :-1, :].view(batch_size, -1)
                attention_mask_2 = attention_mask[:, -1, :]
            

            # mean embeddings
            bert_output_1 = self.bert.forward(input_ids=input_ids_1, attention_mask=attention_mask_1)
            bert_output_2 = self.bert.forward(input_ids=input_ids_2, attention_mask=attention_mask_2)
            attention_mask_1 = attention_mask_1.unsqueeze(-1)
            attention_mask_2 = attention_mask_2.unsqueeze(-1)
            mean_output_1 = torch.sum(bert_output_1[0]*attention_mask_1, dim=1) / torch.sum(attention_mask_1, dim=1)
            mean_output_2 = torch.sum(bert_output_2[0]*attention_mask_2, dim=1) / torch.sum(attention_mask_2, dim=1)

            cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
            return cnst_feat1, cnst_feat2, mean_output_1, mean_output_2
            
    # pass BERT embedding through the contrastive heads to get logits
    def contrast_logits(self, embd1, embd2):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        feat2 = F.normalize(self.contrast_head(embd2), dim=1)
        return feat1, feat2

    # calculate the embedding of an input sentence as the average embeddings of its tokens
    def get_mean_embeddings(self, input_ids, attention_mask):
        # mean embeddings
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings
