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

class CustomModel(nn.Module):
    def __init__(self, model_name, num_classes=2, feat_dim=128, precision='None', is_teacher=False):
        super(CustomModel, self).__init__()
        print(f"-----Initializing {'Teacher' if is_teacher else 'Student'} CustomModel with {model_name} (Precision: {precision if precision!='None' else 'float32'})-----")

        # Определяем точность (None = float32)
        if precision == "fp16":
            self.autocast_dtype = torch.float16
            self.scaler = GradScaler()
        elif precision == "bf16":
            self.autocast_dtype = torch.bfloat16
            self.scaler = None  # bf16 не требует GradScaler
        else:
            self.autocast_dtype = None  # Обычный float32
            self.scaler = None

        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, config=self.config, trust_remote_code=True)

        self.emb_size = self.config.hidden_size
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.is_teacher = is_teacher

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False)
        )
        
        # Distillation projection layer to align teacher and student representations
        self.distill_proj = nn.Linear(self.emb_size, self.emb_size, bias=False)

    def get_mean_embeddings(self, input_ids, attention_mask):
        if self.autocast_dtype:  # Mixed Precision, если включено
            with autocast(device_type="cuda", dtype=self.autocast_dtype):
                return self._compute_embeddings(input_ids, attention_mask)
        else:  # Обычный float32
            return self._compute_embeddings(input_ids, attention_mask)

    def _compute_embeddings(self, input_ids, attention_mask):
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_embeddings = torch.sum(model_output.last_hidden_state * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_embeddings

    def contrast_logits(self, mean_output_1, mean_output_2):
        if self.autocast_dtype:
            with autocast(device_type="cuda", dtype=self.autocast_dtype):
                return self._compute_contrast_logits(mean_output_1, mean_output_2)
        else:
            return self._compute_contrast_logits(mean_output_1, mean_output_2)

    def _compute_contrast_logits(self, mean_output_1, mean_output_2):
        cnst_feat1 = self.contrast_head(mean_output_1)
        cnst_feat2 = self.contrast_head(mean_output_2)
        return cnst_feat1, cnst_feat2

    def forward(self, input_ids, attention_mask, task_type="train", future_input_ids=None, future_attention_mask=None):
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
            
        if task_type == "distill" and self.is_teacher and future_input_ids is not None:
            # Teacher processes context + future
            combined_input_ids = torch.cat([input_ids, future_input_ids], dim=1)
            combined_attention_mask = torch.cat([attention_mask, future_attention_mask], dim=1)
            return self.get_mean_embeddings(combined_input_ids, combined_attention_mask)
        
        if task_type == "distill" and not self.is_teacher:
            # Student processes only context
            return self.get_mean_embeddings(input_ids, attention_mask)

        if self.autocast_dtype:
            with autocast(device_type="cuda", dtype=self.autocast_dtype):
                return self._compute_forward(input_ids, attention_mask)
        else:
            return self._compute_forward(input_ids, attention_mask)

    def _compute_forward(self, input_ids, attention_mask):
        if input_ids.shape[1] == 2:
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1)
        else:
            batch_size = input_ids.shape[0]
            input_ids_1 = input_ids[:, :-1, :].view(batch_size, -1)
            input_ids_2 = input_ids[:, -1, :]
            attention_mask_1 = attention_mask[:, :-1, :].view(batch_size, -1)
            attention_mask_2 = attention_mask[:, -1, :]

        bert_output_1 = self.model(input_ids=input_ids_1, attention_mask=attention_mask_1)
        bert_output_2 = self.model(input_ids=input_ids_2, attention_mask=attention_mask_2)

        attention_mask_1 = attention_mask_1.unsqueeze(-1)
        attention_mask_2 = attention_mask_2.unsqueeze(-1)
        mean_output_1 = torch.sum(bert_output_1.last_hidden_state * attention_mask_1, dim=1) / torch.sum(attention_mask_1, dim=1)
        mean_output_2 = torch.sum(bert_output_2.last_hidden_state * attention_mask_2, dim=1) / torch.sum(attention_mask_2, dim=1)

        cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)

        return cnst_feat1, cnst_feat2, mean_output_1, mean_output_2
    
    def get_distill_embeddings(self, embeddings):
        """Project embeddings for distillation"""
        return self.distill_proj(embeddings)
    
    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        config_path = os.path.join(save_directory, "config.json")
        
        torch.save(self.state_dict(), model_path)
        self.config.save_pretrained(save_directory)
        print(f"Model and config saved to {save_directory}")
        
    def copy_parameters_from(self, source_model):
        """Copy parameters from source model to this model"""
        self.load_state_dict(source_model.state_dict())
        print("Model parameters copied successfully")



class PSCBert(BertPreTrainedModel):
    def __init__(self, config, num_classes=2, feat_dim=128, is_teacher=False):
        super(PSCBert, self).__init__(config)
        print(f"-----Initializing {'Teacher' if is_teacher else 'Student'} PSCBert-----")
        self.bert = BertModel(config)
        self.emb_size = self.bert.config.hidden_size
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.is_teacher = is_teacher

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))
            
        # Distillation projection layer
        self.distill_proj = nn.Linear(self.emb_size, self.emb_size, bias=False)
        
    def forward(self, input_ids, attention_mask, task_type, future_input_ids=None, future_attention_mask=None):        
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
            
        elif task_type == "distill" and self.is_teacher and future_input_ids is not None:
            # Teacher processes context + future
            combined_input_ids = torch.cat([input_ids, future_input_ids], dim=1)
            combined_attention_mask = torch.cat([attention_mask, future_attention_mask], dim=1)
            return self.get_mean_embeddings(combined_input_ids, combined_attention_mask)
        
        elif task_type == "distill" and not self.is_teacher:
            # Student processes only context
            return self.get_mean_embeddings(input_ids, attention_mask)
            
        else:
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
        
    def get_distill_embeddings(self, embeddings):
        """Project embeddings for distillation"""
        return self.distill_proj(embeddings)
        
    def copy_parameters_from(self, source_model):
        """Copy parameters from source model to this model"""
        self.load_state_dict(source_model.state_dict())
        print("Model parameters copied successfully")



class PSCRoberta(RobertaPreTrainedModel):
    def __init__(self, config, num_classes=2, feat_dim=128, is_teacher=False):
        super(PSCRoberta, self).__init__(config)
        print(f"-----Initializing {'Teacher' if is_teacher else 'Student'} PSCRoberta-----")
        self.roberta = RobertaModel(config)
        self.emb_size = self.roberta.config.hidden_size
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.is_teacher = is_teacher

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))
            
        # Distillation projection layer
        self.distill_proj = nn.Linear(self.emb_size, self.emb_size, bias=False)
        
    def forward(self, input_ids, attention_mask, task_type, future_input_ids=None, future_attention_mask=None):        
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
            
        if task_type == "distill" and self.is_teacher and future_input_ids is not None:
            # Teacher processes context + future
            combined_input_ids = torch.cat([input_ids, future_input_ids], dim=1)
            combined_attention_mask = torch.cat([attention_mask, future_attention_mask], dim=1)
            return self.get_mean_embeddings(combined_input_ids, combined_attention_mask)
        
        if task_type == "distill" and not self.is_teacher:
            # Student processes only context
            return self.get_mean_embeddings(input_ids, attention_mask)
            
        else:
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
            bert_output_1 = self.roberta.forward(input_ids=input_ids_1, attention_mask=attention_mask_1)
            bert_output_2 = self.roberta.forward(input_ids=input_ids_2, attention_mask=attention_mask_2)
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
        bert_output = self.roberta.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings
        
    def get_distill_embeddings(self, embeddings):
        """Project embeddings for distillation"""
        return self.distill_proj(embeddings)
        
    def copy_parameters_from(self, source_model):
        """Copy parameters from source model to this model"""
        self.load_state_dict(source_model.state_dict())
        print("Model parameters copied successfully")



class PSCDistilBERT(DistilBertPreTrainedModel):
    def __init__(self, config, num_classes=2, feat_dim=128, is_teacher=False):
        super(PSCDistilBERT, self).__init__(config)
        print(f"-----Initializing {'Teacher' if is_teacher else 'Student'} PSCDistilBERT-----")
        self.distilbert = DistilBertModel(config)
        self.emb_size = self.distilbert.config.hidden_size
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.is_teacher = is_teacher

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))
            
        # Distillation projection layer
        self.distill_proj = nn.Linear(self.emb_size, self.emb_size, bias=False)
        
    def forward(self, input_ids, attention_mask, task_type, future_input_ids=None, future_attention_mask=None):        
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
            
        if task_type == "distill" and self.is_teacher and future_input_ids is not None:
            # Teacher processes context + future
            combined_input_ids = torch.cat([input_ids, future_input_ids], dim=1)
            combined_attention_mask = torch.cat([attention_mask, future_attention_mask], dim=1)
            return self.get_mean_embeddings(combined_input_ids, combined_attention_mask)
        
        if task_type == "distill" and not self.is_teacher:
            # Student processes only context
            return self.get_mean_embeddings(input_ids, attention_mask)
            
        else:
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
            bert_output_1 = self.distilbert.forward(input_ids=input_ids_1, attention_mask=attention_mask_1)
            bert_output_2 = self.distilbert.forward(input_ids=input_ids_2, attention_mask=attention_mask_2)
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
        bert_output = self.distilbert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings
        
    def get_distill_embeddings(self, embeddings):
        """Project embeddings for distillation"""
        return self.distill_proj(embeddings)
        
    def copy_parameters_from(self, source_model):
        """Copy parameters from source model to this model"""
        self.load_state_dict(source_model.state_dict())
        print("Model parameters copied successfully")