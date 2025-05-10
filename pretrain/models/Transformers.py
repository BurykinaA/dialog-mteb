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

    def _get_combined_mean_embeddings(self, input_ids_ctx, attention_mask_ctx, future_input_ids, future_attention_mask):
        # Helper for teacher: combines context and future utterance then gets mean embeddings.
        # Assumes inputs are tokenized segments that can be directly concatenated.
        # Max length handling should be done upstream (dataloader/tokenizer).
        combined_input_ids = torch.cat([input_ids_ctx, future_input_ids], dim=1)
        combined_attention_mask = torch.cat([attention_mask_ctx, future_attention_mask], dim=1)
        # _compute_embeddings handles the actual model call and pooling
        return self._compute_embeddings(combined_input_ids, combined_attention_mask)

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

    def forward(self, input_ids, attention_mask, task_type="contrastive_learning", 
                future_input_ids=None, future_attention_mask=None):

        if task_type == "evaluate":
            # get_mean_embeddings handles autocast internally
            return self.get_mean_embeddings(input_ids, attention_mask)

        # --- Teacher Path ---
        if self.is_teacher:
            if task_type != "distillation_teacher_forward":
                raise ValueError(f"Teacher model called with invalid task_type: {task_type}. Expected 'distillation_teacher_forward'.")
            if future_input_ids is None or future_attention_mask is None:
                raise ValueError("Teacher model in 'distillation_teacher_forward' mode requires future_input_ids and future_attention_mask.")

            if self.autocast_dtype:
                with autocast(device_type="cuda", dtype=self.autocast_dtype):
                    mean_embeddings_teacher = self._get_combined_mean_embeddings(input_ids, attention_mask, future_input_ids, future_attention_mask)
                    projected_teacher_emb = self.distill_proj(mean_embeddings_teacher)
            else:
                mean_embeddings_teacher = self._get_combined_mean_embeddings(input_ids, attention_mask, future_input_ids, future_attention_mask)
                projected_teacher_emb = self.distill_proj(mean_embeddings_teacher)
            return projected_teacher_emb

        # --- Student Path ---
        if task_type == "contrastive_learning":
            # _compute_forward expects paired inputs and handles autocast internally for its main computation.
            # It returns: cnst_feat1, cnst_feat2, mean_output_1, mean_output_2
            return self._compute_forward(input_ids, attention_mask)

        elif task_type == "distillation_student_forward":
            # Student processes only context (input_ids provided should be the context utterance)
            # get_mean_embeddings handles autocast for model forward pass
            mean_embeddings_student = self.get_mean_embeddings(input_ids, attention_mask)
            
            if self.autocast_dtype: # distill_proj might need autocast too
                with autocast(device_type="cuda", dtype=self.autocast_dtype):
                    projected_student_emb = self.distill_proj(mean_embeddings_student)
            else:
                projected_student_emb = self.distill_proj(mean_embeddings_student)
            return projected_student_emb

        elif task_type == "combined_learning_student_forward":
            # 1. Perform contrastive part using _compute_forward (handles autocast for its main ops)
            #    _compute_forward expects paired input_ids.
            #    It returns: cnst_feat1, cnst_feat2, mean_output_1 (embedding of 1st part), mean_output_2 (embedding of 2nd part)
            cnst_feat1, cnst_feat2, mean_out1_for_distill, _ = self._compute_forward(input_ids, attention_mask)
            
            # 2. Use mean_out1_for_distill (embedding of the first part of the pair) for student's distillation embedding.
            if self.autocast_dtype:
                with autocast(device_type="cuda", dtype=self.autocast_dtype):
                    projected_student_emb_combined = self.distill_proj(mean_out1_for_distill)
            else:
                projected_student_emb_combined = self.distill_proj(mean_out1_for_distill)
                
            return cnst_feat1, cnst_feat2, projected_student_emb_combined
            
        else:
            raise ValueError(f"Unknown task_type for student: {task_type}. Supported: 'contrastive_learning', 'distillation_student_forward', 'combined_learning_student_forward'.")

    def _compute_forward(self, input_ids, attention_mask):
        # This method is for contrastive learning, expecting paired inputs
        # input_ids shape: (Batch_Size, 2, Max_Seq_Len) or (Batch_Size, Num_Turns+1, Max_Seq_Len)
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
        self.base_model_prefix = "bert" # Added

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))
            
        # Distillation projection layer
        self.distill_proj = nn.Linear(self.emb_size, self.emb_size, bias=False)
        
    def _get_raw_mean_embeddings(self, input_ids, attention_mask):
        # Helper to get mean embeddings from the base model for a single sequence
        base_model = getattr(self, self.base_model_prefix)
        model_output = base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = model_output[0] 
        
        expanded_attention_mask = attention_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(last_hidden_state * expanded_attention_mask, dim=1)
        sum_mask = torch.sum(expanded_attention_mask, dim=1)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def _get_teacher_combined_embeddings(self, input_ids_ctx, attention_mask_ctx, future_input_ids, future_attention_mask):
        # Helper for teacher: combines context and future utterance then gets mean embeddings.
        combined_input_ids = torch.cat([input_ids_ctx, future_input_ids], dim=1)
        combined_attention_mask = torch.cat([attention_mask_ctx, future_attention_mask], dim=1)
        return self._get_raw_mean_embeddings(combined_input_ids, combined_attention_mask)

    def _get_contrastive_outputs(self, paired_input_ids, paired_attention_mask):
        # Processes paired inputs for contrastive learning.
        if paired_input_ids.shape[1] == 2:
            input_ids_1, input_ids_2 = torch.unbind(paired_input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(paired_attention_mask, dim=1) 
        else: # Multi-turn case
            batch_size = paired_input_ids.shape[0]
            input_ids_1 = paired_input_ids[:, :-1, :].reshape(batch_size, -1)
            attention_mask_1 = paired_attention_mask[:, :-1, :].reshape(batch_size, -1)
            input_ids_2 = paired_input_ids[:, -1, :]
            attention_mask_2 = paired_attention_mask[:, -1, :]
        
        mean_output_1 = self._get_raw_mean_embeddings(input_ids_1, attention_mask_1)
        mean_output_2 = self._get_raw_mean_embeddings(input_ids_2, attention_mask_2)

        cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
        return cnst_feat1, cnst_feat2, mean_output_1, mean_output_2

    def forward(self, input_ids, attention_mask, task_type, 
                future_input_ids=None, future_attention_mask=None):        
        
        if task_type == "evaluate":
            return self._get_raw_mean_embeddings(input_ids, attention_mask)

        # --- Teacher Path ---
        if self.is_teacher:
            if task_type != "distillation_teacher_forward":
                raise ValueError(f"Teacher model called with invalid task_type: {task_type}. Expected 'distillation_teacher_forward'.")
            if future_input_ids is None or future_attention_mask is None:
                raise ValueError("Teacher model in 'distillation_teacher_forward' mode requires future_input_ids and future_attention_mask.")
            
            mean_embeddings_teacher = self._get_teacher_combined_embeddings(input_ids, attention_mask, future_input_ids, future_attention_mask)
            projected_teacher_emb = self.distill_proj(mean_embeddings_teacher)
            return projected_teacher_emb

        # --- Student Path ---
        if task_type == "contrastive_learning":
            return self._get_contrastive_outputs(input_ids, attention_mask)

        elif task_type == "distillation_student_forward":
            mean_embeddings_student = self._get_raw_mean_embeddings(input_ids, attention_mask)
            projected_student_emb = self.distill_proj(mean_embeddings_student)
            return projected_student_emb

        elif task_type == "combined_learning_student_forward":
            cnst_feat1, cnst_feat2, mean_out1_for_distill, _ = self._get_contrastive_outputs(input_ids, attention_mask)
            projected_student_emb_combined = self.distill_proj(mean_out1_for_distill)
            return cnst_feat1, cnst_feat2, projected_student_emb_combined
            
        else:
            raise ValueError(f"Unknown task_type for student: {task_type}. Supported: 'contrastive_learning', 'distillation_student_forward', 'combined_learning_student_forward'.")
            
    # pass BERT embedding through the contrastive heads to get logits
    def contrast_logits(self, embd1, embd2):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        feat2 = F.normalize(self.contrast_head(embd2), dim=1)
        return feat1, feat2

    # calculate the embedding of an input sentence as the average embeddings of its tokens
    def get_mean_embeddings(self, input_ids, attention_mask):
        # This method remains for external compatibility if needed
        return self._get_raw_mean_embeddings(input_ids, attention_mask)
        
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
        self.base_model_prefix = "roberta" # Added

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))
            
        # Distillation projection layer
        self.distill_proj = nn.Linear(self.emb_size, self.emb_size, bias=False)
        
    def _get_raw_mean_embeddings(self, input_ids, attention_mask):
        # Helper to get mean embeddings from the base model for a single sequence
        base_model = getattr(self, self.base_model_prefix)
        model_output = base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = model_output[0] 
        
        expanded_attention_mask = attention_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(last_hidden_state * expanded_attention_mask, dim=1)
        sum_mask = torch.sum(expanded_attention_mask, dim=1)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def _get_teacher_combined_embeddings(self, input_ids_ctx, attention_mask_ctx, future_input_ids, future_attention_mask):
        # Helper for teacher: combines context and future utterance then gets mean embeddings.
        combined_input_ids = torch.cat([input_ids_ctx, future_input_ids], dim=1)
        combined_attention_mask = torch.cat([attention_mask_ctx, future_attention_mask], dim=1)
        return self._get_raw_mean_embeddings(combined_input_ids, combined_attention_mask)

    def _get_contrastive_outputs(self, paired_input_ids, paired_attention_mask):
        # Processes paired inputs for contrastive learning.
        if paired_input_ids.shape[1] == 2:
            input_ids_1, input_ids_2 = torch.unbind(paired_input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(paired_attention_mask, dim=1) 
        else: # Multi-turn case
            batch_size = paired_input_ids.shape[0]
            input_ids_1 = paired_input_ids[:, :-1, :].reshape(batch_size, -1)
            attention_mask_1 = paired_attention_mask[:, :-1, :].reshape(batch_size, -1)
            input_ids_2 = paired_input_ids[:, -1, :]
            attention_mask_2 = paired_attention_mask[:, -1, :]
        
        mean_output_1 = self._get_raw_mean_embeddings(input_ids_1, attention_mask_1)
        mean_output_2 = self._get_raw_mean_embeddings(input_ids_2, attention_mask_2)

        cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
        return cnst_feat1, cnst_feat2, mean_output_1, mean_output_2
        
    def forward(self, input_ids, attention_mask, task_type, 
                future_input_ids=None, future_attention_mask=None):        
        
        if task_type == "evaluate":
            return self._get_raw_mean_embeddings(input_ids, attention_mask)

        # --- Teacher Path ---
        if self.is_teacher:
            if task_type != "distillation_teacher_forward":
                raise ValueError(f"Teacher model called with invalid task_type: {task_type}. Expected 'distillation_teacher_forward'.")
            if future_input_ids is None or future_attention_mask is None:
                raise ValueError("Teacher model in 'distillation_teacher_forward' mode requires future_input_ids and future_attention_mask.")
            
            mean_embeddings_teacher = self._get_teacher_combined_embeddings(input_ids, attention_mask, future_input_ids, future_attention_mask)
            projected_teacher_emb = self.distill_proj(mean_embeddings_teacher)
            return projected_teacher_emb

        # --- Student Path ---
        if task_type == "contrastive_learning":
            return self._get_contrastive_outputs(input_ids, attention_mask)

        elif task_type == "distillation_student_forward":
            mean_embeddings_student = self._get_raw_mean_embeddings(input_ids, attention_mask)
            projected_student_emb = self.distill_proj(mean_embeddings_student)
            return projected_student_emb

        elif task_type == "combined_learning_student_forward":
            cnst_feat1, cnst_feat2, mean_out1_for_distill, _ = self._get_contrastive_outputs(input_ids, attention_mask)
            projected_student_emb_combined = self.distill_proj(mean_out1_for_distill)
            return cnst_feat1, cnst_feat2, projected_student_emb_combined
            
        else:
            raise ValueError(f"Unknown task_type for student: {task_type}. Supported: 'contrastive_learning', 'distillation_student_forward', 'combined_learning_student_forward'.")
            
    # pass BERT embedding through the contrastive heads to get logits
    def contrast_logits(self, embd1, embd2):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        feat2 = F.normalize(self.contrast_head(embd2), dim=1)
        return feat1, feat2

    # calculate the embedding of an input sentence as the average embeddings of its tokens
    def get_mean_embeddings(self, input_ids, attention_mask):
        # This method remains for external compatibility if needed
        return self._get_raw_mean_embeddings(input_ids, attention_mask)
        
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
        self.base_model_prefix = "distilbert" # Added

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))
            
        # Distillation projection layer
        self.distill_proj = nn.Linear(self.emb_size, self.emb_size, bias=False)
        
    def _get_raw_mean_embeddings(self, input_ids, attention_mask):
        # Helper to get mean embeddings from the base model for a single sequence
        base_model = getattr(self, self.base_model_prefix)
        # DistilBertModel output is a tuple, last_hidden_state is the first element
        model_output = base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = model_output[0] 
        
        expanded_attention_mask = attention_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(last_hidden_state * expanded_attention_mask, dim=1)
        sum_mask = torch.sum(expanded_attention_mask, dim=1)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def _get_teacher_combined_embeddings(self, input_ids_ctx, attention_mask_ctx, future_input_ids, future_attention_mask):
        # Helper for teacher: combines context and future utterance then gets mean embeddings.
        combined_input_ids = torch.cat([input_ids_ctx, future_input_ids], dim=1)
        combined_attention_mask = torch.cat([attention_mask_ctx, future_attention_mask], dim=1)
        return self._get_raw_mean_embeddings(combined_input_ids, combined_attention_mask)

    def _get_contrastive_outputs(self, paired_input_ids, paired_attention_mask):
        # Processes paired inputs for contrastive learning.
        if paired_input_ids.shape[1] == 2:
            input_ids_1, input_ids_2 = torch.unbind(paired_input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(paired_attention_mask, dim=1) 
        else: # Multi-turn case
            batch_size = paired_input_ids.shape[0]
            input_ids_1 = paired_input_ids[:, :-1, :].reshape(batch_size, -1)
            attention_mask_1 = paired_attention_mask[:, :-1, :].reshape(batch_size, -1)
            input_ids_2 = paired_input_ids[:, -1, :]
            attention_mask_2 = paired_attention_mask[:, -1, :]
        
        mean_output_1 = self._get_raw_mean_embeddings(input_ids_1, attention_mask_1)
        mean_output_2 = self._get_raw_mean_embeddings(input_ids_2, attention_mask_2)

        cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
        return cnst_feat1, cnst_feat2, mean_output_1, mean_output_2

    def forward(self, input_ids, attention_mask, task_type, 
                future_input_ids=None, future_attention_mask=None):        
        
        if task_type == "evaluate":
            return self._get_raw_mean_embeddings(input_ids, attention_mask)

        # --- Teacher Path ---
        if self.is_teacher:
            if task_type != "distillation_teacher_forward":
                raise ValueError(f"Teacher model called with invalid task_type: {task_type}. Expected 'distillation_teacher_forward'.")
            if future_input_ids is None or future_attention_mask is None:
                raise ValueError("Teacher model in 'distillation_teacher_forward' mode requires future_input_ids and future_attention_mask.")
            
            mean_embeddings_teacher = self._get_teacher_combined_embeddings(input_ids, attention_mask, future_input_ids, future_attention_mask)
            projected_teacher_emb = self.distill_proj(mean_embeddings_teacher)
            return projected_teacher_emb

        # --- Student Path ---
        if task_type == "contrastive_learning":
            return self._get_contrastive_outputs(input_ids, attention_mask)

        elif task_type == "distillation_student_forward":
            mean_embeddings_student = self._get_raw_mean_embeddings(input_ids, attention_mask)
            projected_student_emb = self.distill_proj(mean_embeddings_student)
            return projected_student_emb

        elif task_type == "combined_learning_student_forward":
            cnst_feat1, cnst_feat2, mean_out1_for_distill, _ = self._get_contrastive_outputs(input_ids, attention_mask)
            projected_student_emb_combined = self.distill_proj(mean_out1_for_distill)
            return cnst_feat1, cnst_feat2, projected_student_emb_combined
            
        else:
            raise ValueError(f"Unknown task_type for student: {task_type}. Supported: 'contrastive_learning', 'distillation_student_forward', 'combined_learning_student_forward'.")
            
    # pass BERT embedding through the contrastive heads to get logits
    def contrast_logits(self, embd1, embd2):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        feat2 = F.normalize(self.contrast_head(embd2), dim=1)
        return feat1, feat2

    # calculate the embedding of an input sentence as the average embeddings of its tokens
    def get_mean_embeddings(self, input_ids, attention_mask):
        # This method remains for external compatibility if needed
        return self._get_raw_mean_embeddings(input_ids, attention_mask)
        
    def get_distill_embeddings(self, embeddings):
        """Project embeddings for distillation"""
        return self.distill_proj(embeddings)
        
    def copy_parameters_from(self, source_model):
        """Copy parameters from source model to this model"""
        self.load_state_dict(source_model.state_dict())
        print("Model parameters copied successfully")