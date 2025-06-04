import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, MultiheadAttention
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel, DistilBertPreTrainedModel, DistilBertModel
from transformers import BertForMaskedLM, RobertaForMaskedLM, DistilBertForMaskedLM
from transformers import AutoModelForMaskedLM, AutoModel, AutoConfig
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
        # Try to load ForMaskedLM if it's a student and might do MLM
        # For teacher, base model is fine as it doesn't do MLM.
        if not is_teacher:
            try:
                self.model = AutoModelForMaskedLM.from_pretrained(model_name, config=self.config, trust_remote_code=True)
                print(f"CustomModel ({'Student' if not is_teacher else 'Teacher'}) loaded {model_name} as AutoModelForMaskedLM.")
            except Exception as e:
                print(f"Could not load {model_name} as AutoModelForMaskedLM for CustomModel, falling back to AutoModel. MLM loss might not be computed by model. Error: {e}")
                self.model = AutoModel.from_pretrained(model_name, config=self.config, trust_remote_code=True)
        else:
             self.model = AutoModel.from_pretrained(model_name, config=self.config, trust_remote_code=True)
             print(f"CustomModel (Teacher) loaded {model_name} as AutoModel.")


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
        # THIS IS FOR THE OLD DISTILLATION LOGIC (mean embeddings)
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

    def _prepare_teacher_input_ids_and_mask(self, context_ids, context_mask, future_ids, future_mask, tokenizer):
        # This method is now REMOVED / NO LONGER USED by model's forward.
        # Preparation is done in PSCTrainer.prepare_distillation_input
        pass

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None,
                task_type="contrastive_learning", 
                future_input_ids=None, # Kept for old task types, but not for new "teacher_distill"
                future_attention_mask=None # Kept for old task types
                ):

        if task_type == "evaluate":
            # get_mean_embeddings handles autocast internally
            return self.get_mean_embeddings(input_ids, attention_mask)


        # --- New FutureTOD-aligned Teacher Path for CustomModel ---
        if task_type == "teacher_distill":
            if self.is_teacher:
                # Teacher now expects combined input_ids, attention_mask, and token_type_ids
                # future_input_ids and future_attention_mask are NOT used here.

                model_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
                # Check if the underlying model uses token_type_ids
                if hasattr(self.model, 'forward') and 'token_type_ids' in self.model.forward.__code__.co_varnames:
                     if token_type_ids is not None:
                        model_kwargs['token_type_ids'] = token_type_ids
                     # else: print warning or ensure token_type_ids always passed if model expects it


                if self.autocast_dtype:
                    with autocast(device_type="cuda", dtype=self.autocast_dtype):
                        outputs = self.model(**model_kwargs, output_hidden_states=True, return_dict=True)
                else:
                    outputs = self.model(**model_kwargs, output_hidden_states=True, return_dict=True)

                if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                     raise ValueError("CustomModel's self.model did not return 'hidden_states'. Ensure it's a transformer model and output_hidden_states=True.")
                
                all_hidden_states = outputs.hidden_states
                cls_embeddings = [h_layer[:, 0, :] for h_layer in all_hidden_states[1:]]
                return cls_embeddings
            else: 
                 raise ValueError("Student model received 'teacher_distill' task_type.")

        # --- New FutureTOD-aligned Student Path for CustomModel ---
        elif task_type == "student_distill_mlm":
            if self.is_teacher:
                raise ValueError("Teacher model received 'student_distill_mlm' task_type.")

            model_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if hasattr(self.model, 'forward') and 'token_type_ids' in self.model.forward.__code__.co_varnames:
                 if token_type_ids is not None: # token_type_ids might not be passed for single sequence
                    model_kwargs['token_type_ids'] = token_type_ids
            
            # Pass labels if the model is AutoModelForMaskedLM and labels are provided
            can_compute_loss = hasattr(self.model, 'forward') and 'labels' in self.model.forward.__code__.co_varnames
            if labels is not None and can_compute_loss:
                model_kwargs['labels'] = labels

            if self.autocast_dtype:
                with autocast(device_type="cuda", dtype=self.autocast_dtype):
                    outputs = self.model(**model_kwargs, output_hidden_states=True, return_dict=True)
            else:
                outputs = self.model(**model_kwargs, output_hidden_states=True, return_dict=True)

            mlm_loss = torch.tensor(0.0).to(input_ids.device)
            if labels is not None and hasattr(outputs, 'loss') and outputs.loss is not None:
                mlm_loss = outputs.loss
            
            if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                 raise ValueError("CustomModel's self.model did not return 'hidden_states'. Ensure it's a transformer model and output_hidden_states=True.")

            all_hidden_states = outputs.hidden_states
            cls_embeddings = [h_layer[:, 0, :] for h_layer in all_hidden_states[1:]]
            return mlm_loss, cls_embeddings

        # --- Existing Teacher Path (for old distillation logic if ever used) ---
        if self.is_teacher: # This implies old task type `distillation_teacher_forward`
            if task_type == "distillation_teacher_forward": # Check if this is the task
                if future_input_ids is None or future_attention_mask is None: # Still needed for this old path
                    raise ValueError("Teacher model in 'distillation_teacher_forward' mode requires future_input_ids and future_attention_mask.")

                if self.autocast_dtype:
                    with autocast(device_type="cuda", dtype=self.autocast_dtype):
                        # This uses the OLD _get_combined_mean_embeddings with separate future inputs
                        mean_embeddings_teacher = self._get_combined_mean_embeddings(input_ids, attention_mask, future_input_ids, future_attention_mask)
                        projected_teacher_emb = self.distill_proj(mean_embeddings_teacher)
                else:
                    mean_embeddings_teacher = self._get_combined_mean_embeddings(input_ids, attention_mask, future_input_ids, future_attention_mask)
                    projected_teacher_emb = self.distill_proj(mean_embeddings_teacher)
                return projected_teacher_emb
            # If task_type is not "distillation_teacher_forward" and is_teacher, it should have been handled by "teacher_distill"
            # or it's an invalid state if it reaches here for a teacher.

        # --- Existing Student Paths ---
        if task_type == "contrastive_learning":
            return self._compute_forward(input_ids, attention_mask)

        elif task_type == "distillation_student_forward": # Old distillation logic
            mean_embeddings_student = self.get_mean_embeddings(input_ids, attention_mask)
            if self.autocast_dtype: 
                with autocast(device_type="cuda", dtype=self.autocast_dtype):
                    projected_student_emb = self.distill_proj(mean_embeddings_student)
            else:
                projected_student_emb = self.distill_proj(mean_embeddings_student)
            return projected_student_emb

        elif task_type == "combined_learning_student_forward": # Old combined logic
            cnst_feat1, cnst_feat2, mean_out1_for_distill, _ = self._compute_forward(input_ids, attention_mask)
            if self.autocast_dtype:
                with autocast(device_type="cuda", dtype=self.autocast_dtype):
                    projected_student_emb_combined = self.distill_proj(mean_out1_for_distill)
            else:
                projected_student_emb_combined = self.distill_proj(mean_out1_for_distill)
            return cnst_feat1, cnst_feat2, projected_student_emb_combined
            
        else:
            raise ValueError(f"Unknown task_type for student: {task_type}. Supported task types have been updated.")

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
        print(f"-----Initializing {'Teacher' if is_teacher else 'Student'} PSCBert (using BertForMaskedLM if student)-----")
        if not is_teacher:
            self.bert = BertForMaskedLM(config) # For MLM loss and hidden states
        else:
            self.bert = BertModel(config) # Teacher doesn't compute MLM, just needs hidden states

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
        model_output = base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # For BertForMaskedLM, the logits are output[0], for BertModel, last_hidden_state is output[0]
        # We need last_hidden_state. If it's BertForMaskedLM, this needs care.
        # BertForMaskedLM output: (loss), logits, hidden_states, attentions
        # BertModel output: last_hidden_state, pooler_output, (hidden_states), (attentions)
        # The PSCTrainer needs CLS embeddings, not mean. This method is for OLD logic.
        last_hidden_state = model_output.last_hidden_state if hasattr(model_output, 'last_hidden_state') else model_output.hidden_states[-1]

        expanded_attention_mask = attention_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(last_hidden_state * expanded_attention_mask, dim=1)
        sum_mask = torch.sum(expanded_attention_mask, dim=1)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def _get_teacher_combined_embeddings(self, input_ids_ctx, attention_mask_ctx, future_input_ids, future_attention_mask):
        # Helper for teacher: combines context and future utterance then gets mean embeddings.
        # THIS IS FOR THE OLD DISTILLATION LOGIC (mean embeddings)
        combined_input_ids = torch.cat([input_ids_ctx, future_input_ids], dim=1)
        combined_attention_mask = torch.cat([attention_mask_ctx, future_attention_mask], dim=1)
        return self._get_raw_mean_embeddings(combined_input_ids, combined_attention_mask)

    def _get_contrastive_outputs(self, paired_input_ids, paired_attention_mask, token_type_ids=None): # Added token_type_ids
        # Processes paired inputs for contrastive learning.
        if paired_input_ids.shape[1] == 2:
            input_ids_1, input_ids_2 = torch.unbind(paired_input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(paired_attention_mask, dim=1)
            token_type_ids_1, token_type_ids_2 = (None, None) # Initialize
            if token_type_ids is not None:
                token_type_ids_1, token_type_ids_2 = torch.unbind(token_type_ids, dim=1)

        else: # Multi-turn case
            batch_size = paired_input_ids.shape[0]
            input_ids_1 = paired_input_ids[:, :-1, :].reshape(batch_size, -1)
            attention_mask_1 = paired_attention_mask[:, :-1, :].reshape(batch_size, -1)
            input_ids_2 = paired_input_ids[:, -1, :]
            attention_mask_2 = paired_attention_mask[:, -1, :]
            token_type_ids_1, token_type_ids_2 = (None, None)
        
        outputs1 = self.bert(input_ids_1, attention_mask=attention_mask_1, token_type_ids=token_type_ids_1, return_dict=True)
        outputs2 = self.bert(input_ids_2, attention_mask=attention_mask_2, token_type_ids=token_type_ids_2, return_dict=True)

        last_hidden1 = outputs1.hidden_states[-1] if hasattr(outputs1, 'hidden_states') and outputs1.hidden_states is not None else outputs1.last_hidden_state
        last_hidden2 = outputs2.hidden_states[-1] if hasattr(outputs2, 'hidden_states') and outputs2.hidden_states is not None else outputs2.last_hidden_state
        
        mean_output_1 = last_hidden1[:, 0, :] # CLS token
        mean_output_2 = last_hidden2[:, 0, :] # CLS token

        cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
        return cnst_feat1, cnst_feat2, mean_output_1, mean_output_2

    def _prepare_teacher_input(self, context_ids, context_mask, future_ids, future_mask):
        # This method is now REMOVED / NO LONGER USED.
        # Preparation is done in PSCTrainer.prepare_distillation_input
        pass

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None,
                task_type="contrastive_learning", 
                future_input_ids=None, # Kept for old task types, not for new "teacher_distill"
                future_attention_mask=None # Kept for old task types
                ):        
        
        if task_type == "evaluate": 
            return self._get_raw_mean_embeddings(input_ids, attention_mask)

        # --- New FutureTOD Teacher Path ---
        if task_type == "teacher_distill":
            if not self.is_teacher:
                raise ValueError("Student model received 'teacher_distill' task_type.")
            # Teacher now expects combined input_ids, attention_mask, and token_type_ids
            # future_input_ids and future_attention_mask are NOT used here.
            
            outputs = self.bert( # self.bert is BertModel for teacher
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids, # Use the combined token_type_ids
                output_hidden_states=True,
                return_dict=True
            )
            all_hidden_states = outputs.hidden_states 
            if all_hidden_states is None: 
                 raise ValueError("Teacher model did not return hidden_states.")
            cls_embeddings = [h_layer[:, 0, :] for h_layer in all_hidden_states[1:]] 
            return cls_embeddings


        # --- New FutureTOD Student Path ---
        elif task_type == "student_distill_mlm":
            if self.is_teacher:
                 raise ValueError("Teacher model received 'student_distill_mlm' task_type.")
            # self.bert is BertForMaskedLM for student
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids, # Pass if available (e.g., for NSP pre-training style)
                labels=labels,
                output_hidden_states=True,
                return_dict=True
            )
            mlm_loss = outputs.loss if labels is not None and outputs.loss is not None else torch.tensor(0.0).to(input_ids.device)
            
            all_hidden_states = outputs.hidden_states
            if all_hidden_states is None:
                 raise ValueError("Student model did not return hidden_states.")
            cls_embeddings = [h_layer[:, 0, :] for h_layer in all_hidden_states[1:]]
            return mlm_loss, cls_embeddings


        # --- Teacher Path (OLD, for mean embedding distillation) ---
        elif self.is_teacher: # Implies task_type == "distillation_teacher_forward"
            if task_type == "distillation_teacher_forward":
                if future_input_ids is None or future_attention_mask is None:
                    raise ValueError("Teacher model (BertForMaskedLM) in 'distillation_teacher_forward' mode requires future_input_ids and future_attention_mask.")
                mean_embeddings_teacher = self._get_teacher_combined_embeddings(input_ids, attention_mask, future_input_ids, future_attention_mask)
                projected_teacher_emb = self.distill_proj(mean_embeddings_teacher)
                return projected_teacher_emb
            # If task_type is not "distillation_teacher_forward" and is_teacher, it should have been handled by "teacher_distill"
            # or it's an invalid state if it reaches here for a teacher.

        # --- Student Path (Existing) ---
        elif task_type == "contrastive_learning":
            return self._get_contrastive_outputs(input_ids, attention_mask, token_type_ids) # Pass token_type_ids

        elif task_type == "distillation_student_forward": # OLD distillation
            mean_embeddings_student = self._get_raw_mean_embeddings(input_ids, attention_mask)
            projected_student_emb = self.distill_proj(mean_embeddings_student)
            return projected_student_emb

        elif task_type == "combined_learning_student_forward": # OLD combined
            cnst_feat1, cnst_feat2, mean_out1_for_distill, _ = self._get_contrastive_outputs(input_ids, attention_mask, token_type_ids)
            projected_student_emb_combined = self.distill_proj(mean_out1_for_distill)
            return cnst_feat1, cnst_feat2, projected_student_emb_combined
            
        else:
            raise ValueError(f"Unknown task_type for PSCBert student: {task_type}.")
            
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
        print(f"-----Initializing {'Teacher' if is_teacher else 'Student'} PSCRoberta (using RobertaForMaskedLM if student)-----")
        if not is_teacher:
            self.roberta = RobertaForMaskedLM(config)
        else:
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
        model_output = base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = model_output.last_hidden_state if hasattr(model_output, 'last_hidden_state') else model_output.hidden_states[-1]
        
        expanded_attention_mask = attention_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(last_hidden_state * expanded_attention_mask, dim=1)
        sum_mask = torch.sum(expanded_attention_mask, dim=1)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def _get_teacher_combined_embeddings(self, input_ids_ctx, attention_mask_ctx, future_input_ids, future_attention_mask):
        # Helper for teacher: combines context and future utterance then gets mean embeddings.
        # THIS IS FOR THE OLD DISTILLATION LOGIC (mean embeddings)
        combined_input_ids = torch.cat([input_ids_ctx, future_input_ids], dim=1)
        combined_attention_mask = torch.cat([attention_mask_ctx, future_attention_mask], dim=1)
        return self._get_raw_mean_embeddings(combined_input_ids, combined_attention_mask)

    def _get_contrastive_outputs(self, paired_input_ids, paired_attention_mask, token_type_ids=None): # Added token_type_ids
        if paired_input_ids.shape[1] == 2:
            input_ids_1, input_ids_2 = torch.unbind(paired_input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(paired_attention_mask, dim=1)
            token_type_ids_1, token_type_ids_2 = (None, None)
        else: 
            batch_size = paired_input_ids.shape[0]
            input_ids_1 = paired_input_ids[:, :-1, :].reshape(batch_size, -1)
            attention_mask_1 = paired_attention_mask[:, :-1, :].reshape(batch_size, -1)
            input_ids_2 = paired_input_ids[:, -1, :]
            attention_mask_2 = paired_attention_mask[:, -1, :]
            token_type_ids_1, token_type_ids_2 = (None, None)
        
        outputs1 = self.roberta(input_ids_1, attention_mask=attention_mask_1, return_dict=True)
        outputs2 = self.roberta(input_ids_2, attention_mask=attention_mask_2, return_dict=True)

        last_hidden1 = outputs1.hidden_states[-1] if hasattr(outputs1, 'hidden_states') and outputs1.hidden_states is not None else outputs1.last_hidden_state
        last_hidden2 = outputs2.hidden_states[-1] if hasattr(outputs2, 'hidden_states') and outputs2.hidden_states is not None else outputs2.last_hidden_state
        
        mean_output_1 = last_hidden1[:, 0, :] # CLS token
        mean_output_2 = last_hidden2[:, 0, :] # CLS token

        cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
        return cnst_feat1, cnst_feat2, mean_output_1, mean_output_2

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, 
                task_type="contrastive_learning", 
                future_input_ids=None, # Kept for old task types
                future_attention_mask=None # Kept for old task types
                ):        
        # Roberta generally ignores token_type_ids, but we accept it for consistent signature
        
        if task_type == "evaluate":
            return self._get_raw_mean_embeddings(input_ids, attention_mask) 

        if task_type == "teacher_distill":
            if not self.is_teacher:
                raise ValueError("Student model received 'teacher_distill' task_type.")
            # Teacher receives combined inputs
            outputs = self.roberta( # self.roberta is RobertaModel for teacher
                input_ids=input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids, # Roberta generally ignores this
                output_hidden_states=True,
                return_dict=True
            )
            all_hidden_states = outputs.hidden_states
            if all_hidden_states is None: raise ValueError("Teacher Roberta model did not return hidden_states.")
            cls_embeddings = [h_layer[:, 0, :] for h_layer in all_hidden_states[1:]]
            return cls_embeddings

        elif task_type == "student_distill_mlm":
            if self.is_teacher:
                 raise ValueError("Teacher model received 'student_distill_mlm' task_type.")
            # self.roberta is RobertaForMaskedLM for student
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids, # Roberta generally ignores token_type_ids
                labels=labels,
                output_hidden_states=True,
                return_dict=True
            )
            mlm_loss = outputs.loss if labels is not None and outputs.loss is not None else torch.tensor(0.0).to(input_ids.device)
            all_hidden_states = outputs.hidden_states
            if all_hidden_states is None: raise ValueError("Student Roberta model did not return hidden_states.")
            cls_embeddings = [h_layer[:, 0, :] for h_layer in all_hidden_states[1:]]
            return mlm_loss, cls_embeddings
        
        # --- Teacher Path (OLD, for mean embedding distillation) ---
        elif self.is_teacher: # Implies task_type == "distillation_teacher_forward"
            if task_type == "distillation_teacher_forward":
                if future_input_ids is None or future_attention_mask is None:
                    raise ValueError("Teacher model (Roberta) in 'distillation_teacher_forward' mode requires future_input_ids and future_attention_mask.")
                mean_embeddings_teacher = self._get_teacher_combined_embeddings(input_ids, attention_mask, future_input_ids, future_attention_mask)
                projected_teacher_emb = self.distill_proj(mean_embeddings_teacher)
                return projected_teacher_emb

        # --- Student Path (Existing) ---
        elif task_type == "contrastive_learning":
            return self._get_contrastive_outputs(input_ids, attention_mask, token_type_ids) # Pass token_type_ids

        elif task_type == "distillation_student_forward": # OLD distillation
            mean_embeddings_student = self._get_raw_mean_embeddings(input_ids, attention_mask)
            projected_student_emb = self.distill_proj(mean_embeddings_student)
            return projected_student_emb

        elif task_type == "combined_learning_student_forward": # OLD combined
            cnst_feat1, cnst_feat2, mean_out1_for_distill, _ = self._get_contrastive_outputs(input_ids, attention_mask, token_type_ids)
            projected_student_emb_combined = self.distill_proj(mean_out1_for_distill)
            return cnst_feat1, cnst_feat2, projected_student_emb_combined
            
        else:
            raise ValueError(f"Unknown task_type for PSCRoberta student: {task_type}.")
            
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
        print(f"-----Initializing {'Teacher' if is_teacher else 'Student'} PSCDistilBERT (using DistilBertForMaskedLM if student)-----")
        if not is_teacher:
            self.distilbert = DistilBertForMaskedLM(config)
        else:
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
        # THIS IS FOR THE OLD DISTILLATION LOGIC (mean embeddings)
        combined_input_ids = torch.cat([input_ids_ctx, future_input_ids], dim=1)
        combined_attention_mask = torch.cat([attention_mask_ctx, future_attention_mask], dim=1)
        return self._get_raw_mean_embeddings(combined_input_ids, combined_attention_mask)

    def _get_contrastive_outputs(self, paired_input_ids, paired_attention_mask, token_type_ids=None): # Added token_type_ids
        # DistilBERT does not use token_type_ids.
        if paired_input_ids.shape[1] == 2:
            input_ids_1, input_ids_2 = torch.unbind(paired_input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(paired_attention_mask, dim=1) 
        else: 
            batch_size = paired_input_ids.shape[0]
            input_ids_1 = paired_input_ids[:, :-1, :].reshape(batch_size, -1)
            attention_mask_1 = paired_attention_mask[:, :-1, :].reshape(batch_size, -1)
            input_ids_2 = paired_input_ids[:, -1, :]
            attention_mask_2 = paired_attention_mask[:, -1, :]
        
        outputs1 = self.distilbert(input_ids_1, attention_mask=attention_mask_1, return_dict=True)
        outputs2 = self.distilbert(input_ids_2, attention_mask=attention_mask_2, return_dict=True)

        last_hidden1 = outputs1.hidden_states[-1] if hasattr(outputs1, 'hidden_states') and outputs1.hidden_states is not None else outputs1.last_hidden_state
        last_hidden2 = outputs2.hidden_states[-1] if hasattr(outputs2, 'hidden_states') and outputs2.hidden_states is not None else outputs2.last_hidden_state
        
        mean_output_1 = last_hidden1[:, 0, :] # CLS token
        mean_output_2 = last_hidden2[:, 0, :] # CLS token

        cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
        return cnst_feat1, cnst_feat2, mean_output_1, mean_output_2

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, 
                task_type="contrastive_learning", 
                future_input_ids=None, # Kept for old task types
                future_attention_mask=None # Kept for old task types
                ):        
        # DistilBERT does not use token_type_ids, they will be ignored if passed.
        
        if task_type == "evaluate":
            return self._get_raw_mean_embeddings(input_ids, attention_mask) 

        if task_type == "teacher_distill":
            if not self.is_teacher:
                raise ValueError("Student model received 'teacher_distill' task_type.")
            # Teacher receives combined inputs
            outputs = self.distilbert( # self.distilbert is DistilBertModel for teacher
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            all_hidden_states = outputs.hidden_states
            if all_hidden_states is None: raise ValueError("Teacher DistilBERT model did not return hidden_states.")
            cls_embeddings = [h_layer[:, 0, :] for h_layer in all_hidden_states[1:]]
            return cls_embeddings

        elif task_type == "student_distill_mlm":
            if self.is_teacher:
                 raise ValueError("Teacher model received 'student_distill_mlm' task_type.")
            # self.distilbert is DistilBertForMaskedLM for student
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True
            )
            mlm_loss = outputs.loss if labels is not None and outputs.loss is not None else torch.tensor(0.0).to(input_ids.device)
            all_hidden_states = outputs.hidden_states
            if all_hidden_states is None: raise ValueError("Student DistilBERT model did not return hidden_states.")
            cls_embeddings = [h_layer[:, 0, :] for h_layer in all_hidden_states[1:]] # CLS is index 0 for DistilBERT too
            return mlm_loss, cls_embeddings

        # --- Teacher Path (OLD, for mean embedding distillation) ---
        elif self.is_teacher: # Implies task_type == "distillation_teacher_forward"
            if task_type == "distillation_teacher_forward":
                if future_input_ids is None or future_attention_mask is None:
                     raise ValueError(f"Teacher model (DistilBERT) in 'distillation_teacher_forward' mode requires future_input_ids and future_attention_mask.")
                mean_embeddings_teacher = self._get_teacher_combined_embeddings(input_ids, attention_mask, future_input_ids, future_attention_mask)
                projected_teacher_emb = self.distill_proj(mean_embeddings_teacher)
                return projected_teacher_emb

        # --- Student Path (Existing) ---
        elif task_type == "contrastive_learning":
            return self._get_contrastive_outputs(input_ids, attention_mask) # DistilBERT doesn't use token_type_ids

        elif task_type == "distillation_student_forward": # OLD distillation
            mean_embeddings_student = self._get_raw_mean_embeddings(input_ids, attention_mask)
            projected_student_emb = self.distill_proj(mean_embeddings_student)
            return projected_student_emb

        elif task_type == "combined_learning_student_forward": # OLD combined
            cnst_feat1, cnst_feat2, mean_out1_for_distill, _ = self._get_contrastive_outputs(input_ids, attention_mask)
            projected_student_emb_combined = self.distill_proj(mean_out1_for_distill)
            return cnst_feat1, cnst_feat2, projected_student_emb_combined
            
        else:
            raise ValueError(f"Unknown task_type for PSCDistilBERT student: {task_type}.")
            
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