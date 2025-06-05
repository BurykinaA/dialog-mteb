import os
import sys
import csv
import numpy as np

import torch
import torch.nn as nn
from utils.contrastive_utils import HardConLoss
from utils.utils import statistics_log 


from torch.utils.data import DataLoader, SequentialSampler
from sklearn.preprocessing import normalize
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler



class PSCTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args, teacher_model=None):
        super(PSCTrainer, self).__init__()
        self.args = args
        self.model = model  # Student model
        self.teacher_model = teacher_model  # Teacher model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.task_type = self.args.mode
        self.gstep = 0
        self.dev_objective = -1
        self.device = torch.device("cuda")
        self.psc_loss = HardConLoss(temperature=self.args.temperature, contrast_type=self.args.contrast_type).cuda()
        self.classify_loss = nn.CrossEntropyLoss().cuda() # Can be used for MLM if model doesn't return loss
        self.distill_loss_fn_mse_per_layer = nn.MSELoss().cuda()  # Per-layer MSE for Ldis
        # self.distill_loss = nn.MSELoss().cuda() # Old single-layer distillation loss

        # For FutureTOD: Number of BERT layers to use for distillation
        self.num_distill_layers = getattr(args, 'num_distill_layers', 9) # Default to 12 (BERT-base)
        # For FutureTOD: MLM loss (typically CrossEntropyLoss, ignore_index=-100)
        # Assuming model might return MLM loss directly if labels are passed.
        # self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100).cuda()

        
        # For FutureTOD algorithm
        self.use_distillation = self.args.use_distillation # This flag is used for teacher updates
        self.update_teacher_interval = self.args.update_teacher_interval
        
        print(f"\nUsing PSC_Trainer in {self.args.mode} mode, {self.args.contrast_type}\n")
        if self.use_distillation or self.args.mode in ['distill', 'combined']:
            print(f"Distillation settings: num_distill_layers={self.num_distill_layers}, mlm_probability={getattr(args, 'mlm_probability', 0.15)}")
            if self.teacher_model:
                print(f"Using Teacher-Student Distillation with update interval: {self.update_teacher_interval}")
        

    def get_batch_token(self, text, max_length=-1):
        if max_length == -1:
            max_length = self.args.max_length

        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_pairwise_input(self, batch):
        text1, text2, pairsimi = batch['text1'], batch['text2'], batch['pairsimi'].cuda()
        feat1 = self.get_batch_token(text1)
        feat2 = self.get_batch_token(text2)

        
        input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
        return input_ids.cuda(), attention_mask.cuda(), pairsimi.detach()
    


    def prepare_pairwise_input_multiturn_concatenate(self, batch):
        text1, text2, pairsimi = batch['text1'], batch['text2'], batch['pairsimi'].cuda()
        max_query_length = self.args.num_turn * self.args.max_length
        num_keeped_words = int(max_query_length*0.9)
        text1 = [" ".join(t.split()[-num_keeped_words:]) for t in text1]
        feat1 = self.get_batch_token(text1, max_length=max_query_length)
        feat2 = self.get_batch_token(text2, max_length=32)


        batch_size = feat2['input_ids'].shape[0]
        seq_length = feat2['input_ids'].shape[1]



        input_ids = torch.cat([feat1['input_ids'].reshape(batch_size, -1, seq_length), feat2['input_ids'].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([feat1['attention_mask'].reshape(batch_size, -1, seq_length), feat2['attention_mask'].unsqueeze(1)], dim=1)
        return input_ids.cuda(), attention_mask.cuda(), pairsimi.detach()

    def prepare_distillation_input(self, batch):
        # Student MLM inputs (already batched and on device from dataloader)
        student_mlm_input_ids = batch['context_mlm_input_ids'].to(self.device)
        student_mlm_attention_mask = batch['context_mlm_attention_mask'].to(self.device)
        student_mlm_labels = batch['context_mlm_labels'].to(self.device)

        # Inputs for teacher (Context + Future)
        # batch['text1'] is a list of context strings
        # batch['text2'] (originally 'text_future') is a list of future utterance strings
        
        # --- Prepare inputs for Teacher (Context + Future) ---
        teacher_text_pairs = []
        for i in range(len(batch['text1'])):
            context_str = batch['text1'][i]
            future_str = batch['text2'][i] # Corrected to use 'text2' as per typical dataloader output
            teacher_text_pairs.append((context_str, future_str))

        teacher_inputs_tokenized = self.tokenizer(
            teacher_text_pairs,
            padding='longest',
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt",
            return_token_type_ids=True
        )
        teacher_combined_input_ids = teacher_inputs_tokenized['input_ids'].to(self.device)
        teacher_combined_attention_mask = teacher_inputs_tokenized['attention_mask'].to(self.device)
        teacher_combined_token_type_ids = teacher_inputs_tokenized.get('token_type_ids', None)
        if teacher_combined_token_type_ids is not None:
            teacher_combined_token_type_ids = teacher_combined_token_type_ids.to(self.device)

        # student_context_input_ids, student_context_attention_mask, student_context_token_type_ids
        # are no longer prepared here as student CLS embeddings for distillation
        # will be derived from the same forward pass as MLM (using student_mlm_input_ids).

        return (
            student_mlm_input_ids, student_mlm_attention_mask, student_mlm_labels,
            # No longer returning separate student_context_input_ids for unmasked context
            teacher_combined_input_ids, teacher_combined_attention_mask, teacher_combined_token_type_ids
        )
    
    def train_distillation_step(self, context_ids, context_mask, future_ids, future_mask):
        """Perform a distillation training step"""
        use_mixed_precision = self.args.mixed_precision in ["fp16", "bf16"]
        dtype = torch.float16 if self.args.mixed_precision == "fp16" else torch.bfloat16
        
        # Set teacher to eval mode
        self.teacher_model.eval()
        
        if not use_mixed_precision:
            # Get teacher embeddings (with context + future)
            with torch.no_grad():
                teacher_emb = self.teacher_model(context_ids, context_mask, 
                                               task_type='distill',
                                               future_input_ids=future_ids, 
                                               future_attention_mask=future_mask)
            
            # Get student embeddings (with context only)
            student_emb = self.model(context_ids, context_mask, task_type='distill')
            student_proj_emb = self.model.module.get_distill_embeddings(student_emb)
            
            # Calculate distillation loss
            dist_loss = self.distill_loss_fn_mse_per_layer(student_proj_emb, teacher_emb)
            
            # Calculate MLM loss if needed
            # mlm_loss = self.calculate_mlm_loss(context_ids, context_mask)
            # total_loss = dist_loss + mlm_loss
            total_loss = dist_loss
            
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return {"distill_loss": dist_loss.item()}
        
        # Mixed precision training
        with autocast(device_type="cuda", dtype=dtype):
            # Get teacher embeddings (with context + future)
            with torch.no_grad():
                teacher_emb = self.teacher_model(context_ids, context_mask, 
                                               task_type='distill',
                                               future_input_ids=future_ids, 
                                               future_attention_mask=future_mask)
            
            # Get student embeddings (with context only)
            student_emb = self.model(context_ids, context_mask, task_type='distill')
            student_proj_emb = self.model.module.get_distill_embeddings(student_emb)
            
            # Calculate distillation loss
            dist_loss = self.distill_loss_fn_mse_per_layer(student_proj_emb, teacher_emb)
            
            # Calculate MLM loss if needed
            # mlm_loss = self.calculate_mlm_loss(context_ids, context_mask)
            # total_loss = dist_loss + mlm_loss
            total_loss = dist_loss
        
        # FP16 requires GradScaler
        if self.args.mixed_precision == "fp16":
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:  # bf16 or float32
            total_loss.backward()
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        return {"distill_loss": dist_loss.item()}
    
    def update_teacher(self):
        """Update teacher model with student parameters"""
        self.teacher_model.module.copy_parameters_from(self.model.module)
        print("Teacher model updated with student parameters")

    def save_model(self, epoch, best_dev=False):
        if best_dev:
            save_dir = os.path.join(self.args.resPath, 'dev')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.model.module.save_pretrained(save_dir, safe_serialization=False)
            self.tokenizer.save_pretrained(save_dir)
        else:
            save_dir = os.path.join(self.args.resPath, str(epoch+1))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.model.module.save_pretrained(save_dir, safe_serialization=False)
            self.tokenizer.save_pretrained(save_dir)

    def train(self):
        all_iter = self.args.epochs * len(self.train_loader)
        print('\n={}/{}=Iterations/Batches'.format(all_iter, len(self.train_loader)))

        self.model.train()
        if self.teacher_model:
            self.teacher_model.eval()
            
        epoch_iterator = tqdm(self.train_loader, desc="Iteration")

        self.scaler = GradScaler() if self.args.mixed_precision == "fp16" else None

        for epoch in range(self.args.epochs):
            for j, batch in enumerate(epoch_iterator):
                if self.args.mode == 'combined':
                    if self.args.num_turn > 1:
                        contrastive_input_ids, contrastive_attention_mask, pairsimi = self.prepare_pairwise_input_multiturn_concatenate(batch)
                    else:
                        contrastive_input_ids, contrastive_attention_mask, pairsimi = self.prepare_pairwise_input(batch)
                    
                    student_mlm_input_ids, student_mlm_attention_mask, mlm_labels, \
                    teacher_combined_input_ids, teacher_combined_attention_mask, teacher_combined_token_type_ids = \
                        self.prepare_distillation_input(batch)
                    
                    losses = self.train_combined(
                        contrastive_input_ids, contrastive_attention_mask, pairsimi,
                        student_mlm_input_ids, student_mlm_attention_mask, mlm_labels,
                        teacher_combined_input_ids, teacher_combined_attention_mask, teacher_combined_token_type_ids
                    )
                    
                elif self.args.mode == 'distill':
                    student_mlm_input_ids, student_mlm_attention_mask, mlm_labels, \
                    teacher_combined_input_ids, teacher_combined_attention_mask, teacher_combined_token_type_ids = \
                        self.prepare_distillation_input(batch)
                    
                    losses = self.train_distillation(
                        student_mlm_input_ids, student_mlm_attention_mask, mlm_labels,
                        teacher_combined_input_ids, teacher_combined_attention_mask, teacher_combined_token_type_ids
                    )
                    
                elif self.args.mode == 'contrastive':
                    if self.args.num_turn > 1:
                        input_ids, attention_mask, pairsimi = self.prepare_pairwise_input_multiturn_concatenate(batch)
                    else:
                        input_ids, attention_mask, pairsimi = self.prepare_pairwise_input(batch)
                    
                    losses = self.train_contrastive(input_ids, attention_mask, pairsimi)

                statistics_log(losses=losses, global_step=self.gstep)

                if self.gstep > self.args.max_iter:
                    break

                self.gstep += 1

            print("Finish Epoch: ", epoch)
            
            # Update teacher model if needed
            if self.teacher_model and self.use_distillation and (epoch + 1) % self.update_teacher_interval == 0:
                self.update_teacher()
                
            # Save model every 10th epoch if self.args.save_model_every_epoch is True
            if self.args.save_model_every_epoch:
                if (epoch + 1) % 10 == 0:
                    self.save_model(epoch, best_dev=False)
                # Optionally, save the last epoch if it's not a multiple of 10 and total epochs is small,
                # or if you always want the final model. For now, strictly adhering to "every 10th epoch".
                # Example: if you want to save the very last one too:
                # elif (epoch + 1) == self.args.epochs:
                #     self.save_model(epoch, best_dev=False)

        return None

    def train_combined(self, contrastive_input_ids, contrastive_attention_mask, pairsimi,
                       student_mlm_input_ids, student_mlm_attention_mask, mlm_labels,
                       teacher_combined_input_ids, teacher_combined_attention_mask, teacher_combined_token_type_ids):
        use_mixed_precision = self.args.mixed_precision in ["fp16", "bf16"]
        dtype = torch.float16 if self.args.mixed_precision == "fp16" else torch.bfloat16

        def forward_backward():
            # 1) Контрастивная часть (студент)
            # Модель вернет: cnst_feat1, cnst_feat2, mean_output_1, mean_output_2
            # cnst_feat1, cnst_feat2, _, _ = self.model(... task_type="contrastive_learning")
            # For simplicity, assuming the model.forward for "combined_learning_student_forward" can handle this,
            # or it needs to be a separate call.
            # Let's assume self.model can handle input_ids of shape (batch, 2, seq_len) for contrastive.
            
            # For contrastive loss, student processes paired inputs
            # The current `task_type="combined_learning_student_forward"` in the original code for this model
            # returned `cnst_feat1, cnst_feat2, student_proj_emb_for_distill`.
            # We need to ensure the model's forward method is updated.
            # For now, let's assume the first two outputs are for contrastive loss.
            # And we will make a separate call for student's distillation-related outputs.

            # Contrastive part
            # Note: contrastive_input_ids might be shaped (batch_size, num_pairs=2, seq_len)
            # The model's "contrastive_learning" task type expects this.
            cnst_feat1_student, cnst_feat2_student, _, _ = self.model(
                input_ids=contrastive_input_ids.to(self.device),
                attention_mask=contrastive_attention_mask.to(self.device),
                task_type="contrastive_learning" 
            )
            contrastive_losses_dict = self.psc_loss(cnst_feat1_student, cnst_feat2_student, pairsimi.to(self.device))
            contrastive_loss = contrastive_losses_dict["instdisc_loss"]

            # 2) Дистилляционная часть (Ldis + Lmlm)
            # 2.1) Студент (контекст C) -> MLM loss, многослойные CLS эмбеддинги
            # Модель должна вернуть (mlm_loss, student_cls_hidden_states_list)
            # student_cls_hidden_states_list: список тензоров CLS эмбеддингов для каждого из self.num_distill_layers
            student_mlm_loss, student_cls_layers = self.model(
                input_ids=student_mlm_input_ids.to(self.device), # Use student_mlm_input_ids
                attention_mask=student_mlm_attention_mask.to(self.device), # Use student_mlm_attention_mask
                labels=mlm_labels.to(self.device) if mlm_labels is not None else None,
                task_type="student_distill_mlm" 
            )
            if mlm_labels is None and student_mlm_loss is None: 
                student_mlm_loss = torch.tensor(0.0, device=self.device)
            elif student_mlm_loss is None: # If model doesn't return MLM loss but labels were provided
                 # This case should ideally be handled by the model returning a loss or logits for manual calculation.
                 # For now, if student_mlm_loss is None but labels were present, we'll assume 0.
                 # A more robust solution would involve calculating MLM loss here if model returns logits.
                student_mlm_loss = torch.tensor(0.0, device=self.device)


            # 2.2) Учитель (контекст C + будущее F) -> многослойные CLS эмбеддинги
            # Модель должна вернуть (teacher_cls_hidden_states_list)
            with torch.no_grad():
                teacher_cls_layers = self.teacher_model(
                    input_ids=teacher_combined_input_ids.to(self.device), 
                    attention_mask=teacher_combined_attention_mask.to(self.device),
                    token_type_ids=teacher_combined_token_type_ids.to(self.device), # Pass token_type_ids
                    task_type="teacher_distill"
                    # future_input_ids and future_attention_mask are no longer needed here
                )

            # Расчет Ldis (сумма MSE по слоям)
            ldis_sum_layers = torch.tensor(0.0, device=self.device)
            if len(student_cls_layers) != len(teacher_cls_layers):
                # This should not happen if models are configured correctly for self.num_distill_layers
                print(f"Warning: Mismatch in number of layers for distillation. Student: {len(student_cls_layers)}, Teacher: {len(teacher_cls_layers)}")
            
            num_layers_to_distill = min(len(student_cls_layers), len(teacher_cls_layers), self.num_distill_layers)

            for i in range(num_layers_to_distill):
                s_layer_cls = student_cls_layers[i] # Assuming these are [batch_size, hidden_dim]
                t_layer_cls = teacher_cls_layers[i]
                ldis_sum_layers += self.distill_loss_fn_mse_per_layer(s_layer_cls, t_layer_cls)
            
            # Общий лосс FutureTOD компонента = Ldis + Lmlm
            futuretod_loss = ldis_sum_layers + student_mlm_loss

            # Комбинированный лосс
            # L_total = L_contrastive + weight_distill * (Ldis + Lmlm)
            total_loss = contrastive_loss + self.args.distill_weight * futuretod_loss
            
            return total_loss, contrastive_loss, ldis_sum_layers, student_mlm_loss

        # Собственно шаг оптимизации
        if not use_mixed_precision:
            total_loss, cl, dl, ml = forward_backward()
            total_loss.backward()
            self.optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast(device_type="cuda", dtype=dtype):
                total_loss, cl, dl, ml = forward_backward()
            if self.args.mixed_precision == "fp16":
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:  # bf16
                total_loss.backward()
                self.optimizer.step()

        self.optimizer.zero_grad()

        return {
            "instdisc_loss": cl.item(),
            "Ldis_sum_layers": dl.item(),
            "mlm_loss": ml.item() if isinstance(ml, torch.Tensor) else ml, 
            "futuretod_loss": (dl + (ml if isinstance(ml, torch.Tensor) else torch.tensor(ml, device=dl.device))).item(),
            "total_loss": total_loss.item()
        }

    def train_contrastive(self, input_ids, attention_mask, pairsimi):
        """Обучение только с контрастивной потерей"""
        use_mixed_precision = self.args.mixed_precision in ["fp16", "bf16"]
        dtype = torch.float16 if self.args.mixed_precision == "fp16" else torch.bfloat16
        
        # Функция для выполнения прямого прохода
        def forward():
            # Модель вернет: cnst_feat1, cnst_feat2, mean_output_1, mean_output_2
            cnst_feat1, cnst_feat2, _, _ = self.model(
                input_ids, 
                attention_mask, 
                task_type="contrastive_learning" # Updated task_type
            )
            losses = self.psc_loss(cnst_feat1, cnst_feat2, pairsimi)
            return losses, losses["instdisc_loss"]
        
        # Обработка с учетом precision
        if not use_mixed_precision:
            losses, loss = forward()
            loss.backward()
            self.optimizer.step()
        else:
            with autocast(device_type="cuda", dtype=dtype):
                losses, loss = forward()
                
            if self.args.mixed_precision == "fp16":
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:  # bf16
                loss.backward()
                self.optimizer.step()
        
        self.optimizer.zero_grad()
        return losses

    def train_distillation(self, student_mlm_input_ids, student_mlm_attention_mask, mlm_labels,
                           teacher_combined_input_ids, teacher_combined_attention_mask, teacher_combined_token_type_ids):
        """Обучение с дистилляцией (Ldis) и MLM (Lmlm) согласно FutureTOD"""
        use_mixed_precision = self.args.mixed_precision in ["fp16", "bf16"]
        dtype = torch.float16 if self.args.mixed_precision == "fp16" else torch.bfloat16
        
        self.model.train()
        if self.teacher_model:
            self.teacher_model.eval()

        def forward():
            # Студент (контекст C, обработанный для MLM) -> MLM loss, многослойные CLS эмбеддинги
            student_mlm_loss, student_cls_layers = self.model(
                input_ids=student_mlm_input_ids.to(self.device), 
                attention_mask=student_mlm_attention_mask.to(self.device),
                labels=mlm_labels.to(self.device) if mlm_labels is not None else None,
                task_type="student_distill_mlm" 
            )
            if mlm_labels is None and student_mlm_loss is None: 
                student_mlm_loss = torch.tensor(0.0, device=self.device)

            # Учитель (контекст C + будущее F, УЖЕ КОМБИНИРОВАННЫЕ) -> многослойные CLS эмбеддинги
            with torch.no_grad():
                teacher_cls_layers = self.teacher_model(
                    input_ids=teacher_combined_input_ids.to(self.device), 
                    attention_mask=teacher_combined_attention_mask.to(self.device),
                    token_type_ids=teacher_combined_token_type_ids.to(self.device), # Pass token_type_ids
                    task_type="teacher_distill"
                    # future_input_ids and future_attention_mask are no longer needed here
                )
            
            # Расчет Ldis (сумма MSE по слоям)
            ldis_sum_layers = torch.tensor(0.0, device=self.device)
            if len(student_cls_layers) != len(teacher_cls_layers):
                 print(f"Warning: Mismatch in number of layers for distillation. Student: {len(student_cls_layers)}, Teacher: {len(teacher_cls_layers)}")

            num_layers_to_distill = min(len(student_cls_layers), len(teacher_cls_layers), self.num_distill_layers)

            for i in range(num_layers_to_distill):
                s_layer_cls = student_cls_layers[i]
                t_layer_cls = teacher_cls_layers[i]
                ldis_sum_layers += self.distill_loss_fn_mse_per_layer(s_layer_cls, t_layer_cls)
            
            # Общий лосс = Ldis + Lmlm
            total_loss = ldis_sum_layers + student_mlm_loss
            return total_loss, ldis_sum_layers, student_mlm_loss
        
        # Обработка с учетом precision
        if not use_mixed_precision:
            total_loss, ldis, mlm = forward()
            total_loss.backward()
            self.optimizer.step()
        else:
            with autocast(device_type="cuda", dtype=dtype):
                total_loss, ldis, mlm = forward()
                
            if self.args.mixed_precision == "fp16":
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:  # bf16
                total_loss.backward()
                self.optimizer.step()
        
        self.optimizer.zero_grad()
        return {
            "total_loss": total_loss.item(),
            "Ldis_sum_layers": ldis.item(),
            "mlm_loss": mlm.item() if isinstance(mlm, torch.Tensor) else mlm
        }
