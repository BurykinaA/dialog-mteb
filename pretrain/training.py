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
        self.classify_loss = nn.CrossEntropyLoss().cuda()
        self.distill_loss = nn.MSELoss().cuda()  # Distillation loss


        # Буферы для EWMA (экспоненциально взвешенного скользящего среднего) лоссов
        self.running_c_loss = torch.zeros(1, device='cuda')
        self.running_d_loss = torch.zeros(1, device='cuda')
        self.momentum = 0.9  # коэффициент для EWMA

        # Небольшая эпсилон, чтобы не делить на ноль
        self.eps = 1e-8

        
        # For FutureTOD algorithm
        self.use_distillation = self.args.use_distillation
        self.update_teacher_interval = self.args.update_teacher_interval
        
        print(f"\nUsing PSC_Trainer in {self.args.mode} mode, {self.args.contrast_type}\n")
        if self.use_distillation:
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
        """Prepare inputs for distillation using the same fields as contrastive learning"""
        # Use text1 as context and text2 as future
        context, future = batch['text1'], batch['text2']
        
        context_feat = self.get_batch_token(context)
        future_feat = self.get_batch_token(future)
        
        context_ids = context_feat['input_ids'].cuda()
        context_mask = context_feat['attention_mask'].cuda()
        future_ids = future_feat['input_ids'].cuda()
        future_mask = future_feat['attention_mask'].cuda()
        
        return context_ids, context_mask, future_ids, future_mask
    
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
            dist_loss = self.distill_loss(student_proj_emb, teacher_emb)
            
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
            dist_loss = self.distill_loss(student_proj_emb, teacher_emb)
            
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
            self.model.module.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
        else:
            save_dir = os.path.join(self.args.resPath, str(epoch+1))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.model.module.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

    def train(self):
        all_iter = self.args.epochs * len(self.train_loader)
        print('\n={}/{}=Iterations/Batches'.format(all_iter, len(self.train_loader)))

        self.model.train()
        if self.teacher_model:
            self.teacher_model.eval()
            
        epoch_iterator = tqdm(self.train_loader, desc="Iteration")

        # Создаем GradScaler, если включен fp16
        self.scaler = GradScaler() if self.args.mixed_precision == "fp16" else None

        for epoch in range(self.args.epochs):
            for j, batch in enumerate(epoch_iterator):
                # Подготовка данных и обучение в зависимости от режима
                if self.args.mode == 'combined':
                    # Подготовка данных для контрастивного обучения
                    if self.args.num_turn > 1:
                        contrastive_input_ids, contrastive_attention_mask, pairsimi = self.prepare_pairwise_input_multiturn_concatenate(batch)
                    else:
                        contrastive_input_ids, contrastive_attention_mask, pairsimi = self.prepare_pairwise_input(batch)
                    
                    # Подготовка данных для дистилляции
                    context_ids, context_mask, future_ids, future_mask = self.prepare_distillation_input(batch)
                    
                    # Обучение с комбинированными данными
                    losses = self.train_combined(
                        contrastive_input_ids, contrastive_attention_mask, pairsimi,
                        context_ids, context_mask, future_ids, future_mask
                    )
                    
                elif self.args.mode == 'distill':
                    # Подготовка данных для дистилляции
                    context_ids, context_mask, future_ids, future_mask = self.prepare_distillation_input(batch)
                    losses = self.train_distillation(context_ids, context_mask, future_ids, future_mask)
                    
                elif self.args.mode == 'contrastive':
                    # Подготовка данных для контрастивного обучения
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
            if self.use_distillation and (epoch + 1) % self.update_teacher_interval == 0:
                self.update_teacher()
                
            if self.args.save_model_every_epoch:
                self.save_model(epoch, best_dev=False)

        return None

    def train_combined(self, contrastive_input_ids, contrastive_attention_mask, pairsimi,
                       context_ids, context_mask, future_ids, future_mask):
        use_mixed_precision = self.args.mixed_precision in ["fp16", "bf16"]
        dtype = torch.float16 if self.args.mixed_precision == "fp16" else torch.bfloat16

        def forward_backward():
            # 1) контрастивная часть
            cnst_feat1, cnst_feat2, student_proj_emb_for_distill = self.model(
                input_ids=contrastive_input_ids.to(self.device),
                attention_mask=contrastive_attention_mask.to(self.device),
                task_type="combined_learning_student_forward"
            )
            contrastive_losses = self.psc_loss(cnst_feat1, cnst_feat2, pairsimi.to(self.device))
            contrastive_loss = contrastive_losses["instdisc_loss"]  # уже на self.device

            # 2) дистилляционная часть: учитель
            with torch.no_grad():
                teacher_emb_target = self.teacher_model(
                    input_ids=context_ids.to(self.device),
                    attention_mask=context_mask.to(self.device),
                    task_type="distillation_teacher_forward",
                    future_input_ids=future_ids.to(self.device),
                    future_attention_mask=future_mask.to(self.device)
                )
            dist_loss = self.distill_loss(student_proj_emb_for_distill, teacher_emb_target)

            # Обновляем EWMA (без градиента). Все тензоры на self.device.
            with torch.no_grad():
                self.running_c_loss.mul_(self.momentum).add_(
                    contrastive_loss.detach() * (1 - self.momentum)
                )
                self.running_d_loss.mul_(self.momentum).add_(
                    dist_loss.detach() * (1 - self.momentum)
                )

            # Считаем коэффициент (тоже тензор на self.device)
            coeff = (self.running_c_loss + self.eps) / (self.running_d_loss + self.eps)

            # Комбинированный лосс
            total_loss = contrastive_loss + self.args.distill_weight * coeff * dist_loss
            return total_loss, contrastive_loss, dist_loss, coeff

        # Собственно шаг оптимизации
        if not use_mixed_precision:
            total_loss, contrastive_loss, dist_loss, coeff = forward_backward()
            total_loss.backward()
            self.optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast(device_type="cuda", dtype=dtype):
                total_loss, contrastive_loss, dist_loss, coeff = forward_backward()
            if self.args.mixed_precision == "fp16":
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:  # bf16
                total_loss.backward()
                self.optimizer.step()

        self.optimizer.zero_grad()

        return {
            "instdisc_loss": contrastive_loss.item(),
            "distill_loss": dist_loss.item(),
            "coeff": coeff.item(),
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

    def train_distillation(self, context_ids, context_mask, future_ids, future_mask):
        """Обучение только с дистилляционной потерей"""
        use_mixed_precision = self.args.mixed_precision in ["fp16", "bf16"]
        dtype = torch.float16 if self.args.mixed_precision == "fp16" else torch.bfloat16
        
        # Функция для выполнения прямого прохода
        def forward():
            # Учительская модель: получает контекст + будущее
            with torch.no_grad():
                teacher_emb_target = self.teacher_model(
                    input_ids=context_ids, 
                    attention_mask=context_mask,
                    task_type="distillation_teacher_forward", # Updated task_type
                    future_input_ids=future_ids, 
                    future_attention_mask=future_mask
                )
            
            # Студенческая модель: получает только контекст, возвращает спроецированные эмбеддинги
            student_proj_emb = self.model(
                input_ids=context_ids, 
                attention_mask=context_mask, 
                task_type="distillation_student_forward" # Updated task_type
            )
            dist_loss = self.distill_loss(student_proj_emb, teacher_emb_target)
            return dist_loss
        
        # Обработка с учетом precision
        if not use_mixed_precision:
            dist_loss = forward()
            dist_loss.backward()
            self.optimizer.step()
        else:
            with autocast(device_type="cuda", dtype=dtype):
                dist_loss = forward()
                
            if self.args.mixed_precision == "fp16":
                self.scaler.scale(dist_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:  # bf16
                dist_loss.backward()
                self.optimizer.step()
        
        self.optimizer.zero_grad()
        return {"distill_loss": dist_loss.item()}
