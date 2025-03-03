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
        
        self.psc_loss = HardConLoss(temperature=self.args.temperature, contrast_type=self.args.contrast_type).cuda()
        self.classify_loss = nn.CrossEntropyLoss().cuda()
        self.distill_loss = nn.MSELoss().cuda()  # Distillation loss
        
        # For FutureTOD algorithm
        self.use_distillation = self.args.use_distillation and self.teacher_model is not None
        self.update_teacher_interval = self.args.update_teacher_interval
        
        print("\nUsing PSC_Trainer, {}\n".format(self.args.contrast_type))
        if self.use_distillation:
            print("Using Teacher-Student Distillation with update interval: {}".format(self.update_teacher_interval))
        

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
        """Prepare inputs for distillation (context and future)"""
        context, future = batch['context'], batch['future']
        
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
                # Check if we're using distillation
                if self.use_distillation and 'context' in batch and 'future' in batch:
                    # FutureTOD distillation
                    context_ids, context_mask, future_ids, future_mask = self.prepare_distillation_input(batch)
                    losses = self.train_distillation_step(context_ids, context_mask, future_ids, future_mask)
                else:
                    # Regular contrastive training
                    if self.args.num_turn > 1:
                        input_ids, attention_mask, pairsimi = self.prepare_pairwise_input_multiturn_concatenate(batch)
                    else:
                        input_ids, attention_mask, pairsimi = self.prepare_pairwise_input(batch)
                    
                    losses = self.train_step(input_ids, attention_mask, pairsimi)

                statistics_log(self.args.tensorboard, losses=losses, global_step=self.gstep)

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

    def train_step(self, input_ids, attention_mask, pairsimi, speaker_query_labels=None, speaker_response_labels=None):
        # Определяем precision
        use_mixed_precision = self.args.mixed_precision in ["fp16", "bf16"]
        dtype = torch.float16 if self.args.mixed_precision == "fp16" else torch.bfloat16

        # Обычный float32, если mixed_precision выключен
        if not use_mixed_precision:
            feat1, feat2, _, _ = self.model(input_ids, attention_mask, task_type='contrastive')
            losses = self.psc_loss(feat1, feat2, pairsimi)
            loss = losses["instdisc_loss"]
            # with torch.autograd.detect_anomaly():
                # feat1, feat2, _, _ = self.model(input_ids, attention_mask, task_type='contrastive')
                # losses = self.psc_loss(feat1, feat2, pairsimi)
                # loss = losses["instdisc_loss"]


            #with torch.autograd.detect_anomaly():
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return losses

        # Используем mixed precision
        with autocast(device_type="cuda", dtype=dtype):
            feat1, feat2, _, _ = self.model(input_ids, attention_mask, task_type='contrastive')
            losses = self.psc_loss(feat1, feat2, pairsimi)
            loss = losses["instdisc_loss"]

        # FP16 требует GradScaler
        if self.args.mixed_precision == "fp16":
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:  # bf16 или float32
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad()
        return losses
