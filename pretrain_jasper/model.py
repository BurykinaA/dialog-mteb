import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, MultiheadAttention
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel, DistilBertPreTrainedModel, DistilBertModel, BertForMaskedLM

from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


class PSCBert(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_special_tokens=0,
                 cosine_loss_weight=10.0, similarity_loss_weight=200.0,
                 contrastive_loss_weight=20.0, contrastive_margin=0.5):
        super(PSCBert, self).__init__()
        self.student = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        self.teacher = BertModel.from_pretrained(model_name, output_hidden_states=True)
        
        if num_special_tokens > 0:
            self.student.resize_token_embeddings(self.student.config.vocab_size + num_special_tokens)
            self.teacher.resize_token_embeddings(self.teacher.config.vocab_size + num_special_tokens)

        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.cosine_loss_weight = cosine_loss_weight
        self.similarity_loss_weight = similarity_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_margin = contrastive_margin

        #self.update_teacher()

    def update_teacher(self):
        """
        Update the teacher model with the student model's weights.
        """
        self.teacher.load_state_dict(self.student.bert.state_dict(), strict=False)

    def forward(self, 
                context_input_ids, 
                context_attention_mask, 
                context_mlm_labels,
                full_input_ids, 
                full_attention_mask,
                **kwargs):
        
        student_outputs = self.student(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            labels=context_mlm_labels,
            output_hidden_states=True
        )
        mlm_loss = student_outputs.loss
        student_hidden_states = student_outputs.hidden_states
        student_embedding = student_hidden_states[-1][:, 0]

        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                output_hidden_states=True
            )
        teacher_hidden_states = teacher_outputs.hidden_states
        teacher_embedding = teacher_hidden_states[-1][:, 0]

        teacher_embedding_norm = F.normalize(teacher_embedding, p=2, dim=-1)
        student_embedding_norm = F.normalize(student_embedding, p=2, dim=-1)

        cosine_loss = self.cosine_embedding_loss(student_embedding, teacher_embedding_norm)
        
        teacher_similarity = teacher_embedding_norm @ teacher_embedding_norm.transpose(-1, -2)
        similarity_loss = self.pair_inbatch_similarity_loss(student_embedding_norm, teacher_similarity)
        
        contrastive_loss = self.contrastive_loss_with_hard_negatives(student_embedding_norm, teacher_embedding_norm, self.contrastive_margin)

        weighted_cosine_loss = cosine_loss * self.cosine_loss_weight
        weighted_similarity_loss = similarity_loss * self.similarity_loss_weight
        weighted_contrastive_loss = contrastive_loss * self.contrastive_loss_weight

        distillation_loss = weighted_cosine_loss + weighted_similarity_loss + weighted_contrastive_loss
        total_loss = mlm_loss + distillation_loss
        
        return {
            "loss": total_loss,
            "mlm_loss": mlm_loss,
            "distillation_loss": distillation_loss,
            "cosine_loss": weighted_cosine_loss,
            "similarity_loss": weighted_similarity_loss,
            "contrastive_loss": weighted_contrastive_loss,
        }

    def cosine_embedding_loss(self, student_embeddings, teacher_embeddings):
        student_embeddings = F.normalize(student_embeddings, p=2, dim=-1)
        target = torch.ones(student_embeddings.size(0), device=student_embeddings.device)
        loss = F.cosine_embedding_loss(student_embeddings, teacher_embeddings, target)
        return loss

    def pair_inbatch_similarity_loss(self, student_embeddings, teacher_similarity):
        student_similarity = student_embeddings @ student_embeddings.transpose(-1, -2)
        loss = F.mse_loss(student_similarity, teacher_similarity)
        return loss

    def contrastive_loss_with_hard_negatives(self, student_embeddings, teacher_embeddings, margin):
        scores = torch.matmul(student_embeddings, teacher_embeddings.T)
        batch_size = scores.size(0)
        
        positive_sim = torch.diag(scores)
        
        mask = torch.eye(batch_size, device=scores.device).bool()
        negative_scores = scores.masked_fill(mask, -float('inf'))
        
        hard_negative_sim = torch.max(negative_scores, dim=1)[0]
        
        loss = F.relu(margin - positive_sim + hard_negative_sim).mean()
        return loss


if __name__ == '__main__':
    from transformers import BertTokenizer
    import csv
    from dataloader import get_dataloader

    # Example usage
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    special_tokens = ['[USR]', '[SYS]']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    model = PSCBert(num_special_tokens=len(special_tokens))

    # Create a dummy data file for testing
    dummy_data = [
        ["Hello, I need help with my booking. Sure, what is your booking reference?", "It's a gift for my friend. Okay, I can help with that."]
    ]
    dummy_filename = 'dummy_data.tsv'
    with open(dummy_filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(dummy_data)
        
    dataloader = get_dataloader(dummy_filename, tokenizer, batch_size=2, max_len=128)
    
    batch = next(iter(dataloader))
    
    outputs = model(**batch)
    print("Total loss:", outputs['loss'].item())
    print("MLM loss:", outputs['mlm_loss'].item())
    print("Distillation loss:", outputs['distillation_loss'].item())
    print("Contrastive loss:", outputs['contrastive_loss'].item())
