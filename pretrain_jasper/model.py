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
                 triplet_loss_weight=20.0, triplet_margin=0.015):
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
        self.triplet_loss_weight = triplet_loss_weight
        self.triplet_margin = triplet_margin

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

        teacher_embedding = F.normalize(teacher_embedding, p=2, dim=-1)
        student_embedding_norm = F.normalize(student_embedding, p=2, dim=-1)

        cosine_loss = self.cosine_embedding_loss(student_embedding, teacher_embedding)
        
        teacher_similarity = teacher_embedding @ teacher_embedding.transpose(-1, -2)
        similarity_loss = self.pair_inbatch_similarity_loss(student_embedding_norm, teacher_similarity)
        
        triplet_label = torch.where(self.get_score_diff(teacher_embedding) < 0, 1, -1)
        triplet_loss = self.pair_inbatch_triplet_loss(student_embedding_norm, triplet_label)

        weighted_cosine_loss = cosine_loss * self.cosine_loss_weight
        weighted_similarity_loss = similarity_loss * self.similarity_loss_weight
        weighted_triplet_loss = triplet_loss * self.triplet_loss_weight

        distillation_loss = weighted_cosine_loss + weighted_similarity_loss + weighted_triplet_loss
        total_loss = mlm_loss + distillation_loss
        
        return {
            "loss": total_loss,
            "mlm_loss": mlm_loss,
            "distillation_loss": distillation_loss,
            "cosine_loss": weighted_cosine_loss,
            "similarity_loss": weighted_similarity_loss,
            "triplet_loss": weighted_triplet_loss,
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

    def pair_inbatch_triplet_loss(self, student_embeddings, triplet_label):
        loss = F.relu(self.get_score_diff(student_embeddings) * triplet_label + self.triplet_margin).mean()
        return loss

    def get_score_diff(self, embedding):
        scores = torch.matmul(embedding, embedding.T)
        scores = scores[torch.triu(torch.ones_like(scores), diagonal=1).bool()]
        score_diff = scores.reshape((1, -1)) - scores.reshape((-1, 1))
        score_diff = score_diff[torch.triu(torch.ones_like(score_diff), diagonal=1).bool()]
        return score_diff


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
