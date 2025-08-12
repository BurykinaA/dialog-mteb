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
    def __init__(self, model_name='bert-base-uncased', num_special_tokens=0):
        super(PSCBert, self).__init__()
        self.student = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        self.teacher = BertModel.from_pretrained(model_name, output_hidden_states=True)

        if num_special_tokens > 0:
            self.student.resize_token_embeddings(self.student.config.vocab_size + num_special_tokens)
            self.teacher.resize_token_embeddings(self.teacher.config.vocab_size + num_special_tokens)

        for param in self.teacher.parameters():
            param.requires_grad = False

        # self.update_teacher()

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
                full_attention_mask):
        student_outputs = self.student(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            labels=context_mlm_labels,
            output_hidden_states=True
        )
        mlm_loss = student_outputs.loss
        student_hidden = student_outputs.hidden_states[-1]
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                output_hidden_states=True
            )
        teacher_hidden = teacher_outputs.hidden_states[-1]

        # Distillation loss (Cosine Similarity)
        # We need to align student and teacher hidden states for the context part.
        seq_len = student_hidden.shape[1]
        teacher_hidden_context = teacher_hidden[:, :seq_len, :]

        # Debug: Print shapes and some values
        # print(f"Student hidden shape: {student_hidden.shape}")
        # print(f"Teacher hidden shape: {teacher_hidden.shape}")
        # print(f"Teacher context shape: {teacher_hidden_context.shape}")
        # print(f"Context attention mask shape: {context_attention_mask.shape}")

        # Compute cosine similarity between student and teacher hidden states
        cos_sim = F.cosine_similarity(student_hidden, teacher_hidden_context, dim=-1)

        # Mask out padded tokens from the similarity score
        masked_cos_sim = cos_sim * context_attention_mask.float()

        # Compute the mean similarity PER SEQUENCE, then average across batch
        num_valid_tokens_per_seq = context_attention_mask.sum(dim=1)  # [batch_size]
        mean_cos_sim_per_seq = masked_cos_sim.sum(dim=1) / num_valid_tokens_per_seq.clamp(min=1)  # [batch_size]
        mean_cos_sim = mean_cos_sim_per_seq.mean()  # Average across batch

        # The distillation loss encourages the similarity to be close to 1
        distillation_loss = 1 - mean_cos_sim

        return {
            "mlm_loss": mlm_loss,
            "distillation_loss": distillation_loss
        }

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
    