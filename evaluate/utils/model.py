import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from transformers import BertModel, BertPreTrainedModel, DistilBertPreTrainedModel, DistilBertModel
#для hf моделей
from transformers import AutoModel, PreTrainedModel


class GeneralModelForSequenceClassification(PreTrainedModel):
    def __init__(self, config, model_name, weights, pooling="average"):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights = weights
        self.pooling = pooling
        self.model_name = model_name

        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
    
        if self.model_name == "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1":
            outputs = self.encoder(input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict
                            )
        elif self.model_name == "jxm/cde-small-v1":
            outputs = self.encoder(input_ids,
                                attention_mask=attention_mask,
                                dataset_input_ids = input_ids,
                                #dataset_attention_mask=attention_mask,
                                output_hidden_states=output_hidden_states
                                )
        else:
            outputs = self.encoder(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict
                                )

        if self.pooling == "average":
            attention_mask = attention_mask.unsqueeze(-1)
            pooled_output = torch.sum(outputs[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        elif self.pooling == "cls_nopool":
            pooled_output = outputs[0][:, 0, :]
        elif self.pooling == "cls":
            pooled_output = outputs[1] if len(outputs) > 1 else outputs[0][:, 0, :]
        else:
            raise ValueError("Choose args.pooling from ['cls', 'cls_nopool', 'average']")

        pooled_output = self.dropout(pooled_output)
        pooled_output = pooled_output.to(torch.float32)
        seq_logits = self.classifier(pooled_output)

        outputs = (seq_logits,) + outputs[2:] 
        if labels is not None:
            seq_loss_fct = nn.CrossEntropyLoss().cuda()
            loss = seq_loss_fct(seq_logits, labels)
            outputs = (loss,) + outputs

        return outputs


class GeneralModelForNaturalLanguageInference(PreTrainedModel):
    def __init__(self, config, model_name, weights, pooling="average"):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights = weights
        self.pooling = pooling

        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size*2, self.num_labels)

        self.init_weights()
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels=None):  
        context_output = self.get_embeddings(input_ids_1, attention_mask_1)
        response_output = self.get_embeddings(input_ids_2, attention_mask_2)
        context_output = self.dropout(context_output)
        response_output = self.dropout(response_output)
        seq_logits = self.classifier(torch.cat([context_output, response_output], dim=1))

        outputs = (seq_logits,)
        if labels is not None:
            seq_loss_fct = nn.CrossEntropyLoss().cuda()
            loss = seq_loss_fct(seq_logits, labels)
            outputs = (loss,) + outputs

        return outputs
    
    def get_embeddings(self, input_ids, attention_mask):
        if self.pooling == "average":
            outputs = self.encoder.forward(input_ids=input_ids, attention_mask=attention_mask)
            attention_mask = attention_mask.unsqueeze(-1)
            embeddings = torch.sum(outputs[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        elif self.pooling == "cls":
            outputs = self.encoder.forward(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs[1] if len(outputs) > 1 else outputs[0][:, 0, :]
        else:
            raise ValueError()
        
        return embeddings

class GeneralModelForDialogueActionPrediction(PreTrainedModel):
    def __init__(self, config, model_name, **kwargs):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.emb_size = config.hidden_size
        self.classifier_dropout = 0.2
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()
        
    def forward(self, input_ids, attention_mask, labels=None):        
        pooled_output = self.get_mean_embeddings(input_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        seq_logits = self.classifier(pooled_output)
        seq_logits = self.sigmoid(seq_logits)

        outputs = (seq_logits,)
        if labels is not None:
            seq_loss_fct = nn.BCELoss()
            loss = seq_loss_fct(seq_logits, labels.float())
            outputs = (loss,) + outputs

        return outputs
    
    def get_mean_embeddings(self, input_ids, attention_mask):
        outputs = self.encoder.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(outputs[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings

class GeneralModelForResponseSelection(PreTrainedModel):
    def __init__(self, config, model_name, **kwargs):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, method="mean"):  
        context_output = self.get_embeddings(input_ids_1, attention_mask_1, method=method)
        response_output = self.get_embeddings(input_ids_2, attention_mask_2, method=method)
        return context_output, response_output
    
    def get_embeddings(self, input_ids, attention_mask, method="mean"):
        if method == "mean":
            outputs = self.encoder.forward(input_ids=input_ids, attention_mask=attention_mask)
            attention_mask = attention_mask.unsqueeze(-1)
            embeddings = torch.sum(outputs[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        elif method == "cls":
            outputs = self.encoder.forward(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs[1] if len(outputs) > 1 else outputs[0][:, 0, :]
        else:
            raise ValueError()
        
        embeddings = embeddings.to(torch.float32)
        return embeddings



class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, weights, pooling="average"):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights = weights
        self.pooling = pooling

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict
                            )

        if self.pooling == "average":
            attention_mask = attention_mask.unsqueeze(-1)
            pooled_output = torch.sum(outputs[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        elif self.pooling == "cls_nopool":
            pooled_output = outputs[0][:, 0, :]
        elif self.pooling == "cls":
            pooled_output = outputs[1]
        else:
            raise ValueError("Choose args.pooling from ['cls', 'cls_nopool', 'average']")

        pooled_output = self.dropout(pooled_output)
        seq_logits = self.classifier(pooled_output)

        outputs = (seq_logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # calculate sequence classification loss
            seq_loss_fct = nn.CrossEntropyLoss().cuda()
            loss = seq_loss_fct(seq_logits, labels)

            outputs = (loss,) + outputs

        return outputs


class BertForNaturalLanguageInference(BertPreTrainedModel):
    def __init__(self, config, weights, pooling="average"):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights = weights
        self.pooling = pooling

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size*2, self.num_labels)

        self.init_weights()
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels=None):  
        context_output = self.get_embeddings(input_ids_1, attention_mask_1)
        response_output = self.get_embeddings(input_ids_2, attention_mask_2)
        context_output = self.dropout(context_output)
        response_output = self.dropout(response_output)
        seq_logits = self.classifier(torch.cat([context_output, response_output], dim=1))

        outputs = (seq_logits,)
        if labels is not None:
            # calculate sequence classification loss
            seq_loss_fct = nn.CrossEntropyLoss().cuda()
            loss = seq_loss_fct(seq_logits, labels)

            outputs = (loss,) + outputs

        return outputs
    
    def get_embeddings(self, input_ids, attention_mask):
        # mean embeddings
        if self.pooling == "average":
            bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
            attention_mask = attention_mask.unsqueeze(-1)
            embeddings = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        elif self.pooling == "cls":
            embeddings = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)[1]
        else:
            raise ValueError()
        
        return embeddings




    

class BertMultiTurnForDialogueActionPredictionConcat(BertPreTrainedModel):
    def __init__(self, config,  **kwargs):
        super(BertMultiTurnForDialogueActionPredictionConcat, self).__init__(config)
        self.bert = BertModel(config)
        self.emb_size = config.hidden_size
        self.classifier_dropout = 0.2
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()
        
    def forward(self, input_ids, attention_mask, labels=None):        
        pooled_output = self.get_mean_embeddings(input_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        seq_logits = self.classifier(pooled_output)
        seq_logits = self.sigmoid(seq_logits)

        outputs = (seq_logits,)  # add hidden states and attention if they are here
        if labels is not None:
            # calculate sequence classification loss
            seq_loss_fct = nn.BCELoss()
            loss = seq_loss_fct(seq_logits, labels.float())
            outputs = (loss,) + outputs

        return outputs

    
    def get_mean_embeddings(self, input_ids, attention_mask):
        # mean embeddings
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings



class BertMultiTurnForResponseSelectionConcat(BertPreTrainedModel):
    def __init__(self, config,  **kwargs):
        super(BertMultiTurnForResponseSelectionConcat, self).__init__(config)
        self.bert = BertModel(config)
        
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, method="mean"):  
        context_output = self.get_embeddings(input_ids_1, attention_mask_1, method=method)
        response_output = self.get_embeddings(input_ids_2, attention_mask_2, method=method)


        return context_output, response_output

    
    def get_embeddings(self, input_ids, attention_mask, method="mean"):
        # mean embeddings
        if method == "mean":
            bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
            attention_mask = attention_mask.unsqueeze(-1)
            embeddings = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        elif method == "cls":
            embeddings = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)[1]
        else:
            raise ValueError()
        
        return embeddings








class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config, weights, pooling="average"):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights = weights
        self.pooling = pooling

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None):
        outputs = self.distilbert(input_ids,
                            attention_mask=attention_mask)

        if self.pooling == "average":
            attention_mask = attention_mask.unsqueeze(-1)
            pooled_output = torch.sum(outputs[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        elif self.pooling == "cls_nopool":
            pooled_output = outputs[0][:, 0, :]
        elif self.pooling == "cls":
            pooled_output = outputs[1]
        else:
            raise ValueError("Choose args.pooling from ['cls', 'cls_nopool', 'average']")

        pooled_output = self.dropout(pooled_output)
        seq_logits = self.classifier(pooled_output)

        outputs = (seq_logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # calculate sequence classification loss
            seq_loss_fct = nn.CrossEntropyLoss().cuda()
            loss = seq_loss_fct(seq_logits, labels)

            outputs = (loss,) + outputs

        return outputs


class DistilBertForNaturalLanguageInference(DistilBertPreTrainedModel):
    def __init__(self, config, weights, pooling="average"):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights = weights
        self.pooling = pooling

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size*2, self.num_labels)

        self.init_weights()
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels=None):  
        context_output = self.get_embeddings(input_ids_1, attention_mask_1)
        response_output = self.get_embeddings(input_ids_2, attention_mask_2)
        context_output = self.dropout(context_output)
        response_output = self.dropout(response_output)
        seq_logits = self.classifier(torch.cat([context_output, response_output], dim=1))

        outputs = (seq_logits,)
        if labels is not None:
            # calculate sequence classification loss
            seq_loss_fct = nn.CrossEntropyLoss().cuda()
            loss = seq_loss_fct(seq_logits, labels)

            outputs = (loss,) + outputs

        return outputs
    
    def get_embeddings(self, input_ids, attention_mask):
        # mean embeddings
        if self.pooling == "average":
            bert_output = self.distilbert.forward(input_ids=input_ids, attention_mask=attention_mask)
            attention_mask = attention_mask.unsqueeze(-1)
            embeddings = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        elif self.pooling == "cls":
            embeddings = self.distilbert.forward(input_ids=input_ids, attention_mask=attention_mask)[1]
        else:
            raise ValueError()
        
        return embeddings




    

class DistilBertMultiTurnForDialogueActionPredictionConcat(DistilBertPreTrainedModel):
    def __init__(self, config,  **kwargs):
        super(DistilBertMultiTurnForDialogueActionPredictionConcat, self).__init__(config)
        self.distilbert = DistilBertModel(config)
        self.emb_size = config.hidden_size
        self.classifier_dropout = 0.2
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()
        
    def forward(self, input_ids, attention_mask, labels=None):        
        pooled_output = self.get_mean_embeddings(input_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        seq_logits = self.classifier(pooled_output)
        seq_logits = self.sigmoid(seq_logits)

        outputs = (seq_logits,)  # add hidden states and attention if they are here
        if labels is not None:
            # calculate sequence classification loss
            seq_loss_fct = nn.BCELoss()
            loss = seq_loss_fct(seq_logits, labels.float())
            outputs = (loss,) + outputs

        return outputs

    
    def get_mean_embeddings(self, input_ids, attention_mask):
        # mean embeddings
        bert_output = self.distilbert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings



class DistilBertMultiTurnForResponseSelectionConcat(DistilBertPreTrainedModel):
    def __init__(self, config,  **kwargs):
        super(DistilBertMultiTurnForResponseSelectionConcat, self).__init__(config)
        self.distilbert = DistilBertModel(config)
        
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, method="mean"):  
        context_output = self.get_embeddings(input_ids_1, attention_mask_1, method=method)
        response_output = self.get_embeddings(input_ids_2, attention_mask_2, method=method)


        return context_output, response_output

    
    def get_embeddings(self, input_ids, attention_mask, method="mean"):
        # mean embeddings
        if method == "mean":
            bert_output = self.distilbert.forward(input_ids=input_ids, attention_mask=attention_mask)
            attention_mask = attention_mask.unsqueeze(-1)
            embeddings = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        elif method == "cls":
            embeddings = self.distilbert.forward(input_ids=input_ids, attention_mask=attention_mask)[1]
        else:
            raise ValueError()
        
        return embeddings