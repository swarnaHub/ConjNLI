from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_transformers.modeling_bert import BertPreTrainedModel
from pytorch_transformers.modeling_roberta import RobertaConfig, RobertaModel, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

logger = logging.getLogger(__name__)


class RobertaClassificationHeadWithSRL(nn.Module):
    def __init__(self, config, srl_embed_dim):
        super(RobertaClassificationHeadWithSRL, self).__init__()
        self.dense = nn.Linear(config.hidden_size + 2 * srl_embed_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, SRL_embed_a, SRL_embed_b, **kwargs):
        x = features[:, 0, :]
        x = torch.cat((x, SRL_embed_a, SRL_embed_b), 1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassificationWithSRL(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassificationWithSRL, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.srl_embed_dim = 40
        self.classifier = RobertaClassificationHeadWithSRL(config, self.srl_embed_dim)
        self.linear = nn.Linear(config.hidden_size, self.srl_embed_dim)

    def forward(self, input_ids, SRL_embed_a, SRL_embed_b, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, self.linear(SRL_embed_a), self.linear(SRL_embed_b))

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
