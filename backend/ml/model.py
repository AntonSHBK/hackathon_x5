from dataclasses import dataclass
from inspect import signature
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.utils import ModelOutput

from ml.crf import CRF


@dataclass
class TokenClassifierCRFOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predictions: Optional[torch.LongTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    attention_mask: Optional[torch.LongTensor] = None



class AutoModelForTokenClassificationWithCRF(PreTrainedModel):
    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.backbone = AutoModel.from_config(config)
        dropout_prob = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=config.initializer_range)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
        self.reset_crf_parameters()

    def reset_crf_parameters(self):
        nn.init.uniform_(self.crf.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.crf.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.crf.transitions, -0.1, 0.1)
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> TokenClassifierCRFOutput:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }

        # Фильтруем только те, которые реально есть у backbone.forward
        sig = signature(self.backbone.forward)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        outputs = self.backbone(**filtered_kwargs)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        
        loss = None
        predictions = None

        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = 0  
            
            loss = -self.crf(
                emissions, 
                labels, 
                mask=attention_mask.bool(),
                reduction="token_mean"
            )

        predictions = self.crf.decode(emissions, mask=attention_mask.bool())
        
        max_len = emissions.size(1)
        pred_tensor = torch.full(
            (len(predictions), max_len), 
            fill_value=-100, 
            dtype=torch.long,
            device=emissions.device
        )
        for i, seq in enumerate(predictions):
            pred_tensor[i, :len(seq)] = torch.tensor(
                seq, dtype=torch.long, device=emissions.device)

        if not return_dict:
            output = (emissions,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierCRFOutput(
            loss=loss,
            logits=emissions,
            predictions=pred_tensor,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
        )