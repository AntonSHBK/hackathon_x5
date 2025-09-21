from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from transformers import BertForTokenClassification
from transformers.utils import ModelOutput
from torchcrf import CRF


@dataclass
class TokenClassifierCRFOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predictions: Optional[torch.LongTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    attention_mask: Optional[torch.LongTensor] = None


class BertForTokenClassificationCRF(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)
    
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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        loss, predictions = None, None
        if labels is not None:
            labels_for_crf = labels.clone()
            labels_for_crf[labels_for_crf == -100] = 0

            loss = -self.crf(
                emissions,
                labels_for_crf,
                mask=attention_mask.bool(),
                reduction="mean"
            )

        decoded = self.crf.decode(emissions, mask=attention_mask.bool())

        max_len = emissions.size(1)
        predictions_padded = torch.full(
            (len(decoded), max_len),
            fill_value=-100,
            dtype=torch.long,
            device=emissions.device,
        )
        for i, seq in enumerate(decoded):
            predictions_padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=emissions.device)

        
        if not return_dict:
            output = (emissions,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierCRFOutput(
            loss=loss,
            logits=emissions,
            predictions=predictions_padded,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask
        )
