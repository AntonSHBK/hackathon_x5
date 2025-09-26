from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
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



class AutoModelForTokenClassificationWithCRF(PreTrainedModel):
    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.backbone = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
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
        
        outputs = self.backbone(
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
        # emissions = emissions.clamp(-20, 20)
        
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
        
        
# __version__ = '0.7.2'

# from typing import List, Optional

# import torch
# import torch.nn as nn


# class CRF(nn.Module):
#     """Conditional random field.

#     This module implements a conditional random field [LMP01]_. The forward computation
#     of this class computes the log likelihood of the given sequence of tags and
#     emission score tensor. This class also has `~CRF.decode` method which finds
#     the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

#     Args:
#         num_tags: Number of tags.
#         batch_first: Whether the first dimension corresponds to the size of a minibatch.

#     Attributes:
#         start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
#             ``(num_tags,)``.
#         end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
#             ``(num_tags,)``.
#         transitions (`~torch.nn.Parameter`): Transition score tensor of size
#             ``(num_tags, num_tags)``.


#     .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
#        "Conditional random fields: Probabilistic models for segmenting and
#        labeling sequence data". *Proc. 18th International Conf. on Machine
#        Learning*. Morgan Kaufmann. pp. 282–289.

#     .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
#     """

#     def __init__(self, num_tags: int, batch_first: bool = False) -> None:
#         if num_tags <= 0:
#             raise ValueError(f'invalid number of tags: {num_tags}')
#         super().__init__()
#         self.num_tags = num_tags
#         self.batch_first = batch_first
#         self.start_transitions = nn.Parameter(torch.empty(num_tags))
#         self.end_transitions = nn.Parameter(torch.empty(num_tags))
#         self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         """Initialize the transition parameters.

#         The parameters will be initialized randomly from a uniform distribution
#         between -0.1 and 0.1.
#         """
#         nn.init.uniform_(self.start_transitions, -0.1, 0.1)
#         nn.init.uniform_(self.end_transitions, -0.1, 0.1)
#         nn.init.uniform_(self.transitions, -0.1, 0.1)

#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}(num_tags={self.num_tags})'

#     def forward(
#             self,
#             emissions: torch.Tensor,
#             tags: torch.LongTensor,
#             mask: Optional[torch.ByteTensor] = None,
#             reduction: str = 'sum',
#     ) -> torch.Tensor:
#         """Compute the conditional log likelihood of a sequence of tags given emission scores.

#         Args:
#             emissions (`~torch.Tensor`): Emission score tensor of size
#                 ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
#                 ``(batch_size, seq_length, num_tags)`` otherwise.
#             tags (`~torch.LongTensor`): Sequence of tags tensor of size
#                 ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
#                 ``(batch_size, seq_length)`` otherwise.
#             mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
#                 if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
#             reduction: Specifies  the reduction to apply to the output:
#                 ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
#                 ``sum``: the output will be summed over batches. ``mean``: the output will be
#                 averaged over batches. ``token_mean``: the output will be averaged over tokens.

#         Returns:
#             `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
#             reduction is ``none``, ``()`` otherwise.
#         """
#         self._validate(emissions, tags=tags, mask=mask)
#         if reduction not in ('none', 'sum', 'mean', 'token_mean'):
#             raise ValueError(f'invalid reduction: {reduction}')
#         if mask is None:
#             mask = torch.ones_like(tags, dtype=torch.uint8)

#         if self.batch_first:
#             emissions = emissions.transpose(0, 1)
#             tags = tags.transpose(0, 1)
#             mask = mask.transpose(0, 1)

#         # shape: (batch_size,)
#         numerator = self._compute_score(emissions, tags, mask)
#         # shape: (batch_size,)
#         denominator = self._compute_normalizer(emissions, mask)
#         # shape: (batch_size,)
#         llh = numerator - denominator

#         if reduction == 'none':
#             return llh
#         if reduction == 'sum':
#             return llh.sum()
#         if reduction == 'mean':
#             return llh.mean()
#         assert reduction == 'token_mean'
#         return llh.sum() / mask.float().sum()

#     def decode(self, emissions: torch.Tensor,
#                mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
#         """Find the most likely tag sequence using Viterbi algorithm.

#         Args:
#             emissions (`~torch.Tensor`): Emission score tensor of size
#                 ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
#                 ``(batch_size, seq_length, num_tags)`` otherwise.
#             mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
#                 if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

#         Returns:
#             List of list containing the best tag sequence for each batch.
#         """
#         self._validate(emissions, mask=mask)
#         if mask is None:
#             mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

#         if self.batch_first:
#             emissions = emissions.transpose(0, 1)
#             mask = mask.transpose(0, 1)

#         return self._viterbi_decode(emissions, mask)

#     def _validate(
#             self,
#             emissions: torch.Tensor,
#             tags: Optional[torch.LongTensor] = None,
#             mask: Optional[torch.ByteTensor] = None) -> None:
#         if emissions.dim() != 3:
#             raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
#         if emissions.size(2) != self.num_tags:
#             raise ValueError(
#                 f'expected last dimension of emissions is {self.num_tags}, '
#                 f'got {emissions.size(2)}')

#         if tags is not None:
#             if emissions.shape[:2] != tags.shape:
#                 raise ValueError(
#                     'the first two dimensions of emissions and tags must match, '
#                     f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

#         if mask is not None:
#             if emissions.shape[:2] != mask.shape:
#                 raise ValueError(
#                     'the first two dimensions of emissions and mask must match, '
#                     f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
#             no_empty_seq = not self.batch_first and mask[0].all()
#             no_empty_seq_bf = self.batch_first and mask[:, 0].all()
#             if not no_empty_seq and not no_empty_seq_bf:
#                 raise ValueError('mask of the first timestep must all be on')

#     def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor) -> torch.Tensor:
#         print("=== _compute_score ===")
#         print("emissions.shape:", emissions.shape)
#         print("tags.shape:", tags.shape)
#         print("mask.shape:", mask.shape)
#         print("emissions.isnan().any():", torch.isnan(emissions).any().item())
#         print("tags.min():", tags.min().item(), "tags.max():", tags.max().item())
#         print("mask.sum():", mask.sum().item())

#         assert emissions.dim() == 3 and tags.dim() == 2
#         assert emissions.shape[:2] == tags.shape
#         assert emissions.size(2) == self.num_tags
#         assert mask.shape == tags.shape
#         assert mask[0].all()

#         seq_length, batch_size = tags.shape
#         mask = mask.float()

#         score = self.start_transitions[tags[0]]
#         print("start_transitions:", self.start_transitions)
#         print("first score:", score)

#         score += emissions[0, torch.arange(batch_size), tags[0]]
#         print("score after first emission:", score)

#         for i in range(1, seq_length):
#             trans = self.transitions[tags[i - 1], tags[i]] * mask[i]
#             emis = emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
#             print(f"step {i} trans.isnan:", torch.isnan(trans).any().item(), "emis.isnan:", torch.isnan(emis).any().item())
#             score += trans
#             score += emis
#             print(f"score step {i}:", score)

#         seq_ends = mask.long().sum(dim=0) - 1
#         print("seq_ends:", seq_ends)

#         last_tags = tags[seq_ends, torch.arange(batch_size)]
#         print("last_tags:", last_tags)

#         score += self.end_transitions[last_tags]
#         print("final score:", score)

#         return score

#     def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
#         print("=== _compute_normalizer ===")
#         print("emissions.shape:", emissions.shape)
#         print("mask.shape:", mask.shape)
#         print("emissions.isnan().any():", torch.isnan(emissions).any().item())
#         print("mask.sum():", mask.sum().item())

#         assert emissions.dim() == 3 and mask.dim() == 2
#         assert emissions.shape[:2] == mask.shape
#         assert emissions.size(2) == self.num_tags
#         assert mask[0].all()

#         seq_length = emissions.size(0)
#         score = self.start_transitions + emissions[0]
#         print("initial score.isnan():", torch.isnan(score).any().item())
        

#         for i in range(1, seq_length):
#             broadcast_score = score.unsqueeze(2)
#             broadcast_emissions = emissions[i].unsqueeze(1)
#             next_score = broadcast_score + self.transitions + broadcast_emissions
#             print(f"step {i} next_score.isnan:", torch.isnan(next_score).any().item())

#             next_score = torch.logsumexp(next_score, dim=1)
#             print(f"step {i} after logsumexp.isnan:", torch.isnan(next_score).any().item())

#             score = torch.where(mask[i].unsqueeze(1), next_score, score)
#             print(f"step {i} score.isnan:", torch.isnan(score).any().item())
            
#             print("transitions.isnan:", torch.isnan(self.transitions).any().item(),
#                 "transitions.min:", self.transitions.min().item(),
#                 "transitions.max:", self.transitions.max().item())
#             print("broadcast_emissions.isnan:", torch.isnan(broadcast_emissions).any().item(),
#                 "broadcast_emissions.min:", broadcast_emissions.min().item(),
#                 "broadcast_emissions.max:", broadcast_emissions.max().item())
#             print("broadcast_score.isnan:", torch.isnan(broadcast_score).any().item(),
#                 "broadcast_score.min:", broadcast_score.min().item(),
#                 "broadcast_score.max:", broadcast_score.max().item())

#         score += self.end_transitions
#         print("score after end_transitions.isnan:", torch.isnan(score).any().item())

#         result = torch.logsumexp(score, dim=1)
#         print("final normalizer.isnan:", torch.isnan(result).any().item())
#         return result


#     def _viterbi_decode(self, emissions: torch.FloatTensor,
#                         mask: torch.ByteTensor) -> List[List[int]]:
#         # emissions: (seq_length, batch_size, num_tags)
#         # mask: (seq_length, batch_size)
#         assert emissions.dim() == 3 and mask.dim() == 2
#         assert emissions.shape[:2] == mask.shape
#         assert emissions.size(2) == self.num_tags
#         assert mask[0].all()

#         seq_length, batch_size = mask.shape

#         # Start transition and first emission
#         # shape: (batch_size, num_tags)
#         score = self.start_transitions + emissions[0]
#         history = []

#         # score is a tensor of size (batch_size, num_tags) where for every batch,
#         # value at column j stores the score of the best tag sequence so far that ends
#         # with tag j
#         # history saves where the best tags candidate transitioned from; this is used
#         # when we trace back the best tag sequence

#         # Viterbi algorithm recursive case: we compute the score of the best tag sequence
#         # for every possible next tag
#         for i in range(1, seq_length):
#             # Broadcast viterbi score for every possible next tag
#             # shape: (batch_size, num_tags, 1)
#             broadcast_score = score.unsqueeze(2)

#             # Broadcast emission score for every possible current tag
#             # shape: (batch_size, 1, num_tags)
#             broadcast_emission = emissions[i].unsqueeze(1)

#             # Compute the score tensor of size (batch_size, num_tags, num_tags) where
#             # for each sample, entry at row i and column j stores the score of the best
#             # tag sequence so far that ends with transitioning from tag i to tag j and emitting
#             # shape: (batch_size, num_tags, num_tags)
#             next_score = broadcast_score + self.transitions + broadcast_emission

#             # Find the maximum score over all possible current tag
#             # shape: (batch_size, num_tags)
#             next_score, indices = next_score.max(dim=1)

#             # Set score to the next score if this timestep is valid (mask == 1)
#             # and save the index that produces the next score
#             # shape: (batch_size, num_tags)
#             score = torch.where(mask[i].unsqueeze(1), next_score, score)
#             history.append(indices)

#         # End transition score
#         # shape: (batch_size, num_tags)
#         score += self.end_transitions

#         # Now, compute the best path for each sample

#         # shape: (batch_size,)
#         seq_ends = mask.long().sum(dim=0) - 1
#         best_tags_list = []

#         for idx in range(batch_size):
#             # Find the tag which maximizes the score at the last timestep; this is our best tag
#             # for the last timestep
#             _, best_last_tag = score[idx].max(dim=0)
#             best_tags = [best_last_tag.item()]

#             # We trace back where the best last tag comes from, append that to our best tag
#             # sequence, and trace it back again, and so on
#             for hist in reversed(history[:seq_ends[idx]]):
#                 best_last_tag = hist[idx][best_tags[-1]]
#                 best_tags.append(best_last_tag.item())

#             # Reverse the order because we start from the last timestep
#             best_tags.reverse()
#             best_tags_list.append(best_tags)

#         return best_tags_list
