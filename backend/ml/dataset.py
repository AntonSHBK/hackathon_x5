import torch
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DebertaV2Tokenizer
from typing import Tuple, Dict, Optional


class NerDataSet(Dataset):
    def __init__(
        self, df: pd.DataFrame, 
        max_length: int, 
        tokenizer_path: str, 
        label2idx: Dict[str, int],
        cache_dir: str = None, 
        text_label: str = 'sample',
        target_label: str = 'annotation',        
        dtype_input_ids: torch.dtype = torch.long,
        dtype_token_type_ids: torch.dtype = torch.long,
        dtype_attention_mask: torch.dtype = torch.long,
        dtype_labels : torch.dtype = torch.long,
        debug: bool = False,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.max_length = max_length
        self.text_label = text_label
        self.target_label = target_label
        self.debug = debug
        
        self.label2idx = label2idx
        
        # TODO добавить класс для типизации
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=cache_dir,
            use_fast=True,
        )

        self.dtype_input_ids = dtype_input_ids
        self.dtype_token_type_ids = dtype_token_type_ids
        self.dtype_attention_mask = dtype_attention_mask
        self.dtype_labels  = dtype_labels 

        self.input_ids, self.token_type_ids, self.attention_mask, self.labels = self.tokenize_data()

    def tokenize_data(self):
        input_ids, token_type_ids, attention_mask, labels = [], [], [], []
        tokens_ids_debug, tokens_text_debug, labels_debug = [], [], []

        for _, row in tqdm(
            self.df.iterrows(),
            total=len(self.df),
            desc="Tokenizing data",
            ncols=100
        ):
            text = row[self.text_label]
            ann_list = row[self.target_label]

            if isinstance(ann_list, str):
                ann_list = eval(ann_list)

            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
                return_token_type_ids=True,
            )

            offsets = encoded["offset_mapping"]
            seq_labels = ["O"] * len(offsets)

            for start, end, ent_label in ann_list:
                inside = False
                for i, (tok_start, tok_end) in enumerate(offsets):
                    if tok_start >= end:
                        break
                    if tok_end <= start:
                        continue

                    if not inside:
                        seq_labels[i] = ent_label
                        inside = True
                    else:
                        if ent_label.startswith("B-"):
                            seq_labels[i] = "I-" + ent_label.split("-", 1)[1]
                        else:
                            seq_labels[i] = ent_label

            label_ids = []
            for i, label in enumerate(seq_labels):
                if encoded["attention_mask"][i] == 0:
                    label_ids.append(-100)
                else:
                    label_ids.append(self.label2idx.get(label, self.label2idx["O"]))

            input_ids.append(torch.tensor(encoded["input_ids"], dtype=self.dtype_input_ids))
            token_type_ids.append(torch.tensor(encoded.get("token_type_ids", [0]*len(label_ids)), dtype=self.dtype_token_type_ids))
            attention_mask.append(torch.tensor(encoded["attention_mask"], dtype=self.dtype_attention_mask))
            labels.append(torch.tensor(label_ids, dtype=self.dtype_labels))

            if self.debug:
                tokens_ids_debug.append(encoded["input_ids"])
                tokens_text_debug.append(self.tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
                labels_debug.append(seq_labels)

        input_ids = torch.stack(input_ids)
        token_type_ids = torch.stack(token_type_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)

        if self.debug:
            self.df["tokens_ids_debug"] = tokens_ids_debug
            self.df["tokens_text_debug"] = tokens_text_debug
            self.df["labels_debug"] = labels_debug

        return input_ids, token_type_ids, attention_mask, labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": self.labels[idx],
        }
        
    def plot_token_length_distribution(self):
        if not self.debug:
            raise ValueError("Для построения графика необходимо включить debug=True при инициализации.")

        token_lengths = []
        special_ids = set(self.tokenizer.all_special_ids)

        for token_ids in self.df["tokens_ids_debug"]:
            filtered_tokens = [tid for tid in token_ids if tid not in special_ids]
            token_lengths.append(len(filtered_tokens))

        plt.figure(figsize=(10, 6))
        plt.hist(token_lengths, bins=30, alpha=0.7, edgecolor="black")
        plt.xlabel("Длина текста (количество токенов)")
        plt.ylabel("Частота")
        plt.title("Распределение длин текстов в токенах")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
        
    @staticmethod
    def prepare_text(text: str, tokenizer, max_length: int = 128, device: str = "cpu"):
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded

    @staticmethod
    def decode_predictions(
        text: str,
        predictions: torch.Tensor,
        tokenizer: AutoTokenizer,
        idx2label: Dict[int, str],
        encoded_inputs: Dict[str, torch.Tensor]
    ) -> list[dict]:
        tokens = tokenizer.convert_ids_to_tokens(encoded_inputs["input_ids"][0])
        offsets = encoded_inputs["offset_mapping"][0].cpu().numpy()
        pred_ids = [int(p) for p in predictions]
        tokens = tokens[:len(pred_ids)]
        offsets = offsets[:len(pred_ids)]

        labels = [idx2label.get(pid, "O") for pid in pred_ids]

        entities = []
        current = None

        def flush_current():
            nonlocal current
            if current:
                entities.append(current)
                current = None

        last_emitted_entity_type = None
        last_emitted_was_B = False

        for token, (start, end), label in zip(tokens, offsets, labels):
            if token in tokenizer.all_special_tokens or label == "O" or start == end:
                flush_current()
                last_emitted_entity_type = None
                last_emitted_was_B = False
                continue

            if label.startswith("B-"):
                ent_type = label[2:]
                flush_current()
                current = {
                    "start_index": int(start),
                    "end_index": int(end),
                    "entity": f"B-{ent_type}",
                    "word": text[start:end]
                }
                last_emitted_entity_type = ent_type
                last_emitted_was_B = True

            elif label.startswith("I-"):
                ent_type = label[2:]
                if current and current["end_index"] == int(start) and current["entity"].endswith(ent_type):
                    current["end_index"] = int(end)
                    current["word"] = text[current["start_index"]:end]
                else:
                    flush_current()

                    current = {
                        "start_index": int(start),
                        "end_index": int(end),
                        "entity": f"I-{ent_type}",
                        "word": text[start:end]
                    }
                    last_emitted_entity_type = ent_type
                    last_emitted_was_B = False

            else:
                flush_current()
                last_emitted_entity_type = None
                last_emitted_was_B = False

        flush_current()
        return entities
