from typing import Tuple, Dict, Optional, Literal

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DebertaV2Tokenizer

from ml.visualizer import DynamicEmbeddingVisualizer


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
        
        self._embeddings = [None] * len(self.df)
        self._entropies = [None] * len(self.df)
        
        self.label2idx = label2idx
        
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

        for token, (start, end), label in zip(tokens, offsets, labels):
            if token in tokenizer.all_special_tokens or start == end:
                flush_current()
                continue

            ent_type = None
            if label.startswith("B-"):
                ent_type = label[2:]
                flush_current()
                current = {
                    "start_index": int(start),
                    "end_index": int(end),
                    "entity": f"B-{ent_type}",
                    "word": text[start:end]
                }

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

            elif label == "O":
                if current and current["entity"] == "O" and current["end_index"] == int(start):
                    current["end_index"] = int(end)
                    current["word"] = text[current["start_index"]:end]
                else:
                    flush_current()
                    current = {
                        "start_index": int(start),
                        "end_index": int(end),
                        "entity": "O",
                        "word": text[start:end]
                    }

        flush_current()
        return entities
    
    @staticmethod
    def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropies = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        return entropies
    
    @staticmethod
    def aggregate_word_embeddings(text, emb_full, tokenizer, encoded_inputs):
            tokens = tokenizer.convert_ids_to_tokens(encoded_inputs["input_ids"][0])
            offsets = encoded_inputs["offset_mapping"][0].cpu().numpy()

            words = []
            current_word = None
            current_vecs = []
            word_start, word_end = None, None

            for token, (start, end), vec in zip(tokens, offsets, emb_full):
                if token in tokenizer.all_special_tokens or start == end:
                    continue

                if token.startswith("##"):
                    current_word += text[start:end]
                    current_vecs.append(vec)
                    word_end = end
                else:
                    if current_word is not None:
                        words.append({
                            "word": current_word,
                            "emb": torch.stack(current_vecs).mean(dim=0),
                            "start": word_start,
                            "end": word_end
                        })
                    current_word = text[start:end]
                    word_start, word_end = start, end
                    current_vecs = [vec]

            if current_word is not None:
                words.append({
                    "word": current_word,
                    "emb": torch.stack(current_vecs).mean(dim=0),
                    "start": word_start,
                    "end": word_end
                })

            return words
        
    @staticmethod
    def aggregate_word_entropies(word_embs, full_entropy, encoded_inputs):
        word_entropies = []
        for w in word_embs:
            start, end = w["start"], w["end"]
            sub_tokens_idx = [
                j for j, (s, e) in enumerate(encoded_inputs["offset_mapping"][0].tolist())
                if s >= start and e <= end and s != e
            ]
            if sub_tokens_idx:
                avg_ent = full_entropy[sub_tokens_idx].mean().item()
                word_entropies.append({
                    "word": w["word"],
                    "entropy": avg_ent,
                    "start": w["start"],
                    "end": w["end"]
                })
        return word_entropies
    
    def analyze_with_model(
        self,
        model,
        idx2label,
        batch_size: int = 16,
        device: str = "cpu",
        layer: int = -1,
    ):
        model.eval().to(device)
        loader = DataLoader(self, batch_size=batch_size)

        all_embeddings = []
        all_entropies = []
        all_labels = []
        all_entities = []
        all_correct = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[layer]
                predictions = outputs.predictions
                entropies = self.compute_entropy(outputs.logits)

                for i in range(hidden_states.size(0)):
                    row_idx = batch_idx * loader.batch_size + i
                    text = self.df.loc[row_idx, self.text_label]
                    ann_list = self.df.loc[row_idx, self.target_label]

                    mask = attention_mask[i].bool()

                    full_emb = hidden_states[i]
                    mean_emb = full_emb[mask].mean(dim=0).detach().cpu()
                    cls_emb = full_emb[0].detach().cpu()

                    encoded_inputs = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_offsets_mapping=True,
                        return_token_type_ids=True,
                        return_tensors="pt"
                    ).to(device)

                    word_embs = self.aggregate_word_embeddings(
                        text=text,
                        emb_full=full_emb,
                        tokenizer=self.tokenizer,
                        encoded_inputs=encoded_inputs
                    )

                    full_entropy = entropies[i]
                    mean_entropy = full_entropy[mask].mean().item()
                    cls_entropy = full_entropy[0].item()
                    word_entropies = self.aggregate_word_entropies(
                        word_embs, full_entropy.detach().cpu(), encoded_inputs
                    )

                    all_embeddings.append({
                        "full": full_emb.detach().cpu(),
                        "mean": mean_emb,
                        "cls": cls_emb,
                        "words": word_embs
                    })

                    all_entropies.append({
                        "full": full_entropy.detach().cpu(),
                        "mean": mean_entropy,
                        "cls": cls_entropy,
                        "words": word_entropies
                    })

                    pred_ids = predictions[i].detach().cpu().numpy().tolist()
                    labels = [
                        idx2label.get(pid, "O") if pid != -100 else "O"
                        for pid in pred_ids
                    ]
                    all_labels.append(labels)

                    entities = self.decode_predictions(
                        text=text,
                        predictions=pred_ids,
                        tokenizer=self.tokenizer,
                        idx2label=idx2label,
                        encoded_inputs=encoded_inputs.to("cpu")
                    )
                    all_entities.append(entities)

                    gold_entities_set = {(s, e, l) for (s, e, l) in ann_list}
                    pred_entities_set = {
                        (ent["start_index"], ent["end_index"], ent["entity"])
                        for ent in entities
                    }
                    all_correct.append(gold_entities_set == pred_entities_set)

        self._embeddings = all_embeddings
        self._entropies = all_entropies
        self.df["pred_labels"] = all_labels
        self.df["entities"] = all_entities
        self.df["is_correct"] = all_correct
  
        
    def visualize_embeddings(
        self,
        source: Literal["mean", "cls", "words"] = "mean",
        method: str = "umap",
        n_components: int = 2,
        n_samples: Optional[int] = None,
        random_state: int = 42,
        **kwargs
    ):

        if not hasattr(self, "_embeddings") or self._embeddings is None:
            raise ValueError("Сначала вызовите analyze_with_model(), чтобы получить эмбеддинги.")

        if source in ("mean", "cls"):
            embs = np.stack([emb[source].numpy() for emb in self._embeddings])
            df = self.df.copy()
        elif source == "words":
            words, word_embs, rows = [], [], []
            for i, emb in enumerate(self._embeddings):
                for w in emb["words"]:
                    words.append(w["word"])
                    word_embs.append(w["emb"].numpy())
                    rows.append(i)
            embs = np.stack(word_embs)
            df = pd.DataFrame({"word": words, "row_idx": rows})
        else:
            raise ValueError("source должен быть 'mean', 'cls' или 'words'.")

        if n_samples is not None and n_samples < len(df):
            np.random.seed(random_state)
            idx = np.random.choice(len(df), n_samples, replace=False)
            embs = embs[idx]
            df = df.iloc[idx].reset_index(drop=True)

            if source in ("mean", "cls") and self.labels is not None:
                labels = self.labels[idx]
            else:
                labels = None
        else:
            labels = self.labels if source in ("mean", "cls") else None

        if labels is not None:
            labels = np.array(labels)
            labels = np.where(labels == -100, 0, labels)

        vis = DynamicEmbeddingVisualizer(embs, df, labels=labels)
        vis.reduce_dimensionality(method=method, n_components=n_components)
        vis.visualize(method=method, **kwargs)




