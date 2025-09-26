import torch
from transformers import AutoTokenizer, AutoConfig

from ml.model import AutoModelForTokenClassificationWithCRF
from ml.dataset import NerDataSet

class NERPipelineCRF:
    def __init__(self, model_path: str, label2idx: dict, idx2label: dict, device: str = None, max_length: int = 128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label2idx = label2idx
        self.idx2label = idx2label
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        config.label2id = label2idx
        config.id2label = idx2label

        self.model = AutoModelForTokenClassificationWithCRF.from_pretrained(model_path, config=config)
        self.model.to(self.device)
        self.model.eval()

    def predict_text(self, text: str):
        encoded = NerDataSet.prepare_text(text, self.tokenizer, max_length=self.max_length, device=self.device)

        offset_mapping = encoded.pop("offset_mapping")

        with torch.no_grad():
            outputs = self.model(**encoded)
            preds = outputs.predictions[0]

        encoded["offset_mapping"] = offset_mapping

        entities = NerDataSet.decode_predictions(
            text,
            preds,
            self.tokenizer,
            self.idx2label,
            encoded
        )
        return entities

    def predict_dataset(self, dataset, batch_size: int = 16):
        all_results = []
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = outputs.predictions.cpu().numpy()

                for i, pred_seq in enumerate(preds):
                    text = dataset.df.iloc[len(all_results) + i][dataset.text_label]
                    encoded = NerDataSet.prepare_text(text, self.tokenizer, max_length=self.max_length, device="cpu")
                    entities = NerDataSet.decode_predictions(text, pred_seq, self.tokenizer, self.idx2label, encoded)
                    all_results.append(entities)

        return all_results
    
    def predict(self, texts: list[str], batch_size: int = 64) -> list[list[dict]]: 
        
        all_results = []
        self.model.eval()

        encoded_batch = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )

        offset_mappings = encoded_batch.pop("offset_mapping").cpu().numpy()
        encoded_batch = {k: v.to(self.device) for k, v in encoded_batch.items()}

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_slice = {k: v[i:i+batch_size] for k, v in encoded_batch.items()}
                outputs = self.model(**batch_slice)
                preds = outputs.predictions.cpu().numpy()

                for j, pred_seq in enumerate(preds):
                    idx = i + j
                    encoded_inputs = {
                        "input_ids": encoded_batch["input_ids"][idx:idx+1].cpu(),
                        "offset_mapping": torch.tensor(offset_mappings[idx:idx+1])
                    }
                    entities = NerDataSet.decode_predictions(
                        texts[idx],
                        pred_seq,
                        self.tokenizer,
                        self.idx2label,
                        encoded_inputs
                    )
                    all_results.append(entities)

        return all_results

