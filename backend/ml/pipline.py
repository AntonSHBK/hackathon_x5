import torch
from transformers import AutoTokenizer, AutoConfig

from ml.model import BertForTokenClassificationCRF
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

        self.model = BertForTokenClassificationCRF.from_pretrained(model_path, config=config)
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
