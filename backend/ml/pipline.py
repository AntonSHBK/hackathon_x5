import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.tokenization_utils_base import BatchEncoding

from ml.model import AutoModelForTokenClassificationWithCRF
from ml.dataset import NerDataSet

class NERPipelineCRF:
    def __init__(
        self,
        model_path: str,
        label2idx: dict,
        idx2label: dict,
        device: str = None,
        max_length: int = 128
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.label2idx = label2idx
        self.idx2label = idx2label
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        config.label2id = label2idx
        config.id2label = idx2label

        self.model = AutoModelForTokenClassificationWithCRF.from_pretrained(
            model_path,
            config=config
        )
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        texts: list[str],
        batch_size: int = 64,
        return_word: bool = False
    ) -> list[list[dict]]:
        
        all_results = []
        self.model.eval()

        # батчевая токенизация (fast токенайзер → есть .encodings с word_ids)
        encoded_batch = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )

        # сохраняем encodings и offset_mapping
        encodings = encoded_batch.encodings
        offset_mappings = encoded_batch["offset_mapping"].cpu().numpy()

        # убираем offset_mapping и переносим на девайс
        encoded_batch = {k: v.to(self.device) for k, v in encoded_batch.items() if k != "offset_mapping"}

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_slice = {k: v[i:i+batch_size] for k, v in encoded_batch.items()}
                outputs = self.model(**batch_slice)
                preds = outputs.predictions.cpu().numpy()

                for j, pred_seq in enumerate(preds):
                    idx = i + j
                    encoding = encodings[idx]
                    offsets = offset_mappings[idx]

                    # ✅ формируем BatchEncoding, чтобы decode_predictions мог вызвать .word_ids()
                    encoded_inputs = BatchEncoding(
                        {
                            "input_ids": encoded_batch["input_ids"][idx:idx+1].cpu(),
                            "offset_mapping": torch.tensor([offsets])
                        },
                        encoding=[encoding],
                        tensor_type=None
                    )

                    entities = NerDataSet.decode_predictions(
                        texts[idx],
                        pred_seq,
                        self.tokenizer,
                        self.idx2label,
                        encoded_inputs,
                        return_word=return_word
                    )
                    all_results.append(entities)

        return all_results
